"""What the tools do when the situation is awkward."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from panopticon.model import Action, Entry, Situation, ToolCtx
from panopticon.tools import dispatch, tools_for
from panopticon.tools.board import ASSIGN_SELF, CANCEL_FINALIZE, FINALIZE_TASK, UNASSIGN_SELF
from panopticon.tools.comms import SEND_DM
from panopticon.tools.jury import JOIN_JURY, SUBMIT_VERDICT
from panopticon.tools.lifecycle import RELIEVE_SELF, VOTE_GOAL_REACHED
from panopticon.tools.wait import WAIT
from panopticon.tools.workspace import BASH, EDIT_FILE, READ_FILE, WRITE_FILE
from tests.fakes import FakeHarness, seat_everyone


def tool_names(agent, harness: FakeHarness) -> list[str]:
    return [spec.name for spec in tools_for(agent, harness)]


async def run(harness: FakeHarness, name: str, tool: str, **args):
    """Dispatch as the agent would, through the tools its situation actually allows."""
    agent = harness.agents[name]
    return await dispatch(ToolCtx(agent, harness), Action(tool, args), tool_names(agent, harness))


def harness(tmp_path: Path) -> FakeHarness:
    return FakeHarness(tmp_path / "repo")


# dispatch and argument validation


async def test_dispatch_refuses_a_real_tool_the_situation_does_not_offer(tmp_path):
    h = harness(tmp_path)
    result = await run(h, "Ada", BASH, command="ls")  # Ada is idle, no workspace
    assert not result.ok
    assert "not available to you right now" in result.text
    assert WAIT in result.text  # the refusal names what it can use instead


async def test_int_arguments_are_coerced_from_strings_but_not_from_nonsense(tmp_path):
    h = harness(tmp_path)
    assert (await run(h, "Ada", WAIT, minutes="5")).ok
    assert h.agents["Ada"].wake_at is not None

    bad = await run(h, "Ada", WAIT, minutes="soon")
    assert not bad.ok and "expected int" in bad.text

    unknown = await run(h, "Ada", WAIT, hours=2)
    assert not unknown.ok and "unknown argument" in unknown.text

    missing = await run(h, "Ada", SEND_DM, to="Bo")
    assert not missing.ok and "body: missing" in missing.text


async def test_a_handler_that_raises_comes_back_as_a_failure_not_a_crash(tmp_path):
    h = harness(tmp_path)
    h.board.render = lambda: 1 / 0  # type: ignore[method-assign]
    result = await run(h, "Ada", "view_task_board")
    assert not result.ok and "ZeroDivisionError" in result.text


# workspace confinement


async def workspace_agent(tmp_path) -> tuple[FakeHarness, Path]:
    h = harness(tmp_path)
    task = await seat_everyone(h, {"Ada": "dev"})
    return h, h.worktrees.path_for(task.id)


async def test_file_tools_allow_a_nested_path_inside_the_worktree(tmp_path):
    h, root = await workspace_agent(tmp_path)
    (root / "src" / "deep").mkdir(parents=True)
    (root / "src" / "deep" / "ok.py").write_text("value = 1\n")

    result = await run(h, "Ada", READ_FILE, path="src/deep/ok.py")
    assert result.ok and "value = 1" in result.text


async def test_file_tools_reject_every_way_out_of_the_worktree(tmp_path):
    h, root = await workspace_agent(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("not yours")
    (root / "escape").symlink_to(outside)

    for path in ("../secret.txt", str(outside), "escape", "src/../../../secret.txt", "~/.ssh/id"):
        read = await run(h, "Ada", READ_FILE, path=path)
        assert not read.ok, path
        assert "outside" in read.text, path

        write = await run(h, "Ada", WRITE_FILE, path=path, content="owned")
        assert not write.ok, path

    assert outside.read_text() == "not yours"  # the symlink write did not follow through


async def test_the_shell_runs_in_the_worktree(tmp_path):
    h, root = await workspace_agent(tmp_path)
    result = await run(h, "Ada", BASH, command="pwd")
    assert result.ok and str(root) in result.text


async def test_the_shell_reports_the_exit_code_and_both_streams(tmp_path):
    h, _ = await workspace_agent(tmp_path)
    result = await run(h, "Ada", BASH, command="printf onstdout; printf onstderr >&2; exit 3")
    assert not result.ok
    assert "exit 3" in result.text
    assert "onstdout" in result.text and "onstderr" in result.text


async def test_long_shell_output_is_cut_in_the_middle_with_a_marker(tmp_path):
    h, _ = await workspace_agent(tmp_path)
    result = await run(h, "Ada", BASH, command=f"{sys.executable} -c \"print('x' * 40000)\"")
    assert result.ok
    assert "characters cut from the middle" in result.text
    assert len(result.text) < 20_000


async def test_a_hanging_command_is_killed_and_says_so(tmp_path):
    h, _ = await workspace_agent(tmp_path)
    result = await run(h, "Ada", BASH, command="sleep 30", timeout=1)
    assert not result.ok and "killed after 1s" in result.text


# edit_file


async def test_edit_file_refuses_an_ambiguous_match_and_leaves_the_file_alone(tmp_path):
    h, root = await workspace_agent(tmp_path)
    target = root / "conf.py"
    target.write_text("debug = True\nname = 'x'\ndebug = True\n")

    result = await run(h, "Ada", EDIT_FILE, path="conf.py", old="debug = True", new="debug = False")
    assert not result.ok
    assert "appears 2 times" in result.text
    assert target.read_text() == "debug = True\nname = 'x'\ndebug = True\n"


async def test_edit_file_refuses_a_missing_match_and_a_no_op(tmp_path):
    h, root = await workspace_agent(tmp_path)
    (root / "conf.py").write_text("debug = True\n")

    missing = await run(h, "Ada", EDIT_FILE, path="conf.py", old="verbose = True", new="x")
    assert not missing.ok and "does not appear" in missing.text

    noop = await run(h, "Ada", EDIT_FILE, path="conf.py", old="debug = True", new="debug = True")
    assert not noop.ok and "change nothing" in noop.text


async def test_edit_file_applies_a_unique_match(tmp_path):
    h, root = await workspace_agent(tmp_path)
    target = root / "conf.py"
    target.write_text("debug = True\nname = 'x'\ndebug = True\n")

    result = await run(h, "Ada", EDIT_FILE, path="conf.py", old="name = 'x'", new="name = 'y'")
    assert result.ok
    assert target.read_text() == "debug = True\nname = 'y'\ndebug = True\n"


# the task board, end to end


async def test_the_last_seat_starts_the_task_and_clears_every_holder(tmp_path):
    h = harness(tmp_path)
    await run(
        h,
        "Ada",
        "create_task",
        title="fix flake",
        description="green suite",
        roles=["dev", "reviewer"],
    )
    task_id = next(iter(h.board.tasks))

    first = await run(h, "Ada", ASSIGN_SELF, task_id=task_id, role="dev")
    assert first.ok and "reviewer" in first.text
    assert h.agents["Ada"].situation is Situation.WAITING_FOR_SEATS
    assert not h.launched
    assert not any("took the dev seat" in m for m in h.messages_for("Bo"))  # nobody to tell yet

    h.agents["Ada"].entries.append(Entry("note", "before"))
    assert (await run(h, "Bo", ASSIGN_SELF, task_id=task_id, role="reviewer")).ok
    assert any("took the reviewer seat" in m for m in h.messages_for("Ada"))

    assert h.launched and h.board.tasks[task_id].worktree
    assert h.agents["Ada"].situation is Situation.ON_TASK
    assert h.agents["Bo"].situation is Situation.ON_TASK
    assert h.agents["Ada"].entries == []  # context cleared on the way in
    preprompt = h.preprompt_for("Ada")
    assert "Bo as reviewer" in preprompt
    assert str(h.worktrees.path_for(task_id)) in preprompt


async def test_a_second_seat_is_refused_while_you_hold_one(tmp_path):
    h = harness(tmp_path)
    task = h.board.create("Ada", "one", "d", ["dev", "reviewer"])
    other = h.board.create("Ada", "two", "d", ["dev"])
    assert (await run(h, "Ada", ASSIGN_SELF, task_id=task.id, role="dev")).ok

    # waiting for seats does not offer it, and the handler refuses it even so
    assert ASSIGN_SELF not in tool_names(h.agents["Ada"], h)
    result = await dispatch(
        ToolCtx(h.agents["Ada"], h),
        Action(ASSIGN_SELF, {"task_id": other.id, "role": "dev"}),
        [ASSIGN_SELF],
    )
    assert not result.ok and task.id in result.text


async def test_a_task_ends_only_when_every_seat_finalizes(tmp_path):
    h = harness(tmp_path)
    task = await seat_everyone(h, {"Ada": "dev", "Bo": "reviewer"})

    first = await run(
        h, "Ada", FINALIZE_TASK, task_id=task.id, reason="tests pass", conclusion="suite is green"
    )
    assert first.ok and "reviewer" in first.text
    assert not h.closed
    assert any("Still to finalize: reviewer" in m for m in h.messages_for("Bo"))
    assert CANCEL_FINALIZE in tool_names(h.agents["Ada"], h)
    assert FINALIZE_TASK not in tool_names(h.agents["Ada"], h)
    assert FINALIZE_TASK in tool_names(h.agents["Bo"], h)

    assert (await run(h, "Ada", CANCEL_FINALIZE, task_id=task.id)).ok
    assert FINALIZE_TASK in tool_names(h.agents["Ada"], h)
    assert not h.closed

    await run(h, "Ada", FINALIZE_TASK, task_id=task.id, reason="r", conclusion="c")
    last = await run(h, "Bo", FINALIZE_TASK, task_id=task.id, reason="r", conclusion="c")
    assert last.ok
    assert h.closed == [task]
    assert h.agents["Ada"].situation is Situation.IDLE
    assert h.agents["Bo"].situation is Situation.IDLE


async def test_leaving_a_running_task_goes_through_the_harness(tmp_path):
    h = harness(tmp_path)
    task = await seat_everyone(h, {"Ada": "dev", "Bo": "reviewer"})
    h.agents["Ada"].entries.append(Entry("note", "before"))

    assert (await run(h, "Ada", UNASSIGN_SELF, task_id=task.id)).ok
    assert h.left == [("Ada", task.id, "unassigned themselves")]
    assert task.seat_of("Ada") is None
    assert h.agents["Ada"].situation is Situation.IDLE
    assert h.agents["Ada"].task_id is None
    assert h.agents["Ada"].entries == []
    assert h.agents["Bo"].situation is Situation.WAITING_FOR_SEATS  # the task stopped under Bo
    assert any("unassigned themselves" in m for m in h.messages_for("Bo"))


async def test_unassigning_a_seat_you_do_not_hold_never_reaches_the_harness(tmp_path):
    h = harness(tmp_path)
    task = await seat_everyone(h, {"Ada": "dev"})

    result = await dispatch(
        ToolCtx(h.agents["Bo"], h), Action(UNASSIGN_SELF, {"task_id": task.id}), [UNASSIGN_SELF]
    )
    assert not result.ok and "no seat" in result.text

    gone = await dispatch(
        ToolCtx(h.agents["Ada"], h), Action(UNASSIGN_SELF, {"task_id": "T99"}), [UNASSIGN_SELF]
    )
    assert not gone.ok and "no task T99" in gone.text
    assert not h.left


# comms


async def test_direct_messages_find_the_agent_however_it_was_typed(tmp_path):
    h = harness(tmp_path)
    assert (await run(h, "Ada", SEND_DM, to="bo", body="worktree is dirty")).ok
    assert h.messages_for("Bo") == ["Ada wrote to you: worktree is dirty"]

    unknown = await run(h, "Ada", SEND_DM, to="Zed", body="hi")
    assert not unknown.ok and "Bo" in unknown.text and "Cy" in unknown.text

    assert not (await run(h, "Ada", SEND_DM, to="Ada", body="hi")).ok


# jury


async def test_you_cannot_judge_your_own_submission_but_can_judge_another(tmp_path):
    h = harness(tmp_path)
    h.agents["Ada"].seen_kb = True
    assert (
        await run(h, "Ada", "submit_truth", title="pytest needs -p no:randomly", body="proof")
    ).ok
    submission_id = next(iter(h.kb.pending))

    # the situation hides it, and the handler refuses it anyway
    assert JOIN_JURY not in tool_names(h.agents["Ada"], h)
    mine = await dispatch(
        ToolCtx(h.agents["Ada"], h),
        Action(JOIN_JURY, {"submission_id": submission_id}),
        [JOIN_JURY],
    )
    assert not mine.ok and "your own statement" in mine.text

    h.agents["Bo"].entries.append(Entry("note", "before"))
    assert (await run(h, "Bo", JOIN_JURY, submission_id=submission_id)).ok
    assert h.agents["Bo"].situation is Situation.JURY
    assert h.agents["Bo"].submission_id == submission_id
    assert h.agents["Bo"].entries == []
    assert "pytest needs -p no:randomly" in h.preprompt_for("Bo")


async def test_a_restate_verdict_without_a_restatement_is_refused(tmp_path):
    h = harness(tmp_path)
    submission = h.kb.submit("Ada", "claim", "proof")
    await run(h, "Bo", JOIN_JURY, submission_id=submission.id)

    result = await run(h, "Bo", SUBMIT_VERDICT, verdict="restate", reasoning="too broad")
    assert not result.ok and "restated_title" in result.text
    assert h.agents["Bo"].situation is Situation.JURY  # still on duty

    bad_call = await run(h, "Bo", SUBMIT_VERDICT, verdict="maybe", reasoning="unsure")
    assert not bad_call.ok and "true, false, restate" in bad_call.text


async def test_a_verdict_ends_jury_duty_and_tells_the_submitter(tmp_path):
    h = harness(tmp_path)
    submission = h.kb.submit("Ada", "claim", "proof")
    await run(h, "Bo", JOIN_JURY, submission_id=submission.id)

    result = await run(h, "Bo", SUBMIT_VERDICT, verdict="false", reasoning="the file is gone")
    assert result.ok and "rejected" in result.text
    assert h.agents["Bo"].situation is Situation.IDLE
    assert h.agents["Bo"].submission_id is None
    assert any("rejected by Bo" in m for m in h.messages_for("Ada"))


async def test_an_accepted_verdict_is_told_to_everyone(tmp_path):
    h = harness(tmp_path)
    submission = h.kb.submit("Ada", "the suite needs TZ=UTC", "proof")
    for juror in ("Bo", "Cy"):  # a truth needs two 'true' verdicts to land
        await run(h, juror, JOIN_JURY, submission_id=submission.id)
        await run(h, juror, SUBMIT_VERDICT, verdict="true", reasoning="checked")

    for name in ("Ada", "Bo", "Cy"):
        assert any("entered the knowledge base" in m for m in h.messages_for(name)), name


# lifecycle


async def test_voting_the_goal_reached_is_refused_while_you_hold_a_seat(tmp_path):
    h = harness(tmp_path)
    task = await seat_everyone(h, {"Ada": "dev"})
    h.agents["Ada"].situation = Situation.IDLE  # off task in situation, seat not released

    result = await run(h, "Ada", VOTE_GOAL_REACHED, note="looks done")
    assert not result.ok and task.id in result.text
    assert not h.agents["Ada"].voted_goal_reached


async def test_voting_releases_you_tells_the_others_and_triggers_a_tally(tmp_path):
    h = harness(tmp_path)
    result = await run(h, "Ada", VOTE_GOAL_REACHED, note="suite is green")
    assert result.ok and "1 of 3" in result.text
    assert h.agents["Ada"].situation is Situation.RELEASED
    assert h.tallies == 1
    assert any("voted the goal reached" in m for m in h.messages_for("Bo"))
    assert not h.messages_for("Ada")

    assert (await run(h, "Ada", "rejoin")).ok
    assert h.agents["Ada"].situation is Situation.IDLE
    assert not h.agents["Ada"].voted_goal_reached


async def test_relieve_self_is_unreachable_until_the_human_ends_the_session(tmp_path):
    h = harness(tmp_path)
    assert RELIEVE_SELF not in tool_names(h.agents["Ada"], h)

    h.force_ending = True
    assert (await run(h, "Ada", RELIEVE_SELF)).ok
    assert h.agents["Ada"].situation is Situation.RELIEVED
    assert h.retired == ["Ada"]


async def test_the_closer_archives_the_task_and_retires(tmp_path):
    h = harness(tmp_path)
    task = h.board.create("Ada", "fix flake", "d", ["dev"])
    closer = h.agents["Cy"]
    closer.transient, closer.situation, closer.task_id = True, Situation.CLOSING_TASK, task.id

    result = await run(h, "Cy", "mark_integration_done", summary="merged, worktree removed")
    assert result.ok
    assert task.archived_at and task.outcome == "merged, worktree removed"
    assert h.retired == ["Cy"]
    assert closer.situation is Situation.RELIEVED


@pytest.mark.asyncio
async def test_waiting_forever_is_capped_while_you_hold_a_task_seat(tmp_path):
    """A live run stalled on this: an agent on a running task waited indefinitely, and
    nothing was bound to ever wake it, so the task sat there with nobody working it."""
    h = harness(tmp_path)
    idle, seated = h.agents["Ada"], h.agents["Bo"]
    seated.situation = Situation.ON_TASK

    assert (await run(h, "Ada", WAIT)).ok
    assert idle.wake_at == math.inf

    for situation in (Situation.ON_TASK, Situation.CLOSING_TASK, Situation.JURY):
        seated.situation = situation
        seated.wake_at = None
        result = await run(h, "Bo", WAIT)
        assert result.ok and "hand it back" in result.text
        assert seated.wake_at is not None and seated.wake_at != math.inf
