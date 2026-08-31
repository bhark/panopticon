"""Which tools an agent can reach, and how that shifts under it."""

from __future__ import annotations

from pathlib import Path

from panopticon.model import Agent, Harness, Situation
from panopticon.tools import tools_for
from panopticon.tools.board import CANCEL_FINALIZE, FINALIZE_TASK
from panopticon.tools.comms import SEND_DM, SHOUT, VIEW_SHOUTBOARD
from panopticon.tools.jury import JOIN_JURY, LIST_JURY
from panopticon.tools.knowledge import SUBMIT_TRUTH
from panopticon.tools.lifecycle import REJOIN, RELIEVE_SELF
from panopticon.tools.workspace import BASH, EDIT_FILE, READ_FILE, WRITE_FILE
from tests.fakes import FakeHarness, seat_everyone


def harness(tmp_path: Path) -> FakeHarness:
    return FakeHarness(tmp_path / "repo")


def names(agent: Agent, h: Harness) -> list[str]:
    return [spec.name for spec in tools_for(agent, h)]


def test_no_situation_offers_the_same_tool_twice(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    h.kb.submit("Bo", "claim", "proof")
    h.force_ending = True
    for situation in Situation:
        agent.situation = situation
        agent.seen_kb = True
        offered = names(agent, h)
        assert len(offered) == len(set(offered)), situation


def test_submitting_a_truth_is_locked_until_the_knowledge_base_is_read(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    assert SUBMIT_TRUTH not in names(agent, h)

    agent.seen_kb = True
    assert SUBMIT_TRUTH in names(agent, h)

    # having read it does not follow you onto the jury, where you judge rather than assert
    agent.situation = Situation.JURY
    assert SUBMIT_TRUTH not in names(agent, h)


def test_jury_tools_appear_only_for_a_submission_you_may_judge(tmp_path):
    h = harness(tmp_path)
    ada, bo = h.agents["Ada"], h.agents["Bo"]
    assert JOIN_JURY not in names(ada, h) and LIST_JURY not in names(ada, h)

    submission = h.kb.submit("Ada", "claim", "proof")
    assert JOIN_JURY not in names(ada, h)  # never your own
    assert LIST_JURY not in names(ada, h)
    assert JOIN_JURY in names(bo, h) and LIST_JURY in names(bo, h)

    # a second submission from someone else opens it up to Ada too
    h.kb.submit("Bo", "other claim", "proof")
    assert JOIN_JURY in names(ada, h)

    # already seated: no second jury, and the listing is still offered once
    bo.situation, bo.submission_id = Situation.JURY, submission.id
    assert JOIN_JURY not in names(bo, h)
    assert names(bo, h).count(LIST_JURY) == 1


def test_a_juror_can_look_but_not_write(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    agent.situation = Situation.JURY
    offered = names(agent, h)
    assert BASH in offered and READ_FILE in offered
    assert WRITE_FILE not in offered and EDIT_FILE not in offered


def test_a_released_agent_keeps_the_messages_and_the_way_back_and_nothing_else(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    agent.situation, agent.seen_kb = Situation.RELEASED, True
    h.kb.submit("Bo", "claim", "proof")  # jury work waiting is not theirs to take
    assert set(names(agent, h)) == {"wait", SEND_DM, SHOUT, VIEW_SHOUTBOARD, REJOIN}


async def test_finalizing_swaps_the_tool_for_its_undo(tmp_path):
    h = harness(tmp_path)
    task = await seat_everyone(h, {"Ada": "dev", "Bo": "reviewer"})
    ada, bo = h.agents["Ada"], h.agents["Bo"]
    assert FINALIZE_TASK in names(ada, h) and CANCEL_FINALIZE not in names(ada, h)

    h.board.finalize("Ada", task.id, "done", "green")
    assert CANCEL_FINALIZE in names(ada, h) and FINALIZE_TASK not in names(ada, h)
    assert FINALIZE_TASK in names(bo, h)  # only the seat that finalized is swapped

    h.board.cancel_finalize("Ada", task.id)
    assert FINALIZE_TASK in names(ada, h) and CANCEL_FINALIZE not in names(ada, h)


def test_relieving_yourself_needs_the_human_to_have_ended_the_session(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    assert RELIEVE_SELF not in names(agent, h)

    h.force_ending = True
    assert RELIEVE_SELF in names(agent, h)

    for situation in (Situation.RELIEVED, Situation.DEAD):
        agent.situation = situation
        assert names(agent, h) == []  # a finished agent gets nothing, force end or not
