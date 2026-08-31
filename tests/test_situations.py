"""Which tools an agent can reach, and how that shifts under it."""

from __future__ import annotations

from pathlib import Path

from panopticon import situations
from panopticon.model import Situation
from panopticon.situations import (
    BASH,
    CANCEL_FINALIZE,
    EDIT_FILE,
    FINALIZE_TASK,
    JOIN_JURY,
    LIST_JURY,
    READ_FILE,
    RELIEVE_SELF,
    SUBMIT_TRUTH,
    WRITE_FILE,
    tool_names,
    tools_for,
)
from panopticon.tools import REGISTRY
from tests.fakes import FakeHarness, seat_everyone


def harness(tmp_path: Path) -> FakeHarness:
    return FakeHarness(tmp_path / "repo")


def test_the_table_and_the_registry_name_the_same_tools():
    named = {v for k, v in vars(situations).items() if k.isupper() and isinstance(v, str)}
    assert named == set(REGISTRY)


def test_no_situation_offers_the_same_tool_twice(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    h.kb.submit("Bo", "claim", "proof")
    h.force_ending = True
    for situation in Situation:
        agent.situation = situation
        agent.seen_kb = True
        names = tool_names(agent, h)
        assert len(names) == len(set(names)), situation


def test_submitting_a_truth_is_locked_until_the_knowledge_base_is_read(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    assert SUBMIT_TRUTH not in tool_names(agent, h)

    agent.seen_kb = True
    assert SUBMIT_TRUTH in tool_names(agent, h)

    # having read it does not follow you onto the jury, where you judge rather than assert
    agent.situation = Situation.JURY
    assert SUBMIT_TRUTH not in tool_names(agent, h)


def test_jury_tools_appear_only_for_a_submission_you_may_judge(tmp_path):
    h = harness(tmp_path)
    ada, bo = h.agents["Ada"], h.agents["Bo"]
    assert JOIN_JURY not in tool_names(ada, h) and LIST_JURY not in tool_names(ada, h)

    submission = h.kb.submit("Ada", "claim", "proof")
    assert JOIN_JURY not in tool_names(ada, h)  # never your own
    assert LIST_JURY not in tool_names(ada, h)
    assert JOIN_JURY in tool_names(bo, h) and LIST_JURY in tool_names(bo, h)

    # a second submission from someone else opens it up to Ada too
    h.kb.submit("Bo", "other claim", "proof")
    assert JOIN_JURY in tool_names(ada, h)

    # already seated: no second jury, and the listing is offered once by the jury table itself
    bo.situation, bo.submission_id = Situation.JURY, submission.id
    assert JOIN_JURY not in tool_names(bo, h)
    assert tool_names(bo, h).count(LIST_JURY) == 1


def test_a_juror_can_look_but_not_write(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    agent.situation = Situation.JURY
    names = tool_names(agent, h)
    assert BASH in names and READ_FILE in names
    assert WRITE_FILE not in names and EDIT_FILE not in names


async def test_finalizing_swaps_the_tool_for_its_undo(tmp_path):
    h = harness(tmp_path)
    task = await seat_everyone(h, {"Ada": "dev", "Bo": "reviewer"})
    ada, bo = h.agents["Ada"], h.agents["Bo"]
    assert FINALIZE_TASK in tool_names(ada, h) and CANCEL_FINALIZE not in tool_names(ada, h)

    h.board.finalize("Ada", task.id, "done", "green")
    assert CANCEL_FINALIZE in tool_names(ada, h) and FINALIZE_TASK not in tool_names(ada, h)
    assert FINALIZE_TASK in tool_names(bo, h)  # only the seat that finalized is swapped

    h.board.cancel_finalize("Ada", task.id)
    assert FINALIZE_TASK in tool_names(ada, h) and CANCEL_FINALIZE not in tool_names(ada, h)


def test_relieving_yourself_needs_the_human_to_have_ended_the_session(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    assert RELIEVE_SELF not in tool_names(agent, h)

    h.force_ending = True
    assert RELIEVE_SELF in tool_names(agent, h)

    for situation in (Situation.RELIEVED, Situation.DEAD):
        agent.situation = situation
        assert tool_names(agent, h) == []  # a finished agent gets nothing, force end or not


def test_tools_for_resolves_the_names_to_specs(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    specs = tools_for(agent, h)
    assert [s.name for s in specs] == tool_names(agent, h)
