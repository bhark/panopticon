"""The system prompt has to be stable, and the preprompt has to carry the facts."""

from __future__ import annotations

from pathlib import Path

from panopticon.model import Agent, Entry, QueueItem, Situation
from panopticon.prompts import build_system_prompt, situation_preprompt
from panopticon.situations import tools_for
from tests.fakes import FakeHarness, seat_everyone


def harness(tmp_path: Path) -> FakeHarness:
    return FakeHarness(tmp_path / "repo")


def prompt(h: FakeHarness, name: str) -> str:
    agent = h.agents[name]
    return build_system_prompt(agent, h, tools_for(agent, h))


def test_the_system_prompt_is_byte_identical_while_only_the_transcript_moves(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    before = prompt(h, "Ada")

    agent.turns += 7
    agent.entries.append(Entry("action", "wait"))
    agent.entries.append(Entry("result", "Waiting."))
    agent.inbox.append(QueueItem("dm", "Bo wrote to you: hurry up"))
    agent.usage.input_tokens += 4321
    agent.usage.cost_usd += 0.12
    agent.wake_at = 1_700_000_000.0
    agent.last_action = "wait"
    h.bus.shout("Cy", "the build is broken")
    h.board.create("Bo", "some task", "d", ["dev"])

    assert prompt(h, "Ada") == before


def test_a_closer_spawning_does_not_disturb_everyone_elses_prompt(tmp_path):
    h = harness(tmp_path)
    before = prompt(h, "Ada")
    h.agents["Zeb"] = Agent(name="Zeb", provider="fake", transient=True)
    assert prompt(h, "Ada") == before
    assert "one of 3 agents" in before


def test_the_prompt_moves_when_the_situation_or_the_ending_does(tmp_path):
    h = harness(tmp_path)
    idle = prompt(h, "Ada")

    h.agents["Ada"].situation = Situation.JURY
    assert prompt(h, "Ada") != idle

    h.agents["Ada"].situation = Situation.IDLE
    h.force_ending = True
    ending = prompt(h, "Ada")
    assert ending != idle
    assert "relieve yourself" in ending


def test_the_prompt_carries_the_goal_the_identity_and_every_tool_with_its_arguments(tmp_path):
    h = harness(tmp_path)
    text = prompt(h, "Ada")
    assert h.goal in text
    assert "You are Ada." in text
    for spec in tools_for(h.agents["Ada"], h):
        assert f"## {spec.name}" in text
        for key, arg in spec.args.items():
            need = "required" if arg.required else "optional"
            assert f"- {key} ({arg.type}, {need})" in text


async def test_the_on_task_preprompt_says_where_the_work_lives_and_who_is_on_it(tmp_path):
    h = harness(tmp_path)
    task = await seat_everyone(h, {"Ada": "dev", "Bo": "reviewer"}, title="fix the flake")
    text = situation_preprompt(h.agents["Ada"], h, "", Situation.ON_TASK)

    assert task.id in text and "fix the flake" in text
    assert "you are on it as dev" in text.lower()
    assert str(h.worktrees.path_for(task.id)) in text
    assert "Bo as reviewer" in text
    assert "Ada as dev" not in text  # you are not listed among your own company


def test_the_jury_preprompt_states_the_claim_and_the_read_only_terms(tmp_path):
    h = harness(tmp_path)
    submission = h.kb.submit("Bo", "the suite needs TZ=UTC", "proof: it fails at 23:30 local")
    agent = h.agents["Ada"]
    agent.submission_id = submission.id

    text = situation_preprompt(agent, h, "", Situation.JURY)
    assert "the suite needs TZ=UTC" in text
    assert "proof: it fails at 23:30 local" in text
    assert "Bo" in text
    assert "change nothing" in text


def test_a_preprompt_for_vanished_state_degrades_instead_of_raising(tmp_path):
    h = harness(tmp_path)
    agent = h.agents["Ada"]
    agent.task_id, agent.submission_id = "T99", "S99"

    assert situation_preprompt(agent, h, "", Situation.ON_TASK)
    assert situation_preprompt(agent, h, "", Situation.JURY)
    assert situation_preprompt(agent, h, "", Situation.CLOSING_TASK)


def test_the_idle_preprompt_leads_with_what_just_happened(tmp_path):
    h = harness(tmp_path)
    note = "You finished work on T1, and it sits in the worktree wt-T1."
    text = situation_preprompt(h.agents["Ada"], h, note, Situation.IDLE)
    assert text.startswith(note)
    assert "clean context" in text
