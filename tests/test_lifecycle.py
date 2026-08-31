"""One full run of the harness, end to end, through the mock provider.

Nothing here is scripted turn by turn. Each agent decides from the tools it is actually
offered, so the test exercises the real concurrency: agents interleave, wake each other
through their inboxes, and the run only ends if every hand-off in the design works.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from panopticon.cli import rebuild
from panopticon.config import Config
from panopticon.model import Action, Agent, QueueItem, Situation
from panopticon.orchestrator import Orchestrator
from panopticon.providers.base import TurnRequest
from panopticon.providers.mock import MockProvider
from panopticon.services.taskboard import TaskBoard
from panopticon.services.worktrees import Worktrees
from panopticon.store import (
    STATE_DIRNAME,
    Store,
    restore_agent,
    snapshot,
)
from tests.conftest import make_git_repo

GOAL = "prove the harness closes a task and banks a truth"


def build(tmp_path: Path, policy) -> Orchestrator:
    repo = make_git_repo(tmp_path / "repo")
    store = Store(repo / STATE_DIRNAME)
    return Orchestrator(
        goal=GOAL,
        agents=[Agent(name=n, provider="mock") for n in ("Ada", "Bo", "Cy")],
        providers={"mock": MockProvider(policy)},
        store=store,
        worktrees=Worktrees(repo, store.worktrees),
        config=Config(),
    )


async def until(cond, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.001)


class Crowd:
    """Decides one action per turn from the tools on offer, never from a fixed script."""

    def __init__(self) -> None:
        self.orch: Orchestrator | None = None
        self.ran: set[str] = set()
        self.saw_kb: set[str] = set()
        self.submitted: set[str] = set()

    def __call__(self, req: TurnRequest) -> Action | None:
        assert self.orch is not None
        names = {t.name for t in req.tools}
        me = req.agent

        if "mark_integration_done" in names:
            return self._closer(me, names)
        if "submit_verdict" in names:
            return Action("submit_verdict", {"verdict": "true", "reasoning": "checked the tree"})
        if "finalize_task" in names:
            return self._worker(me)
        if "cancel_finalize" in names:
            return Action("wait", {})
        if "join_jury" in names:
            return self._juror(me)
        if self._work_is_done() and "vote_goal_reached" in names:
            return Action("vote_goal_reached", {"note": "task closed, truth banked"})
        if "create_task" in names:
            return self._idle(me, names)
        return Action("wait", {})

    # branches

    def _closer(self, me: str, names: set[str]) -> Action:
        if me not in self.saw_kb:
            self.saw_kb.add(me)
            return Action("view_knowledge_base", {})
        if me not in self.submitted and "submit_truth" in names:
            self.submitted.add(me)
            return Action(
                "submit_truth",
                {"title": "note.txt exists in the worktree", "body": "written by the task"},
            )
        return Action("mark_integration_done", {"summary": "merged and told everyone"})

    def _worker(self, me: str) -> Action:
        task = self.orch.board.task_of(me)
        if me not in self.ran:
            self.ran.add(me)
            return Action("bash", {"command": "echo hi > note.txt"})
        return Action(
            "finalize_task",
            {"task_id": task.id, "reason": "note written", "conclusion": "note.txt is there"},
        )

    def _juror(self, me: str) -> Action:
        waiting = [s for s in self.orch.kb.pending.values() if s.submitted_by != me]
        if not waiting:
            return Action("wait", {})
        return Action("join_jury", {"submission_id": waiting[0].id})

    def _idle(self, me: str, names: set[str]) -> Action:
        open_tasks = [t for t in self.orch.board.open_tasks() if t.started_at is None]
        if not open_tasks:
            if me != "Ada":
                return Action("wait", {})
            return Action(
                "create_task",
                {
                    "title": "write the note",
                    "description": "put a note in the worktree",
                    "roles": ["writer", "checker"],
                },
            )
        task = open_tasks[0]
        seat = next((s for s in task.seats if s.holder is None), None)
        if seat is None or "assign_self" not in names:
            return Action("wait", {})
        return Action("assign_self", {"task_id": task.id, "role": seat.role})

    def _work_is_done(self) -> bool:
        return bool(self.orch.board.archive()) and bool(self.orch.kb.truths)


@pytest.mark.asyncio
async def test_the_harness_runs_a_task_and_a_jury_to_completion(tmp_path):
    crowd = Crowd()
    orch = build(tmp_path, crowd)
    crowd.orch = orch

    await asyncio.wait_for(orch.run(), timeout=30)

    assert orch.stopped_because.startswith("goal reached"), orch.stopped_because
    archived = orch.board.archive()
    assert len(archived) == 1
    assert (Path(archived[0].worktree) / "note.txt").exists()
    assert [t.title for t in orch.kb.truths] == ["note.txt exists in the worktree"]
    assert not orch.kb.pending
    assert all(
        a.situation is Situation.RELEASED for a in orch.agents.values() if a.counts_toward_goal
    )
    # the closer is transient: it exists, it did the integration, and it is gone
    closers = [a for a in orch.agents.values() if a.transient]
    assert len(closers) == 1 and closers[0].situation is Situation.RELIEVED


@pytest.mark.asyncio
async def test_a_paused_run_saves_a_state_file_that_restores(tmp_path):
    crowd = Crowd()
    orch = build(tmp_path, crowd)
    crowd.orch = orch
    runner = asyncio.create_task(orch.run())
    await until(lambda: bool(orch.board.tasks))
    await orch.pause()
    await asyncio.wait_for(runner, timeout=5)

    state = orch.store.load()
    assert state["goal"] == GOAL
    assert state["board"]

    restored = {a["name"]: restore_agent(dict(a)) for a in state["agents"]}
    for name, agent in restored.items():
        live = orch.agents[name]
        assert agent.situation is live.situation
        assert agent.turns == live.turns
        assert [e.text for e in agent.entries] == [e.text for e in live.entries]

    board = TaskBoard()
    board.restore(state["board"])
    assert [t.id for t in board.tasks.values()] == [t.id for t in orch.board.tasks.values()]
    for restored_task, live_task in zip(
        board.tasks.values(), orch.board.tasks.values(), strict=True
    ):
        assert [s.holder for s in restored_task.seats] == [s.holder for s in live_task.seats]
        assert restored_task.started_at == live_task.started_at


def test_an_undelivered_inbox_survives_the_snapshot(tmp_path):
    """A pause must not lose the DMs an agent had not read yet."""
    crowd = Crowd()
    orch = build(tmp_path, crowd)
    crowd.orch = orch
    orch.post("Cy", QueueItem("dm", "psst"))
    orch.post("Cy", QueueItem("shout", "Ada wrote to the shoutboard"))

    raw = snapshot(orch)
    cy = next(a for a in raw["agents"] if a["name"] == "Cy")
    assert [item["text"] for item in cy["pending"]] == ["psst", "Ada wrote to the shoutboard"]
    assert [i.text for i in restore_agent(dict(cy)).inbox] == [
        "psst",
        "Ada wrote to the shoutboard",
    ]


@pytest.mark.asyncio
async def test_an_agent_survives_its_own_worktree_being_removed(tmp_path):
    """A closer that merges and deletes its worktree used to kill the agent: the provider
    was still spawned with a cwd that no longer existed, four times, and then it died."""
    orch = build(tmp_path, lambda req: Action("wait", {}))
    agent = orch.agents["Ada"]
    task = orch.board.create("Ada", "t", "d", ["solo"])
    orch.board.assign("Ada", task.id, "solo")
    await orch.launch_task(task)

    worktree = Path(task.worktree)
    assert orch._cwd_for(agent) == task.worktree

    shutil.rmtree(worktree)
    assert orch._cwd_for(agent) == str(orch.worktrees.repo)
    assert Path(orch._cwd_for(agent)).is_dir()


@pytest.mark.asyncio
async def test_resuming_a_settled_session_closes_without_spending_a_turn(tmp_path):
    """Resuming a finished run used to wake every released agent and call its provider."""
    provider = MockProvider(lambda req: Action("wait", {}))
    repo = make_git_repo(tmp_path / "repo")
    store = Store(repo / STATE_DIRNAME)
    agents = [
        Agent(name=n, provider="mock", situation=Situation.RELEASED, voted_goal_reached=True)
        for n in ("Ada", "Bo", "Cy")
    ]
    orch = Orchestrator(
        goal=GOAL,
        agents=agents,
        providers={"mock": provider},
        store=store,
        worktrees=Worktrees(repo, store.worktrees),
        config=Config(),
    )
    await asyncio.wait_for(orch.run(), timeout=5)

    assert orch.stopped_because.startswith("goal reached")
    assert provider.calls == []


@pytest.mark.asyncio
async def test_a_run_paused_mid_task_resumes_and_finishes_the_job(tmp_path):
    """The headline feature: pausing partway and picking the same run back up."""
    crowd = Crowd()
    first = build(tmp_path, crowd)
    crowd.orch = first
    runner = asyncio.create_task(first.run())
    await until(lambda: any(t.running for t in first.board.tasks.values()))
    await first.pause()
    await asyncio.wait_for(runner, timeout=5)

    assert not first.board.archive(), "the pause has to land while there is still work left"
    assert not first.kb.truths
    turns_before = {n: a.turns for n, a in first.agents.items()}

    second = rebuild(
        first.store,
        Config(),
        {"mock": MockProvider(crowd)},
        first.worktrees.repo,
    )
    crowd.orch = second
    assert [t.id for t in second.board.tasks.values()] == [t.id for t in first.board.tasks.values()]
    assert all(second.agents[n].turns == t for n, t in turns_before.items())

    await asyncio.wait_for(second.run(), timeout=30)

    assert second.stopped_because.startswith("goal reached"), second.stopped_because
    assert second.kb.truths
    assert second.board.archive()
    # it carried on rather than starting over
    assert any(second.agents[n].turns > t for n, t in turns_before.items())
