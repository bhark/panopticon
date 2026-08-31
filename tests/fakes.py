"""A harness small enough to reason about, standing in for the orchestrator."""

from __future__ import annotations

import itertools
from pathlib import Path

from panopticon.model import (
    Agent,
    Event,
    Finalization,
    QueueItem,
    Seat,
    Situation,
    Submission,
    Task,
    Verdict,
    VerdictCall,
)
from panopticon.prompts import situation_preprompt


class FakeBoard:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self._ids = itertools.count(1)

    def duplicate_of(self, title: str) -> Task | None:
        return next((t for t in self.tasks.values() if t.title == title), None)

    def create(self, creator: str, title: str, description: str, roles: list[str]) -> Task:
        task = Task(
            id=f"T{next(self._ids)}",
            title=title,
            description=description,
            seats=[Seat(role=r) for r in roles],
            created_by=creator,
        )
        self.tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def assign(self, agent: str, task_id: str, role: str) -> tuple[Task, bool]:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"There is no task {task_id}.")
        seat = next((s for s in task.seats if s.role == role and not s.holder), None)
        if seat is None:
            raise ValueError(f"No open {role} seat on {task_id}.")
        seat.holder = agent
        return task, task.ready

    def unassign(self, agent: str, task_id: str) -> Task:
        task = self.tasks.get(task_id)
        seat = task.seat_of(agent) if task else None
        if task is None or seat is None:
            raise ValueError(f"You hold no seat on {task_id}.")
        seat.holder, seat.finalization = None, None
        return task

    def finalize(self, agent: str, task_id: str, reason: str, conclusion: str) -> tuple[Task, bool]:
        task = self.tasks.get(task_id)
        seat = task.seat_of(agent) if task else None
        if task is None or seat is None:
            raise ValueError(f"You hold no seat on {task_id}.")
        seat.finalization = Finalization(reason, conclusion)
        return task, all(s.finalization for s in task.seats)

    def cancel_finalize(self, agent: str, task_id: str) -> Task:
        task = self.tasks.get(task_id)
        seat = task.seat_of(agent) if task else None
        if task is None or seat is None:
            raise ValueError(f"You hold no seat on {task_id}.")
        seat.finalization = None
        return task

    def start(self, task: Task, worktree: str) -> None:
        task.worktree, task.started_at = worktree, 1.0

    def archive_task(self, task: Task, outcome: str) -> None:
        task.archived_at, task.outcome = 2.0, outcome

    def render(self) -> str:
        return "BOARD " + ", ".join(self.tasks)


class FakeKnowledge:
    def __init__(self) -> None:
        self.truths: list[object] = []
        self.pending: dict[str, Submission] = {}
        self.outcome = "pending"
        self._ids = itertools.count(1)

    def render(self) -> str:
        return "KB"

    def render_pending(self) -> str:
        return "PENDING " + ", ".join(self.pending)

    def submit(self, agent: str, title: str, body: str) -> Submission:
        sub = Submission(id=f"S{next(self._ids)}", title=title, body=body, submitted_by=agent)
        self.pending[sub.id] = sub
        return sub

    def join(self, agent: str, submission_id: str) -> Submission:
        sub = self.pending.get(submission_id)
        if sub is None:
            raise ValueError(f"There is no submission {submission_id}.")
        if sub.submitted_by == agent:
            raise ValueError("You cannot judge your own submission.")
        if agent in sub.jurors:
            raise ValueError("You are already on that jury.")
        sub.jurors.append(agent)
        return sub

    def leave(self, agent: str, submission_id: str) -> Submission:
        sub = self.pending[submission_id]
        sub.jurors.remove(agent)
        return sub

    def verdict(
        self,
        agent: str,
        submission_id: str,
        call: VerdictCall,
        reasoning: str,
        restated_title: str = "",
        restated_body: str = "",
    ) -> tuple[Submission, str]:
        sub = self.pending[submission_id]
        sub.verdicts.append(Verdict(agent, call, reasoning))
        return sub, self.outcome


class FakeBus:
    def __init__(self) -> None:
        self.shouts: list[tuple[str, str]] = []

    def shout(self, sender: str, body: str) -> None:
        self.shouts.append((sender, body))

    def render_shoutboard(self) -> str:
        return "SHOUTBOARD " + " | ".join(f"{s}: {b}" for s, b in self.shouts)


class FakeWorktrees:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.root = repo / ".panopticon" / "worktrees"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, task_id: str) -> Path:
        return self.root / task_id

    async def create(self, task_id: str) -> Path:
        path = self.path_for(task_id)
        path.mkdir(parents=True, exist_ok=True)
        return path


class FakeHarness:
    """Records everything a tool does to it, and mirrors the two async transitions."""

    def __init__(self, repo: Path, names: tuple[str, ...] = ("Ada", "Bo", "Cy")) -> None:
        repo.mkdir(parents=True, exist_ok=True)
        self.goal = "Make the flaky integration suite pass."
        self.agents = {n: Agent(name=n, provider="fake") for n in names}
        self.board = FakeBoard()
        self.bus = FakeBus()
        self.kb = FakeKnowledge()
        self.worktrees = FakeWorktrees(repo)
        self.force_ending = False
        self.posted: list[tuple[str, QueueItem]] = []
        self.broadcasts: list[QueueItem] = []
        self.events: list[Event] = []
        self.entered: list[tuple[str, Situation, str]] = []
        self.launched: list[Task] = []
        self.closed: list[Task] = []
        self.retired: list[str] = []
        self.left: list[tuple[str, str, str]] = []
        self.tallies = 0

    def post(self, recipient: str, item: QueueItem) -> None:
        self.posted.append((recipient, item))

    def broadcast(self, item: QueueItem, exclude: tuple[str, ...] = ()) -> None:
        self.broadcasts.append(item)
        for name in self.agents:
            if name not in exclude:
                self.posted.append((name, item))

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def enter(self, agent: Agent, situation: Situation, note: str = "") -> None:
        agent.situation = situation
        agent.entries.clear()
        self.entered.append(
            (agent.name, situation, situation_preprompt(agent, self, note, situation))
        )

    async def launch_task(self, task: Task) -> None:
        worktree = await self.worktrees.create(task.id)
        self.board.start(task, str(worktree))
        for name in task.holders:
            holder = self.agents[name]
            holder.task_id = task.id
            self.enter(holder, Situation.ON_TASK)
        self.launched.append(task)

    def resettle_jury(self, old_id, outcome, submission, exclude=()):
        for agent in self.agents.values():
            if agent.name in exclude or agent.submission_id != old_id:
                continue
            if outcome == "restated" and agent.name in submission.jurors:
                agent.submission_id = submission.id
                continue
            agent.submission_id = None
            agent.situation = Situation.IDLE

    def leave_task(self, name: str, task: Task, why: str, notify: bool = True) -> None:
        seat = task.seat_of(name)
        assert seat is not None
        seat.holder, seat.finalization = None, None
        leaver = self.agents[name]
        leaver.task_id = None
        self.enter(leaver, Situation.IDLE, f"You left task {task.id}.")
        if task.started_at is not None:
            task.started_at = None
            for other in task.holders:
                self.agents[other].situation = Situation.WAITING_FOR_SEATS
        if notify:
            for other in task.holders:
                self.post(other, QueueItem("task", f"{name} {why}. Task {task.id} has stopped."))
        self.left.append((name, task.id, why))

    async def close_task(self, task: Task) -> None:
        for name in task.holders:
            holder = self.agents[name]
            holder.task_id = None
            self.enter(holder, Situation.IDLE, f"You finished task {task.id}.")
        self.closed.append(task)

    def retire(self, agent: Agent) -> None:
        self.retired.append(agent.name)

    def tally_goal(self) -> tuple[int, int]:
        self.tallies += 1
        counted = [a for a in self.agents.values() if a.counts_toward_goal]
        return sum(1 for a in counted if a.voted_goal_reached), len(counted)

    # helpers the tests lean on

    def messages_for(self, name: str) -> list[str]:
        return [item.text for who, item in self.posted if who == name]

    def preprompt_for(self, name: str) -> str:
        return next(p for who, _, p in reversed(self.entered) if who == name)


async def seat_everyone(harness: FakeHarness, roles: dict[str, str], title: str = "fix it") -> Task:
    """Create a task, seat each agent in `roles` (name -> role) and launch it."""
    task = harness.board.create(next(iter(roles)), title, "make it green", list(roles.values()))
    for name, role in roles.items():
        harness.board.assign(name, task.id, role)
    await harness.launch_task(task)
    return task
