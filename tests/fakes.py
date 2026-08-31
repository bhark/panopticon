"""A harness small enough to reason about, standing in for the orchestrator."""

from __future__ import annotations

from pathlib import Path

from panopticon.model import (
    Agent,
    Event,
    QueueItem,
    Situation,
    Task,
)
from panopticon.prompts import situation_preprompt
from panopticon.services.bus import Bus
from panopticon.services.knowledge import Knowledge
from panopticon.services.taskboard import TaskBoard


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

    def __init__(
        self,
        repo: Path,
        names: tuple[str, ...] = ("Ada", "Bo", "Cy"),
        board: TaskBoard | None = None,
    ) -> None:
        repo.mkdir(parents=True, exist_ok=True)
        self.goal = "Make the flaky integration suite pass."
        self.agents = {n: Agent(name=n, provider="fake") for n in names}
        self.board = board or TaskBoard()
        self.kb = Knowledge()
        self.bus = Bus(lambda senders: None)
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
        was_running = task.running
        self.board.unassign(name, task.id)
        leaver = self.agents[name]
        leaver.task_id = None
        self.enter(leaver, Situation.IDLE, f"You left task {task.id}.")
        if was_running:
            for other in task.holders:
                self.agents[other].situation = Situation.WAITING_FOR_SEATS
        if notify:
            self.post(name, QueueItem("task", f"You have been unassigned from {task.id}: {why}."))
            for other in task.holders:
                self.post(other, QueueItem("task", f"{name} {why}. Task {task.id} has stopped."))
        self.emit(Event(kind="seat", text=f"{name} left {task.id}: {why}", agent=name))
        self.left.append((name, task.id, why))

    async def close_task(self, task: Task) -> None:
        for name in task.holders:
            holder = self.agents[name]
            holder.task_id = None
            self.enter(holder, Situation.IDLE, f"You finished task {task.id}.")
        self.closed.append(task)

    def retire(self, agent: Agent) -> None:
        agent.situation = Situation.RELIEVED
        agent.wake_at = None
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
