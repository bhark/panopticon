"""Tasks, seats and finalization. Pure state; the orchestrator does the side effects."""

from __future__ import annotations

import re
import time

from panopticon.model import Finalization, Seat, Task


def _slug(title: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", title.lower()))[:24].strip("-") or "task"


def _ago(then: float, now: float) -> str:
    seconds = int(now - then)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


def _seat_line(seat: Seat) -> str:
    if seat.holder is None:
        return f"{seat.role}=OPEN"
    return f"{seat.role}={seat.holder}" + ("(finalized)" if seat.finalization else "")


class TaskBoard:
    ARCHIVE_SHOWN = 20

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    # queries

    def open_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.archived_at is None]

    def archive(self) -> list[Task]:
        done = [t for t in self.tasks.values() if t.archived_at is not None]
        return sorted(done, key=lambda t: t.archived_at or 0, reverse=True)

    def get(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def task_of(self, agent: str) -> Task | None:
        return next((t for t in self.open_tasks() if t.seat_of(agent)), None)

    def render(self) -> str:
        """The view returned by the view_task_board tool."""
        now = time.time()
        lines: list[str] = []
        open_tasks = self.open_tasks()
        lines.append(f"open tasks ({len(open_tasks)})" if open_tasks else "open tasks: none")
        for task in open_tasks:
            taken = sum(1 for s in task.seats if s.holder)
            state = (
                f"running {_ago(task.started_at or now, now)}"
                if task.running
                else f"waiting {taken}/{len(task.seats)} seats"
            )
            lines.append(
                f"[{task.id}] {task.title} | {state} | by {task.created_by} "
                f"{_ago(task.created_at, now)} ago"
            )
            lines.append(f"  {task.description}")
            lines.append("  " + "  ".join(_seat_line(s) for s in task.seats))

        archived = self.archive()
        shown = archived[: self.ARCHIVE_SHOWN]
        lines.append(f"archive ({len(shown)} of {len(archived)}, newest first)")
        for task in shown:
            outcome = f" | {task.outcome}" if task.outcome else ""
            lines.append(
                f"[{task.id}] {task.title} | closed {_ago(task.archived_at or now, now)} ago"
                f"{outcome}"
            )
        return "\n".join(lines)

    # mutations

    def create(self, creator: str, title: str, description: str, roles: list[str]) -> Task:
        if not roles:
            raise ValueError("a task needs at least one seat, so at least one role")
        task = Task(
            id=self._new_id(title),
            title=title,
            description=description,
            seats=[Seat(role=r) for r in roles],
            created_by=creator,
        )
        self.tasks[task.id] = task
        return task

    def assign(self, agent: str, task_id: str, role: str) -> tuple[Task, bool]:
        """Returns (task, task_just_became_ready). Raises ValueError on a bad seat."""
        task = self._live(task_id)
        if held := task.seat_of(agent):
            raise ValueError(f"you already hold the {held.role} seat on {task.id}")
        wanted = role.strip().casefold()
        seat = next(
            (s for s in task.seats if s.holder is None and s.role.casefold() == wanted), None
        )
        if seat is None:
            open_roles = ", ".join(s.role for s in task.seats if s.holder is None) or "none"
            raise ValueError(f"no open '{role}' seat on {task.id}. open seats: {open_roles}")
        seat.holder = agent
        seat.assigned_at = time.time()
        return task, task.ready and task.started_at is None

    def unassign(self, agent: str, task_id: str) -> Task:
        task = self._live(task_id)
        seat = task.seat_of(agent)
        if seat is None:
            raise ValueError(f"you hold no seat on {task_id}")
        seat.holder = None
        seat.assigned_at = None
        seat.nudges_sent = 0
        seat.finalization = None
        # a task that loses a seat is back to waiting, worktree and all
        task.started_at = None
        return task

    def finalize(self, agent: str, task_id: str, reason: str, conclusion: str) -> tuple[Task, bool]:
        """Returns (task, all_seats_agreed)."""
        task = self._live(task_id)
        seat = task.seat_of(agent)
        if seat is None:
            raise ValueError(f"you hold no seat on {task_id}")
        if not task.running:
            raise ValueError(f"{task_id} has not started yet")
        seat.finalization = Finalization(reason=reason, conclusion=conclusion)
        return task, all(s.finalization for s in task.seats)

    def cancel_finalize(self, agent: str, task_id: str) -> Task:
        task = self._live(task_id)
        seat = task.seat_of(agent)
        if seat is None:
            raise ValueError(f"you hold no seat on {task_id}")
        if seat.finalization is None:
            raise ValueError(f"you have not finalized {task_id}")
        seat.finalization = None
        return task

    def start(self, task: Task, worktree: str) -> None:
        task.started_at = time.time()
        task.worktree = worktree

    def archive_task(self, task: Task, outcome: str) -> None:
        task.archived_at = time.time()
        task.outcome = outcome

    # internals

    def _live(self, task_id: str) -> Task:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"no task {task_id}")
        if task.archived_at is not None:
            raise ValueError(f"{task_id} is archived")
        return task

    def _new_id(self, title: str) -> str:
        slug = _slug(title)
        n = len(self.tasks) + 1
        while f"t{n}-{slug}" in self.tasks:
            n += 1
        return f"t{n}-{slug}"
