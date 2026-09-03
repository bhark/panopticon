"""The clock behind the task board: it delivers what the board says is overdue."""

from __future__ import annotations

import asyncio
import time

from panopticon.model import Event, Harness, QueueItem, Situation
from panopticon.services.taskboard import (
    CHECKIN_MINUTES,
    CHECKIN_STRIKES,
    EXPIRE_MINUTES,
    REPLY_MINUTES,
)

# an agent in one of these can still end up filling somebody's open seat
_HELPERS = (Situation.IDLE, Situation.ON_TASK, Situation.JURY)
CLOSER_ATTEMPTS = 2  # the closer, then one replacement, then the task is archived unintegrated


class Janitor:
    TICK_SECONDS = 30

    def __init__(self, harness: Harness) -> None:
        self.harness = harness

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.TICK_SECONDS)
            try:
                self.tick()
            except Exception as exc:  # upkeep must never take the harness down
                self.harness.emit(Event(kind="error", text=f"janitor: {type(exc).__name__}: {exc}"))

    def tick(self, now: float | None = None) -> None:
        now = now or time.time()
        self._orphaned_seats()
        self._stranded_closers()
        self._waiting_seats(now)
        self._deadlock(now)
        self._running_seats(now)

    # seats

    def _orphaned_seats(self) -> None:
        for task, seat in self.harness.board.held_seats():
            holder = self.harness.agents.get(seat.holder or "")
            if holder is None or not holder.alive:
                # the harness settles the agent and everyone left on the task; the board cannot
                self.harness.leave_task(
                    seat.holder or "", task, "the agent holding it is gone", notify=False
                )

    def _waiting_seats(self, now: float) -> None:
        board = self.harness.board
        for task, seat, minutes in board.waiting_seats(now):
            if minutes >= EXPIRE_MINUTES:
                self.harness.leave_task(
                    seat.holder or "", task, f"{minutes} minutes without the task starting"
                )
            elif board.nudge_due(seat, minutes):
                self.harness.post(
                    seat.holder or "",
                    QueueItem(
                        kind="task",
                        text=(
                            f"You've been assigned to {task.id} ({task.title}) for "
                            f"{minutes} minutes and it still has open seats. Do you want to "
                            "keep waiting, or take a seat on a task that can start? We'll "
                            "notify you again in 15 minutes."
                        ),
                    ),
                )

    def _deadlock(self, now: float) -> None:
        """Every agent waiting on a seat nobody is left to fill. One release breaks it."""
        live = [a for a in self.harness.agents.values() if a.alive and not a.transient]
        if not any(a.situation is Situation.WAITING_FOR_SEATS for a in live):
            return
        if any(a.situation in _HELPERS for a in live):
            return
        seats = self.harness.board.waiting_seats(now)
        if not seats:
            return
        task, seat, minutes = max(seats, key=lambda item: item[2])
        self.harness.leave_task(
            seat.holder or "",
            task,
            f"every agent was waiting on a seat, so the longest wait ({minutes}m) was released",
        )

    def _running_seats(self, now: float) -> None:
        board = self.harness.board
        for task, seat, minutes in board.running_seats(now):
            holder = self.harness.agents.get(seat.holder or "")
            if holder is None:
                continue
            # answering means taking a turn inside the window, not merely turning again later
            asked = seat.checkin_at or 0
            if seat.checkins and asked < holder.last_turn_at <= asked + REPLY_MINUTES * 60:
                board.answered(seat)
                continue
            if minutes < CHECKIN_MINUTES:
                continue
            if seat.checkins >= CHECKIN_STRIKES - 1:
                self.harness.leave_task(
                    seat.holder or "",
                    task,
                    f"no answer to {seat.checkins} check-ins on a running task",
                )
                continue
            last = board.checkin(seat, now) >= CHECKIN_STRIKES - 1
            text = (
                f"Check-in on {task.id} ({task.title}), which has been running for a while. "
                f"Take a turn within {REPLY_MINUTES} minutes to show you are still on it, and "
                "finalize if your part is done."
            )
            if last:
                text += (
                    " This is your last one: miss it and your seat is released, which stops "
                    "the task for everyone on it."
                )
            self.harness.post(seat.holder or "", QueueItem(kind="task", text=text))

    # closers

    def _stranded_closers(self) -> None:
        """A closer that died leaves the task open forever with every seat held and finalized."""
        harness = self.harness
        for task in harness.board.open_tasks():
            if not task.closing:
                continue
            closer = harness.agents.get(task.closer or "")
            if closer is not None and closer.alive:
                continue
            if task.closer_attempts < CLOSER_ATTEMPTS:
                harness.spawn_closer(task)
                harness.emit(Event("task", f"{task.id} lost its closer; a new one has it"))
                continue
            harness.board.archive_task(
                task, f"no closer survived it; the work is unmerged in {task.worktree}"
            )
            harness.broadcast(
                QueueItem(
                    "board",
                    f"Task {task.id} ({task.title}) lost two closers and was archived without "
                    f"being integrated. Its work is still in the git worktree at "
                    f"{task.worktree}. File a task for it if it matters.",
                )
            )
            harness.emit(Event("task", f"{task.id} archived: no closer survived it"))
