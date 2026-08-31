"""Periodic upkeep: seat nudges, seat expiry, dead-agent reaping, state flush."""

from __future__ import annotations

import asyncio
import time

from panopticon.model import Event, Harness, QueueItem, Seat, Task


class Janitor:
    TICK_SECONDS = 30
    NUDGE_MINUTES = (15, 30, 45)
    EXPIRE_MINUTES = 60

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
        for task in self.harness.board.open_tasks():
            for seat in task.seats:
                if seat.holder is None:
                    continue
                agent = self.harness.agents.get(seat.holder)
                if agent is None or not agent.alive:
                    self._release(task, seat, "the agent holding it is gone", notify=False)
                    continue
                if task.started_at is not None:
                    continue
                minutes = int((now - (seat.assigned_at or task.created_at)) // 60)
                if minutes >= self.EXPIRE_MINUTES:
                    self._release(task, seat, f"{minutes} minutes without the task starting")
                    continue
                due = sum(1 for m in self.NUDGE_MINUTES if minutes >= m)
                if due > seat.nudges_sent:
                    seat.nudges_sent = due
                    self.harness.post(
                        seat.holder,
                        QueueItem(
                            kind="task",
                            text=(
                                f"You've been assigned to {task.id} ({task.title}) for "
                                f"{minutes} minutes and it still has open seats. Do you want to "
                                "keep waiting? We'll notify you again in 15 minutes."
                            ),
                        ),
                    )

    def _release(self, task: Task, seat: Seat, why: str, notify: bool = True) -> None:
        # the harness settles the agent and everyone left on the task; the board alone cannot
        self.harness.leave_task(seat.holder or "", task, why, notify=notify)
