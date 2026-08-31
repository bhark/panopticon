"""The clock behind the task board: it delivers what the board says is overdue."""

from __future__ import annotations

import asyncio
import time

from panopticon.model import Event, Harness, QueueItem
from panopticon.services.taskboard import EXPIRE_MINUTES


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
        board = self.harness.board
        for task, seat in board.held_seats():
            holder = self.harness.agents.get(seat.holder or "")
            if holder is None or not holder.alive:
                # the harness settles the agent and everyone left on the task; the board cannot
                self.harness.leave_task(seat.holder or "", task, "the agent holding it is gone",
                                        notify=False)

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
                            "keep waiting? We'll notify you again in 15 minutes."
                        ),
                    ),
                )
