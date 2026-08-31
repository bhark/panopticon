"""Periodic upkeep: seat nudges, seat expiry, dead-agent reaping, state flush."""

from __future__ import annotations

from panopticon.model import Harness


class Janitor:
    TICK_SECONDS = 30
    NUDGE_MINUTES = (15, 30, 45)
    EXPIRE_MINUTES = 60

    def __init__(self, harness: Harness) -> None:
        self.harness = harness

    async def run(self) -> None: ...
