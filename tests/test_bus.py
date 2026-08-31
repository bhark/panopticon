"""The extending shoutboard debounce and what the board shows."""

from __future__ import annotations

import asyncio
import time

from panopticon.services.bus import Bus

DEBOUNCE = 0.2


def fast_bus() -> tuple[Bus, list[list[str]]]:
    fired: list[list[str]] = []
    bus = Bus(fired.append)
    bus.DEBOUNCE_SECONDS = DEBOUNCE
    return bus, fired


async def test_a_second_shout_extends_the_window_instead_of_riding_the_first_timer():
    bus, fired = fast_bus()
    runner = asyncio.create_task(bus.run())

    bus.shout("ada", "one")
    await asyncio.sleep(DEBOUNCE * 0.6)
    bus.shout("bo", "two")
    bus.shout("ada", "three")
    await asyncio.sleep(DEBOUNCE * 0.6)  # past the first shout's window, inside the extended one

    assert fired == []

    await asyncio.sleep(DEBOUNCE * 1.5)
    assert fired == [["ada", "bo"]]  # one flush, each sender once, in order

    bus.shout("cy", "four")
    await asyncio.sleep(DEBOUNCE * 1.5)
    assert fired == [["ada", "bo"], ["cy"]]  # the next burst is its own batch

    runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)
    assert runner.cancelled()


async def test_a_quiet_bus_never_flushes():
    bus, fired = fast_bus()
    runner = asyncio.create_task(bus.run())

    bus.send_direct("ada", "bo", "just for you")
    await asyncio.sleep(DEBOUNCE * 2)

    assert fired == []
    assert bus.direct[0].recipient == "bo"
    runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)


def test_shouts_drop_at_the_hour_boundary():
    bus, _ = fast_bus()
    now = time.time()
    bus.shout("ada", "ancient")
    bus.shout("bo", "still fresh")
    bus.shouts[0].at = now - Bus.MAX_AGE_SECONDS - 1
    bus.shouts[1].at = now - Bus.MAX_AGE_SECONDS + 1

    view = bus.render_shoutboard()

    assert "still fresh" in view
    assert "ancient" not in view


def test_the_board_is_trimmed_to_the_token_budget_oldest_first():
    bus, _ = fast_bus()
    budget = Bus.MAX_TOKENS * 4
    for n in range(60):
        bus.shout(f"a{n}", f"{n:03d} " + "x" * 1000)

    view = bus.render_shoutboard()

    assert len(view) <= budget + 64  # the header sits outside the per-entry budget
    assert "059 " in view
    assert "000 " not in view
