"""The extending shoutboard debounce and what the board shows."""

from __future__ import annotations

import asyncio
import time

from panopticon.services.bus import Bus

DEBOUNCE = 0.2
MAX_BATCH = 0.5


def fast_bus() -> tuple[Bus, list[list[str]]]:
    fired: list[list[str]] = []
    bus = Bus(fired.append)
    bus.DEBOUNCE_SECONDS = DEBOUNCE
    bus.MAX_BATCH_SECONDS = MAX_BATCH
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


async def test_unbroken_shouting_still_flushes_at_the_batch_cap():
    bus, fired = fast_bus()
    runner = asyncio.create_task(bus.run())

    shouting = True

    async def keep_shouting() -> None:
        while shouting:
            bus.shout("ada", "again")
            await asyncio.sleep(DEBOUNCE * 0.5)  # never quiet long enough to end the window

    talker = asyncio.create_task(keep_shouting())
    await asyncio.sleep(MAX_BATCH * 1.5)
    shouting = False
    await asyncio.gather(talker, return_exceptions=True)

    assert fired == [["ada"]]

    runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)


async def test_a_quiet_bus_never_flushes():
    bus, fired = fast_bus()
    runner = asyncio.create_task(bus.run())

    await asyncio.sleep(DEBOUNCE * 2)

    assert fired == []
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
