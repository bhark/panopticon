"""The interface's non-obvious arithmetic. The screens themselves are reviewed via demo.py."""

from __future__ import annotations

from panopticon.model import Entry
from panopticon.tui.format import Coalescer, context_cell, context_pct, elapsed, tail_entries


def test_a_burst_of_events_costs_one_paint():
    paint = Coalescer(min_interval=0.35, max_stale=2.0)
    assert paint.due(100.0)  # first look

    for _ in range(12):
        paint.note()
    assert not paint.due(100.1)  # still inside min_interval
    assert paint.due(100.4)
    assert not paint.due(100.8)  # nothing new since


def test_a_quiet_board_still_repaints_for_the_clock():
    paint = Coalescer(min_interval=0.35, max_stale=2.0)
    paint.due(100.0)
    assert not paint.due(101.0)
    assert paint.due(102.5)


def test_context_percentage_says_unknown_rather_than_zero():
    assert context_pct(0, 200_000) is None  # provider has not reported yet
    assert context_pct(50_000, 0) is None  # window unknown
    assert context_pct(50_000, 200_000) == 25.0
    assert context_pct(400_000, 200_000) == 100.0  # a provider overshoot is not 200%


def test_context_cell_warns_as_the_window_fills():
    assert context_cell(0, 0)[0].strip() == "-"
    quiet = context_cell(20_000, 200_000)
    loud = context_cell(190_000, 200_000)
    assert quiet[1] != loud[1]
    assert loud[0].startswith(" 95%")


def test_transcript_keeps_the_tail_and_counts_what_it_dropped():
    entries = [Entry(kind="action", text=str(i), turn=i) for i in range(500)]
    shown, hidden = tail_entries(entries, limit=250)
    assert len(shown) == 250
    assert hidden == 250
    assert shown[-1].turn == 499
    assert tail_entries(entries[:10], limit=250) == (entries[:10], 0)


def test_elapsed_is_coarse():
    assert elapsed(-5) == "0s"
    assert elapsed(45) == "45s"
    assert elapsed(90) == "1m"
    assert elapsed(3_600) == "1h 00m"
    assert elapsed(90_000) == "1d 01h"
