"""Noticing a newer release: what counts as behind, and what a failed check still costs."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from panopticon import cli
from panopticon import update as update_mod


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    """Never the real ~/.panopticon, and always a terminal so refresh isn't skipped."""
    path = tmp_path / "update.json"
    monkeypatch.setattr(update_mod, "CACHE_FILE", path)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    return path


class TestBehind:
    @pytest.mark.parametrize(
        ("here", "tag", "expected"),
        [
            ("0.2.0", "v0.3.0", True),
            ("0.2.0", "v0.2.0", False),
            ("0.3.0", "v0.2.0", False),
            ("0.2.9", "v0.10.0", True),  # numbers, not strings
            ("0.1.dev0+d20260831", "v0.3.0", False),  # a source checkout is never nudged
            ("0+unknown", "v0.3.0", False),
            ("0.2.0", "nightly", False),
            ("0.2.0", "", False),
        ],
    )
    def test_only_plain_numbers_on_both_sides_count(self, monkeypatch, here, tag, expected):
        monkeypatch.setattr(update_mod, "__version__", here)
        assert update_mod.behind(tag) is expected


class TestCache:
    def test_an_absent_cache_is_a_miss_not_an_error(self):
        assert update_mod._recall() == ("", 0.0)

    def test_an_unreadable_cache_is_a_miss_not_an_error(self, cache):
        cache.write_text("{ not json")
        assert update_mod._recall() == ("", 0.0)

    def test_a_cache_missing_its_keys_is_a_miss(self, cache):
        cache.write_text(json.dumps({"tag": "v9.9.9"}))
        assert update_mod._recall() == ("", 0.0)

    def test_what_was_remembered_comes_back(self, cache):
        update_mod.remember("v0.3.0")
        tag, checked_at = update_mod._recall()
        assert tag == "v0.3.0"
        assert time.time() - checked_at < 5


class TestNote:
    def test_nothing_to_say_when_the_cache_matches(self, monkeypatch):
        monkeypatch.setattr(update_mod, "__version__", "0.3.0")
        update_mod.remember("v0.3.0")
        assert update_mod.note() == ""

    def test_the_line_names_the_version_and_the_command(self, monkeypatch):
        monkeypatch.setattr(update_mod, "__version__", "0.2.0")
        update_mod.remember("v0.3.0")
        assert update_mod.note() == "update 0.3.0 available · panopticon update"


class TestRefresh:
    def test_a_fresh_cache_is_not_refetched(self, monkeypatch):
        update_mod.remember("v0.3.0")
        monkeypatch.setattr(update_mod, "latest", _never)
        asyncio.run(update_mod.refresh())

    def test_nothing_happens_when_stdout_is_not_a_terminal(self, cache, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
        monkeypatch.setattr(update_mod, "latest", _never)
        asyncio.run(update_mod.refresh())
        assert not cache.exists()

    def test_a_stale_cache_is_refetched(self, cache, monkeypatch):
        cache.write_text(json.dumps({"tag": "v0.2.0", "checked_at": 0.0}))
        monkeypatch.setattr(update_mod, "latest", lambda: "v0.3.0")
        asyncio.run(update_mod.refresh())
        assert update_mod._recall()[0] == "v0.3.0"

    def test_a_failed_check_keeps_the_answer_but_counts_as_today(self, cache, monkeypatch):
        cache.write_text(json.dumps({"tag": "v0.2.0", "checked_at": 0.0}))
        monkeypatch.setattr(update_mod, "latest", _offline)
        asyncio.run(update_mod.refresh())
        tag, checked_at = update_mod._recall()
        assert tag == "v0.2.0"
        # without the stamp, an offline machine would refetch after every single command
        assert time.time() - checked_at < 5


def _never() -> str:
    raise AssertionError("the network should not have been touched")


def _offline() -> str:
    raise update_mod.UpdateError("no route to host")


class TestTheCheckNeverHoldsTheRunUp:
    def test_a_slow_check_delays_neither_the_harness_nor_the_exit(self, monkeypatch):
        """A headless run ends when the harness does, however long the feed takes."""

        async def crawl() -> None:
            await asyncio.sleep(30)

        monkeypatch.setattr(update_mod, "refresh", crawl)

        started = asyncio.Event()

        class Harness:
            stopped_because = ""

            async def run(self) -> None:
                started.set()

        async def go() -> None:
            await asyncio.wait_for(cli._headless(Harness()), timeout=2)

        asyncio.run(go())
        assert started.is_set()


class TestTheGoalBar:
    def test_the_note_rides_on_the_header_line(self):
        from panopticon.tui.render import goal_bar

        bar = goal_bar("ship it", time.time(), [], "", "update 0.3.0 available · panopticon update")
        assert "update 0.3.0 available · panopticon update" in bar.plain
        assert "ship it" in bar.plain

    def test_nothing_is_added_when_there_is_no_update(self):
        from panopticon.tui.render import goal_bar

        assert goal_bar("ship it", time.time(), [], "", "").plain.rstrip().endswith("0/0 live")
