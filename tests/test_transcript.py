from __future__ import annotations

import pytest

from panopticon import transcript as tx
from panopticon.model import Action, Agent, Entry, QueueItem, Usage
from panopticon.providers.mock import MockProvider


def agent(name: str = "Suzanne") -> Agent:
    a = Agent(name=name, provider="mock")
    tx.reset(a, "You just joined. Nothing is waiting for you.")
    return a


def turn(a: Agent, tool: str, args: dict, result: str, inbox: list[str] = ()) -> None:
    a.turns += 1
    if inbox:
        tx.append_inbox(a, [QueueItem(kind="dm", text=t) for t in inbox])
    tx.append_action(a, Action(tool=tool, args=args))
    tx.append_result(a, result)


def stable_part(prompt: str) -> str:
    return prompt.removesuffix(tx.TAIL).rstrip("\n")


class TestAppendOnly:
    def test_prefix_holds_across_a_long_run_with_inbox_and_results(self):
        a = agent()
        previous = render_and_check = tx.render(a)
        for n in range(1, 120):
            inbox = [f"Alex: ping {n}"] if n % 3 == 0 else []
            if n % 7 == 0:
                inbox.append("shoutboard: two new messages")
            turn(a, "bash", {"command": f"pytest -q -k t{n}"}, f"exit 0, {n} passed", inbox)
            current = tx.render(a)
            assert current.startswith(stable_part(previous)), f"prefix broke at turn {n}"
            previous = current
        assert render_and_check in previous or stable_part(render_and_check) in previous

    def test_inbox_is_folded_where_it_was_consumed_not_re_emitted_at_the_tail(self):
        a = agent()
        turn(a, "wait", {}, "woken", inbox=["Alex: start on the parser"])
        early = tx.render(a)
        for n in range(5):
            turn(a, "bash", {"command": f"echo {n}"}, "exit 0")
        later = tx.render(a)
        assert later.count("Alex: start on the parser") == 1
        assert later.startswith(stable_part(early))

    def test_a_new_inbox_item_never_reorders_what_came_before(self):
        a = agent()
        turn(a, "bash", {"command": "ls"}, "exit 0")
        before = stable_part(tx.render(a))
        turn(a, "wait", {}, "woken", inbox=["Alex: hi", "Mira: hi"])
        after = tx.render(a)
        assert after.startswith(before)
        assert after.index("Alex: hi") > after.index("[turn 1]")

    def test_prefix_survives_a_compaction_from_the_new_epoch_on(self):
        a = agent()
        for n in range(1, 60):
            turn(a, "bash", {"command": f"echo {n}"}, "x" * 400)
        before = tx.render(a)

        provider = MockProvider(context_window=40_000)
        assert tx.needs_compaction(a, provider)
        assert pytest.importorskip("asyncio").run(tx.compact(a, provider, "you are Suzanne"))

        after = tx.render(a)
        assert not after.startswith(stable_part(before))  # the one deliberate break
        assert "[summary]" in after
        for n in range(60, 80):
            turn(a, "bash", {"command": f"echo {n}"}, "ok")
            current = tx.render(a)
            assert current.startswith(stable_part(after))
            after = current


class TestCompaction:
    def _long(self, a: Agent, turns: int = 40) -> None:
        for n in range(1, turns + 1):
            turn(a, "bash", {"command": f"echo {n}"}, "y" * 500)

    async def test_cut_never_orphans_a_result_from_its_action(self):
        for keep in (200, 800, 2_000, 6_000):
            a = agent()
            self._long(a)
            original = tx.KEEP_TOKENS
            tx.KEEP_TOKENS = keep
            try:
                cut = tx._cut_point(a.entries)
            finally:
                tx.KEEP_TOKENS = original
            assert a.entries[cut].kind != "result", f"cut orphaned a result at keep={keep}"

    async def test_compaction_keeps_the_tail_and_drops_the_head(self):
        a = agent()
        self._long(a)
        head_line = a.entries[1].text
        tail_line = a.entries[-1].text
        provider = MockProvider(context_window=40_000, summary="did 40 turns of echo")

        assert await tx.compact(a, provider, "system")

        rendered = tx.render(a)
        assert "did 40 turns of echo" in rendered
        assert head_line not in rendered
        assert tail_line in rendered
        assert provider.summarized and "1. Goal" in provider.summarized[0]

    async def test_breaker_stops_after_three_consecutive_failures(self):
        a = agent()
        self._long(a)
        provider = MockProvider(context_window=40_000, summary=None)

        for _ in range(3):
            assert await tx.compact(a, provider, "system") is False
        assert len(provider.summarized) == 3

        assert await tx.compact(a, provider, "system") is False
        assert len(provider.summarized) == 3, "breaker should stop calling the provider"

    async def test_a_success_clears_the_strike_count(self):
        a = agent()
        self._long(a)
        failing = MockProvider(context_window=40_000, summary=None)
        assert await tx.compact(a, failing, "system") is False
        assert await tx.compact(a, failing, "system") is False

        working = MockProvider(context_window=40_000, summary="ok")
        assert await tx.compact(a, working, "system") is True

        self._long(a)
        assert await tx.compact(a, failing, "system") is False
        assert len(failing.summarized) == 3, "strikes should have restarted from zero"

    async def test_reset_clears_the_strike_count_too(self):
        a = agent()
        self._long(a)
        provider = MockProvider(context_window=40_000, summary=None)
        for _ in range(3):
            await tx.compact(a, provider, "system")

        tx.reset(a, "fresh start")
        self._long(a)
        await tx.compact(a, provider, "system")
        assert len(provider.summarized) == 4

    def test_threshold_uses_an_absolute_reserve(self):
        a = agent()
        small = MockProvider(context_window=100_000)
        big = MockProvider(context_window=1_000_000)
        a.usage.context_tokens = 100_000 - tx.RESERVE_TOKENS + 1
        a.usage.measured_entries = len(a.entries)
        assert tx.needs_compaction(a, small)
        assert not tx.needs_compaction(a, big)


class TestTokenEstimate:
    def test_falls_back_to_characters_when_no_figure_was_ever_reported(self):
        a = agent()
        turn(a, "bash", {"command": "x" * 4_000}, "y" * 4_000)
        assert a.usage.context_tokens == 0
        assert tx.estimate_tokens(a) == pytest.approx(2_000, rel=0.05)

    def test_switches_to_the_reported_figure_and_counts_only_what_came_after(self):
        a = agent()
        turn(a, "bash", {"command": "x" * 4_000}, "y" * 4_000)

        a.turns += 1
        tx.record_usage(a, Usage(input_tokens=9_000, output_tokens=1_000, context_tokens=10_000))
        assert tx.estimate_tokens(a) == 10_000

        tx.append_action(a, Action(tool="bash", args={"command": "z" * 4_000}))
        tx.append_result(a, "w" * 4_000)
        assert tx.estimate_tokens(a) == pytest.approx(12_000, rel=0.05)

    def test_a_fresh_figure_supersedes_the_stale_one_plus_its_estimate(self):
        a = agent()
        turn(a, "bash", {"command": "x" * 8_000}, "ok")
        tx.record_usage(a, Usage(context_tokens=10_000))
        turn(a, "bash", {"command": "x" * 8_000}, "ok")
        assert tx.estimate_tokens(a) > 11_000

        tx.record_usage(a, Usage(context_tokens=12_500))
        assert tx.estimate_tokens(a) == 12_500

    def test_a_provider_that_reports_nothing_does_not_freeze_the_figure(self):
        a = agent()
        tx.record_usage(a, Usage(context_tokens=10_000))
        measured = a.usage.measured_entries
        tx.record_usage(a, Usage(input_tokens=500, context_tokens=0))
        assert a.usage.context_tokens == 10_000
        assert a.usage.measured_entries == measured
        assert a.usage.input_tokens == 500

    def test_cumulative_counters_accumulate_while_context_is_a_snapshot(self):
        a = agent()
        tx.record_usage(a, Usage(input_tokens=100, output_tokens=10, cost_usd=0.01, context_tokens=110))
        tx.record_usage(a, Usage(input_tokens=200, output_tokens=20, cost_usd=0.02, context_tokens=330))
        assert (a.usage.input_tokens, a.usage.output_tokens) == (300, 30)
        assert a.usage.cost_usd == pytest.approx(0.03)
        assert a.usage.context_tokens == 330

    def test_compaction_drops_the_stale_figure(self):
        a = agent()
        for n in range(40):
            turn(a, "bash", {"command": f"echo {n}"}, "y" * 500)
        tx.record_usage(a, Usage(context_tokens=180_000))
        provider = MockProvider(context_window=200_000, summary="short")

        assert pytest.importorskip("asyncio").run(tx.compact(a, provider, "system"))
        assert a.usage.context_tokens == 0
        assert tx.estimate_tokens(a) < 5_000


class TestRender:
    def test_lines_are_tagged_and_aligned(self):
        a = agent()
        turn(a, "bash", {"command": "pytest -q"}, "exit 0, 212 passed", inbox=["Suzanne: merge it"])
        lines = tx.render(a).splitlines()
        assert lines[1] == "[inbox]   Suzanne: merge it"
        assert lines[2] == '[turn 1]  bash {"command": "pytest -q"}'
        assert lines[3] == "[result]  exit 0, 212 passed"

    def test_reset_drops_everything_and_keeps_only_the_preprompt(self):
        a = agent()
        turn(a, "bash", {"command": "ls"}, "exit 0")
        tx.reset(a, "You finished work on the parser.")
        assert a.entries == [Entry(kind="note", text="You finished work on the parser.", turn=1, at=a.entries[0].at)]
        assert "ls" not in tx.render(a)
