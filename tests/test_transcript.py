from __future__ import annotations

import pytest

from panopticon import transcript as tx
from panopticon.model import Action, Agent, QueueItem, Usage
from panopticon.providers.mock import MockProvider


def agent(name: str = "Suzanne") -> Agent:
    a = Agent(name=name, provider="mock")
    tx.reset(a, "You just joined. Nothing is waiting for you.")
    return a


def turn(a: Agent, tool: str, args: dict, result: str, inbox: tuple[str, ...] = ()) -> None:
    a.turns += 1
    if inbox:
        tx.append_inbox(a, [QueueItem(kind="dm", text=t) for t in inbox])
    tx.append_action(a, Action(tool=tool, args=args))
    tx.append_result(a, result)


def stable_part(prompt: str) -> str:
    """Everything a provider should still have cached: the prompt minus its trailing nudge."""
    return prompt.removesuffix(tx.TAIL).rstrip("\n")


def long_run(a: Agent, turns: int = 100) -> None:
    for n in range(1, turns + 1):
        turn(a, "bash", {"command": f"echo {n}"}, f"exit 0, run {n}: " + "y" * 500)


class TestAppendOnly:
    def test_prefix_holds_across_a_long_run_of_interleaved_inbox_and_results(self):
        a = agent()
        previous = tx.render(a)
        for n in range(1, 120):
            inbox = (f"Alex: ping {n}",) if n % 3 == 0 else ()
            if n % 7 == 0:
                inbox += ("shoutboard: two new messages",)
            turn(a, "bash", {"command": f"pytest -q -k t{n}"}, f"exit 0, {n} passed", inbox)
            current = tx.render(a)
            assert current.startswith(stable_part(previous)), f"prefix broke at turn {n}"
            previous = current

    def test_inbox_is_folded_where_it_was_consumed_not_re_emitted_at_the_tail(self):
        a = agent()
        turn(a, "wait", {}, "woken", inbox=("Alex: start on the parser",))
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
        turn(a, "wait", {}, "woken", inbox=("Alex: hi", "Mira: hi"))
        after = tx.render(a)
        assert after.startswith(before)
        assert after.index("Alex: hi") > after.index("[turn 1]")

    async def test_prefix_resumes_from_the_new_epoch_after_a_compaction(self):
        a = agent()
        long_run(a, turns=200)
        before = tx.render(a)
        provider = MockProvider(context_window=40_000)
        assert tx.needs_compaction(a, provider)

        assert await tx.compact(a, provider, "you are Suzanne")

        after = tx.render(a)
        assert not after.startswith(stable_part(before))  # the one deliberate break
        assert "[summary]" in after
        for n in range(200, 240):
            turn(a, "bash", {"command": f"echo {n}"}, "ok")
            current = tx.render(a)
            assert current.startswith(stable_part(after)), f"prefix broke at turn {n}"
            after = current


class TestCompaction:
    @pytest.mark.parametrize("keep", [200, 800, 2_000, 6_000])
    def test_cut_never_orphans_a_result_from_its_action(self, monkeypatch, keep):
        a = agent()
        long_run(a)
        monkeypatch.setattr(tx, "KEEP_TOKENS", keep)
        cut = tx._cut_point(a.entries)
        assert 0 < cut < len(a.entries)
        assert a.entries[cut].kind != "result"

    async def test_compaction_keeps_the_tail_and_drops_the_head(self):
        a = agent()
        long_run(a)
        head_line = a.entries[2].text
        tail_line = a.entries[-1].text
        provider = MockProvider(context_window=40_000, summary="did 100 turns of echo")

        assert await tx.compact(a, provider, "system")

        rendered = tx.render(a)
        assert "did 100 turns of echo" in rendered
        assert head_line not in rendered
        assert tail_line in rendered
        assert provider.summarized and "1. Goal" in provider.summarized[0]

    async def test_the_summarizer_is_told_it_has_one_turn_and_no_tools(self):
        a = agent()
        long_run(a)
        seen: list[str] = []

        class Recorder(MockProvider):
            async def summarize(self, system: str, text: str) -> str:
                seen.append(system)
                return "ok"

        assert await tx.compact(a, Recorder(context_window=40_000), "you are Suzanne")
        assert "one turn" in seen[0]
        assert "Do not call tools" in seen[0]
        assert "you are Suzanne" in seen[0]

    async def test_breaker_stops_after_three_consecutive_failures(self):
        a = agent()
        long_run(a)
        provider = MockProvider(context_window=40_000, summary=None)

        for _ in range(3):
            assert await tx.compact(a, provider, "system") is False
        assert len(provider.summarized) == 3

        assert await tx.compact(a, provider, "system") is False
        assert len(provider.summarized) == 3, "breaker should stop calling the provider"

    async def test_a_success_clears_the_strike_count(self):
        a = agent()
        long_run(a)
        failing = MockProvider(context_window=40_000, summary=None)
        assert await tx.compact(a, failing, "system") is False
        assert await tx.compact(a, failing, "system") is False

        assert await tx.compact(a, MockProvider(context_window=40_000, summary="ok"), "system")

        long_run(a)
        assert await tx.compact(a, failing, "system") is False
        assert len(failing.summarized) == 3, "strikes should have restarted from zero"

    async def test_reset_clears_the_strike_count_too(self):
        a = agent()
        long_run(a)
        provider = MockProvider(context_window=40_000, summary=None)
        for _ in range(3):
            await tx.compact(a, provider, "system")

        tx.reset(a, "fresh start")
        long_run(a)
        await tx.compact(a, provider, "system")
        assert len(provider.summarized) == 4

    async def test_the_breaker_is_per_agent(self):
        one, two = agent("Suzanne"), agent("Alex")
        long_run(one)
        long_run(two)
        provider = MockProvider(context_window=40_000, summary=None)
        for _ in range(4):
            await tx.compact(one, provider, "system")
        await tx.compact(two, provider, "system")
        assert len(provider.summarized) == 4, "Alex should not inherit Suzanne's strikes"

    def test_threshold_uses_an_absolute_reserve_not_a_share_of_the_window(self):
        a = agent()
        a.usage.context_tokens = 100_000 - tx.RESERVE_TOKENS + 1
        a.usage.measured_entries = len(a.entries)
        assert tx.needs_compaction(a, MockProvider(context_window=100_000))
        assert not tx.needs_compaction(a, MockProvider(context_window=1_000_000))


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
        tx.note_usage(a, Usage(input_tokens=9_000, output_tokens=1_000, context_tokens=10_000))
        assert tx.estimate_tokens(a) == 10_000

        tx.append_action(a, Action(tool="bash", args={"command": "z" * 4_000}))
        tx.append_result(a, "w" * 4_000)
        assert tx.estimate_tokens(a) == pytest.approx(12_000, rel=0.05)

    def test_a_fresh_figure_supersedes_the_stale_one_and_its_estimate(self):
        a = agent()
        turn(a, "bash", {"command": "x" * 8_000}, "ok")
        tx.note_usage(a, Usage(context_tokens=10_000))
        turn(a, "bash", {"command": "x" * 8_000}, "ok")
        assert tx.estimate_tokens(a) > 11_000

        tx.note_usage(a, Usage(context_tokens=12_500))
        assert tx.estimate_tokens(a) == 12_500

    def test_a_provider_that_reports_no_context_does_not_freeze_the_figure(self):
        a = agent()
        tx.note_usage(a, Usage(context_tokens=10_000))
        measured = a.usage.measured_entries
        tx.note_usage(a, Usage(input_tokens=500, context_tokens=0))
        assert a.usage.context_tokens == 10_000
        assert a.usage.measured_entries == measured
        assert a.usage.input_tokens == 500

    def test_counters_accumulate_while_context_stays_a_snapshot(self):
        a = agent()
        tx.note_usage(
            a, Usage(input_tokens=100, output_tokens=10, cost_usd=0.01, context_tokens=110)
        )
        tx.note_usage(
            a, Usage(input_tokens=200, output_tokens=20, cost_usd=0.02, context_tokens=330)
        )
        assert (a.usage.input_tokens, a.usage.output_tokens) == (300, 30)
        assert a.usage.cost_usd == pytest.approx(0.03)
        assert a.usage.context_tokens == 330

    async def test_compaction_drops_the_stale_figure(self):
        a = agent()
        long_run(a)
        tx.note_usage(a, Usage(context_tokens=180_000))

        assert await tx.compact(a, MockProvider(context_window=200_000, summary="short"), "system")
        assert a.usage.context_tokens == 0
        assert tx.estimate_tokens(a) < 10_000


class TestRender:
    def test_lines_are_tagged_and_aligned(self):
        a = agent()
        turn(a, "bash", {"command": "pytest -q"}, "exit 0, 212 passed", ("Suzanne: merge it",))
        lines = tx.render(a).splitlines()
        assert lines[1] == "[inbox]   Suzanne: merge it"
        assert lines[2] == '[turn 1]  bash {"command": "pytest -q"}'
        assert lines[3] == "[result]  exit 0, 212 passed"
        assert lines[-1] == tx.TAIL

    def test_a_note_rides_along_on_the_action_line(self):
        a = agent()
        a.turns = 4
        tx.append_action(a, Action(tool="wait", args={"minutes": 5}, note="nothing to do"))
        assert '[turn 4]  wait {"minutes": 5}  // nothing to do' in tx.render(a)

    def test_reset_drops_everything_and_keeps_only_the_preprompt(self):
        a = agent()
        turn(a, "bash", {"command": "git status"}, "exit 0")
        tx.reset(a, "You finished work on the parser.")
        assert [(e.kind, e.text) for e in a.entries] == [
            ("note", "You finished work on the parser.")
        ]
        assert "git status" not in tx.render(a)
