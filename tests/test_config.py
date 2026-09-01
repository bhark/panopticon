"""Provider configuration: what a roster opens with, and what an old config file becomes."""

from __future__ import annotations

import json
from collections import Counter

from panopticon import config as config_mod
from panopticon.model import Level

PROVIDERS = ["claude", "codex", "kimi"]


def levels(count: int, providers: list[str] | None = None, mix=None) -> list[Level]:
    return [level for _, level in config_mod.deal(picks(count, providers, mix))]


def picks(count: int, providers: list[str] | None = None, mix=None):
    return config_mod.roster(count, providers or PROVIDERS, mix)


class TestRoster:
    def test_the_smallest_roster_gets_one_of_each_level(self):
        assert levels(3) == [Level.CAPABLE, Level.BALANCED, Level.FAST]

    def test_six_agents_come_out_one_capable_three_balanced_two_fast(self):
        got = levels(6)
        assert got.count(Level.CAPABLE) == 1
        assert got.count(Level.BALANCED) == 3
        assert got.count(Level.FAST) == 2

    def test_providers_still_go_round_robin(self):
        dealt = [p for p, _ in config_mod.deal(picks(7))]
        assert sorted(Counter(dealt).values()) == [2, 2, 3]

    def test_an_explicit_mix_is_dealt_exactly_and_interleaved(self):
        mix = {(None, Level.FAST): 3, (None, Level.BALANCED): 2, (None, Level.CAPABLE): 1}
        got = levels(6, mix=mix)
        assert got.count(Level.FAST) == 3
        assert got.count(Level.BALANCED) == 2
        assert got.count(Level.CAPABLE) == 1
        assert got[:3] == [Level.CAPABLE, Level.BALANCED, Level.FAST]  # not grouped by level

    def test_a_mix_of_one_level_only_is_honoured(self):
        assert levels(3, mix={(None, Level.FAST): 3}) == [Level.FAST] * 3

    def test_a_mix_that_names_providers_gets_exactly_those(self):
        mix = {("codex", Level.CAPABLE): 2, ("kimi", Level.FAST): 1}
        assert picks(3, mix=mix) == {("codex", Level.CAPABLE): 2, ("kimi", Level.FAST): 1}

    def test_a_named_provider_does_not_move_the_round_robin_on(self):
        mix = {("kimi", Level.CAPABLE): 2, (None, Level.FAST): 2}
        assert picks(4, mix=mix) == {
            ("kimi", Level.CAPABLE): 2,
            ("claude", Level.FAST): 1,
            ("codex", Level.FAST): 1,
        }


class TestLoad:
    def test_a_config_written_before_levels_existed_gets_the_defaults(self, tmp_path, monkeypatch):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"providers": {"claude": {"kind": "claude_cli", "model": "sonnet"}}})
        )
        monkeypatch.setattr(config_mod, "CONFIG_FILE", path)
        cfg = config_mod.load()
        assert cfg.providers["claude"]["levels"] == config_mod.DEFAULTS["claude"]["levels"]
        assert cfg.providers["claude"]["model"] == "sonnet"  # the rest of the entry is untouched

    def test_levels_the_human_set_are_left_alone(self, tmp_path, monkeypatch):
        mine = {"fast": {"model": "fable"}}
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"providers": {"claude": {"kind": "claude_cli", "levels": mine}}})
        )
        monkeypatch.setattr(config_mod, "CONFIG_FILE", path)
        assert config_mod.load().providers["claude"]["levels"] == mine

    def test_a_loaded_config_does_not_share_nested_defaults(self, tmp_path, monkeypatch):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"providers": {}}))
        monkeypatch.setattr(config_mod, "CONFIG_FILE", path)
        cfg = config_mod.load()
        cfg.providers["claude"]["levels"]["fast"]["model"] = "mutated"
        assert config_mod.DEFAULTS["claude"]["levels"]["fast"]["model"] == "haiku"
