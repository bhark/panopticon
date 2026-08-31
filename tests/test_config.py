"""Provider configuration: what a roster opens with, and what an old config file becomes."""

from __future__ import annotations

import json

from panopticon import config as config_mod
from panopticon.model import Level

PROVIDERS = ["claude", "codex", "kimi"]


def levels(count: int, providers: list[str] | None = None, mix=None) -> list[Level]:
    return [level for _, level in config_mod.spread(count, providers or PROVIDERS, mix)]


class TestSpread:
    def test_the_smallest_roster_gets_one_of_each_level(self):
        assert levels(3) == [Level.CAPABLE, Level.BALANCED, Level.FAST]

    def test_six_agents_come_out_one_capable_three_balanced_two_fast(self):
        got = levels(6)
        assert got.count(Level.CAPABLE) == 1
        assert got.count(Level.BALANCED) == 3
        assert got.count(Level.FAST) == 2

    def test_providers_still_go_round_robin(self):
        assert [p for p, _ in config_mod.spread(7, PROVIDERS)] == [
            "claude",
            "codex",
            "kimi",
            "claude",
            "codex",
            "kimi",
            "claude",
        ]

    def test_an_explicit_mix_is_dealt_exactly_and_interleaved(self):
        got = levels(6, mix={Level.FAST: 3, Level.BALANCED: 2, Level.CAPABLE: 1})
        assert got.count(Level.FAST) == 3
        assert got.count(Level.BALANCED) == 2
        assert got.count(Level.CAPABLE) == 1
        assert got[:3] == [Level.CAPABLE, Level.BALANCED, Level.FAST]  # not grouped by level

    def test_a_mix_of_one_level_only_is_honoured(self):
        assert levels(3, mix={Level.FAST: 3}) == [Level.FAST] * 3


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
