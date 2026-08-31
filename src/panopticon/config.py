"""Provider configuration, at ~/.panopticon/config.json."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from panopticon.model import Level

CONFIG_DIR = Path(os.environ.get("PANOPTICON_HOME", Path.home() / ".panopticon"))
CONFIG_FILE = CONFIG_DIR / "config.json"

# a level overlays these args onto the provider entry; an absent one means the entry as written
DEFAULTS: dict[str, dict] = {
    "claude": {
        "kind": "claude_cli",
        "model": "sonnet",
        "context_window": 200_000,
        "levels": {"fast": {"model": "haiku"}, "capable": {"model": "opus"}},
        "enabled": True,
    },
    "codex": {
        "kind": "codex_cli",
        "model": "gpt-5.6-sol",
        "context_window": 272_000,
        "levels": {
            "fast": {"effort": "low"},
            "balanced": {"effort": "medium"},
            "capable": {"effort": "high"},
        },
        "enabled": True,
    },
    "kimi": {
        # no effort control and one model: every level is the same engine
        "kind": "kimi_cli",
        "model": "kimi-code/k3",
        "context_window": 262_144,
        "enabled": True,
    },
    "openrouter": {
        "kind": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-sonnet-4.5",
        "context_window": 200_000,
        "levels": {
            "fast": {"effort": "low"},
            "balanced": {"effort": "medium"},
            "capable": {"effort": "high"},
        },
        "api_key_env": "OPENROUTER_API_KEY",
        "enabled": False,
    },
}

# one capable and two fast per three balanced; a roster of three still gets one of each
MIX = (Level.CAPABLE, Level.BALANCED, Level.FAST, Level.BALANCED, Level.FAST, Level.BALANCED)
MIX_ORDER = (Level.CAPABLE, Level.BALANCED, Level.FAST)


@dataclass(slots=True)
class Config:
    providers: dict[str, dict] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    max_concurrent_turns: int = 6

    def enabled(self) -> dict[str, dict]:
        return {k: v for k, v in self.providers.items() if v.get("enabled")}

    def usable(self) -> tuple[dict[str, dict], list[str]]:
        """Enabled providers whose credentials are actually present, plus why any were skipped."""
        ok, skipped = {}, []
        for key, cfg in self.enabled().items():
            env = cfg.get("api_key_env")
            if env and not os.environ.get(env):
                skipped.append(f"{key}: ${env} is not set")
                continue
            ok[key] = cfg
        return ok, skipped


def load() -> Config:
    if not CONFIG_FILE.exists():
        return Config()
    data = json.loads(CONFIG_FILE.read_text())
    known = {f.name for f in fields(Config)}  # a setting a later version dropped must not raise
    cfg = Config(**{k: v for k, v in data.items() if k in known})
    # a provider, or its levels, added in a later version is missing from an older config file
    for key, default in DEFAULTS.items():
        entry = cfg.providers.setdefault(key, copy.deepcopy(default))
        if "levels" in default:
            entry.setdefault("levels", copy.deepcopy(default["levels"]))
    return cfg


def save(cfg: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(cfg), indent=2) + "\n")


def spread(
    agent_count: int, providers: list[str], mix: dict[Level, int] | None = None
) -> list[tuple[str, Level]]:
    """One provider and one level per agent: providers round-robin, levels follow the mix.

    The two cycles are independent, so a small roster can leave a provider off a level.
    That is what --mix is for.
    """
    levels = _deal(mix) if mix else [MIX[i % len(MIX)] for i in range(agent_count)]
    return [(providers[i % len(providers)], level) for i, level in enumerate(levels)]


def _deal(mix: dict[Level, int]) -> list[Level]:
    """An explicit mix, interleaved rather than grouped, so it does not line up with a provider."""
    left = {level: mix.get(level, 0) for level in MIX_ORDER}
    out: list[Level] = []
    while any(count > 0 for count in left.values()):
        for level in MIX_ORDER:
            if left[level] > 0:
                out.append(level)
                left[level] -= 1
    return out
