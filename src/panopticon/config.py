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

MIN_AGENTS = 3  # a truth needs two 'true' verdicts and cannot be judged by its submitter

Pick = tuple[str, Level]  # a provider entry and the level it runs at
Wanted = tuple[str | None, Level]  # a pick, or a level with the provider still open


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


def roster(
    agent_count: int, providers: list[str], mix: dict[Wanted, int] | None = None
) -> dict[Pick, int]:
    """How many agents on each provider and level.

    Without a mix, levels follow MIX and providers go round-robin, so a small roster can
    leave a provider off a level. That is what a mix is for; entries in it that name no
    provider still go round-robin.
    """
    wanted = _deal(mix) if mix else [(None, MIX[i % len(MIX)]) for i in range(agent_count)]
    out: dict[Pick, int] = {}
    turn = 0
    for named, level in wanted:
        provider = named
        if provider is None:
            provider = providers[turn % len(providers)]
            turn += 1
        out[provider, level] = out.get((provider, level), 0) + 1
    return out


def deal(picked: dict[Pick, int]) -> list[Pick]:
    """One entry per agent."""
    return [pick for pick, count in picked.items() for _ in range(count)]


def _deal(mix: dict[Wanted, int]) -> list[Wanted]:
    """An explicit mix, interleaved rather than grouped, so it does not line up with a provider."""
    left = {key: count for key, count in sorted(mix.items(), key=_order) if count > 0}
    out: list[Wanted] = []
    while left:
        for key in list(left):
            out.append(key)
            left[key] -= 1
            if not left[key]:
                del left[key]
    return out


def _order(item: tuple[Wanted, int]) -> tuple[int, str]:
    (provider, level), _ = item
    return MIX_ORDER.index(level), provider or ""
