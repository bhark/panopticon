"""Provider configuration, at ~/.panopticon/config.json."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("PANOPTICON_HOME", Path.home() / ".panopticon"))
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS: dict[str, dict] = {
    "claude": {
        "kind": "claude_cli",
        "model": "sonnet",
        "context_window": 200_000,
        "enabled": True,
    },
    "codex": {
        "kind": "codex_cli",
        "model": "gpt-5.6-sol",
        "context_window": 272_000,
        "enabled": True,
    },
    "kimi": {
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
        "api_key_env": "OPENROUTER_API_KEY",
        "enabled": False,
    },
}


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
    # a provider added in a later version is missing from an older config file
    for key, default in DEFAULTS.items():
        cfg.providers.setdefault(key, dict(default))
    return cfg


def save(cfg: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(cfg), indent=2) + "\n")


def spread(agent_count: int, providers: list[str]) -> list[str]:
    """The agent budget, distributed as evenly as possible across providers."""
    return [providers[i % len(providers)] for i in range(agent_count)]
