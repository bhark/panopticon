"""One config entry in, one provider out. Adding a provider means one adapter and one line here."""

from __future__ import annotations

from typing import Any

from panopticon.model import Level
from panopticon.providers.api_openrouter import OpenRouter
from panopticon.providers.base import Provider
from panopticon.providers.cli_claude import ClaudeCLI
from panopticon.providers.cli_codex import CodexCLI
from panopticon.providers.cli_kimi import KimiCLI
from panopticon.providers.mock import MockProvider

_KINDS = {
    "claude_cli": ClaudeCLI,
    "codex_cli": CodexCLI,
    "kimi_cli": KimiCLI,
    "openrouter": OpenRouter,
    "mock": MockProvider,
}

_CLI_ARGS = ("model", "context_window", "timeout", "bin")

_ACCEPTS = {
    "claude_cli": (*_CLI_ARGS, "effort"),
    "codex_cli": (*_CLI_ARGS, "effort"),
    "kimi_cli": _CLI_ARGS,
    "openrouter": ("model", "context_window", "timeout", "api_key_env", "base_url", "effort"),
    "mock": ("context_window",),
}


def build(key: str, cfg: dict[str, Any], level: Level = Level.BALANCED) -> Provider:
    """One engine: the provider entry with its overlay for `level` merged over it."""
    kind = cfg.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"unknown provider kind {kind!r} for {key!r}; known: {', '.join(_KINDS)}")
    overlay = (cfg.get("levels") or {}).get(str(level), {})
    # a mistyped overlay key would otherwise be dropped and run the base model in silence
    if unknown := sorted(set(overlay) - set(_ACCEPTS[kind])):
        raise ValueError(f"provider {key!r} at {level}: {kind} takes no {', '.join(unknown)}")
    merged = cfg | overlay
    if kind != "mock" and not merged.get("model"):
        raise ValueError(f"provider {key!r} has no model")
    kwargs = {name: merged[name] for name in _ACCEPTS[kind] if merged.get(name) is not None}
    return _KINDS[kind](**kwargs)


def build_levels(key: str, cfg: dict[str, Any]) -> dict[Level, Provider]:
    """Every level a provider can be asked for. Levels it does not configure resolve alike."""
    if unknown := sorted(set(cfg.get("levels") or {}) - {str(level) for level in Level}):
        raise ValueError(f"provider {key!r} has unknown levels: {', '.join(unknown)}")
    return {level: build(key, cfg, level) for level in Level}
