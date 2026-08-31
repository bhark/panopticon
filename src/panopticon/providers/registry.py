"""One config entry in, one provider out. Adding a provider means one adapter and one line here."""

from __future__ import annotations

from typing import Any

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

# what each engine actually gives us, so config only has to say so when it differs
_WINDOWS = {
    "claude_cli": 200_000,
    "codex_cli": 272_000,
    "kimi_cli": 262_144,
    "openrouter": 128_000,
    "mock": 200_000,
}

_ACCEPTS = {
    "claude_cli": ("model", "context_window", "timeout", "bin"),
    "codex_cli": ("model", "context_window", "timeout", "bin"),
    "kimi_cli": ("model", "context_window", "timeout", "bin"),
    "openrouter": ("model", "context_window", "timeout", "api_key_env", "base_url"),
    "mock": ("context_window",),
}


def build(key: str, cfg: dict[str, Any]) -> Provider:
    kind = cfg.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"unknown provider kind {kind!r} for {key!r}; known: {', '.join(_KINDS)}")
    if kind != "mock" and not cfg.get("model"):
        raise ValueError(f"provider {key!r} has no model")
    kwargs = {name: cfg[name] for name in _ACCEPTS[kind] if cfg.get(name) is not None}
    kwargs.setdefault("context_window", _WINDOWS[kind])
    return _KINDS[kind](key, **kwargs)
