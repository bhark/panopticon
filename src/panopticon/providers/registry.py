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

_CLI_ARGS = ("model", "context_window", "timeout", "bin")

_ACCEPTS = {
    "claude_cli": _CLI_ARGS,
    "codex_cli": _CLI_ARGS,
    "kimi_cli": _CLI_ARGS,
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
    return _KINDS[kind](**kwargs)
