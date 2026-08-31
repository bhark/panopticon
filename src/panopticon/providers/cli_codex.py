"""Codex as a pure reasoning engine.

Verified against codex 0.151.0. There is no `--append-system-prompt`, so the system
prompt is prepended to the user prompt. `--output-schema` goes through OpenAI strict
mode, which rejects a free-form `args` object, so this adapter uses the strict schema
where args arrives as a JSON string; `parse_action` decodes it.
"""

from __future__ import annotations

import json
import os
import tempfile

from panopticon.model import Usage
from panopticon.providers import _cli
from panopticon.providers.base import TurnRequest, TurnResponse, parse_action, strict_action_schema


class CodexCLI:
    def __init__(
        self,
        key: str,
        *,
        model: str,
        context_window: int = 272_000,
        timeout: float = _cli.DEFAULT_TIMEOUT,
        bin: str = "codex",
    ) -> None:
        self.key = key
        self.context_window = context_window
        self.model = model
        self.timeout = timeout
        self.bin = bin

    def _argv(self, prompt: str, cwd: str | None, schema_path: str | None) -> list[str]:
        argv = [
            self.bin,
            "exec",
            "--json",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "-C",
            cwd or os.getcwd(),
            "-m",
            self.model,
        ]
        if schema_path is not None:
            argv += ["--output-schema", schema_path]
        return [*argv, prompt]

    async def act(self, req: TurnRequest) -> TurnResponse:
        schema = strict_action_schema(req.tools)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(schema, fh)
            path = fh.name
        try:
            argv = self._argv(f"{req.system}\n\n{req.prompt}", req.cwd, path)
            done = await _cli.run(argv, cwd=req.cwd, timeout=self.timeout)
        finally:
            os.unlink(path)
        if done.error:
            return TurnResponse(error=done.error, raw=_cli.tail(done.stderr))
        return parse_output(done.stdout, req.tools, code=done.code, stderr=done.stderr)

    async def summarize(self, system: str, text: str) -> str | None:
        argv = self._argv(f"{system}\n\n{text}", None, None)
        done = await _cli.run(argv, timeout=self.timeout)
        if done.error:
            return None
        message, error, _ = _scan(done.stdout)
        return message if message and not error else None


def parse_output(stdout: str, tools: list, *, code: int = 0, stderr: str = "") -> TurnResponse:
    message, error, usage = _scan(stdout)
    if error:
        return TurnResponse(error=error, usage=usage, raw=error)
    if not message:
        return TurnResponse(
            error=f"codex exited {code} with no agent message",
            usage=usage,
            raw=_cli.tail(stderr or stdout),
        )
    action, parse_error = parse_action(message, tools)
    return TurnResponse(action=action, usage=usage, error=parse_error, raw=message)


def _scan(stdout: str) -> tuple[str, str | None, Usage]:
    # item-level errors are warnings codex runs on through, so they count only if nothing came back
    message, hard, soft, usage = "", None, None, Usage()
    for event in _cli.ndjson(stdout):
        kind = event.get("type")
        if kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                message = str(item.get("text") or "")
            elif item.get("type") == "error":
                soft = soft or str(item.get("message") or "")
        elif kind == "error":
            hard = str(event.get("message") or "")
        elif kind == "turn.failed":
            hard = str((event.get("error") or {}).get("message") or "turn failed")
        elif kind == "turn.completed":
            usage = _usage(event.get("usage") or {})
            hard = None
    return message, hard or (soft if not message else None), usage


def _usage(u: dict) -> Usage:
    got = Usage(
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        cache_read=u.get("cached_input_tokens", 0),
        cache_write=u.get("cache_write_input_tokens", 0),
    )
    got.context_tokens = got.input_tokens + got.output_tokens
    return got
