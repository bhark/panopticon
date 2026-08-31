"""Claude Code as a pure reasoning engine: its own tools off, our schema on.

Every flag below was verified against claude 2.1.252 and each one is load-bearing.
`--output-format stream-json` hard errors without `--verbose`. `--setting-sources ""` is
what stops the user's settings.json hooks from injecting kilobytes into every turn.
`--tools ""` disables the built-in tools but not MCP ones, so the empty `--mcp-config`
and `--strict-mcp-config` are both needed. There is no `--max-turns`.
"""

from __future__ import annotations

import json

from panopticon.model import Usage
from panopticon.providers import _cli
from panopticon.providers.base import TurnRequest, TurnResponse, action_schema, parse_action


class ClaudeCLI:
    def __init__(
        self,
        *,
        model: str,
        context_window: int = 200_000,
        timeout: float = _cli.DEFAULT_TIMEOUT,
        bin: str = "claude",
    ) -> None:
        self.context_window = context_window
        self.model = model
        self.timeout = timeout
        self.bin = bin

    def _argv(self, system: str, prompt: str, schema: dict | None) -> list[str]:
        argv = [
            self.bin,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "",
            "--system-prompt",
            system,
            "--model",
            self.model,
            "--no-session-persistence",
        ]
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema)]
        return argv

    async def act(self, req: TurnRequest) -> TurnResponse:
        argv = self._argv(req.system, req.prompt, action_schema(req.tools))
        done = await _cli.run(argv, cwd=req.cwd, timeout=self.timeout)
        if done.error:
            return TurnResponse(error=_cli.because(done.error, done.stderr))
        return parse_output(done.stdout, req.tools) or TurnResponse(
            error=_cli.because(
                f"claude exited {done.code} with no result event", done.stderr or done.stdout
            )
        )

    async def summarize(self, system: str, text: str) -> str | None:
        argv = self._argv(system, text, None)
        done = await _cli.run(argv, timeout=self.timeout)
        if done.error:
            return None
        event = _result_event(done.stdout)
        if event is None or event.get("is_error"):
            return None
        result = event.get("result")
        return result if isinstance(result, str) and result.strip() else None


def parse_output(stdout: str, tools: list) -> TurnResponse | None:
    """None when the stream carried no result event at all."""
    event = _result_event(stdout)
    if event is None:
        return None
    usage = _usage(event)
    if event.get("is_error"):
        reason = event.get("terminal_reason") or event.get("subtype") or "error"
        return TurnResponse(error=f"{reason}: {event.get('result')}", usage=usage)
    # structured_output is the same object as `result`, already decoded
    structured = event.get("structured_output")
    raw = json.dumps(structured) if isinstance(structured, dict) else str(event.get("result") or "")
    action, error = parse_action(raw, tools)
    return TurnResponse(action=action, usage=usage, error=error and _cli.because(error, raw))


def _result_event(stdout: str) -> dict | None:
    found = None
    for event in _cli.ndjson(stdout):
        if event.get("type") == "result":
            found = event
    return found


def _usage(event: dict) -> Usage:
    u = event.get("usage") or {}
    got = Usage(
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        cache_read=u.get("cache_read_input_tokens", 0),
        cache_write=u.get("cache_creation_input_tokens", 0),
        cost_usd=event.get("total_cost_usd", 0.0) or 0.0,
    )
    got.context_tokens = got.input_tokens + got.output_tokens + got.cache_read + got.cache_write
    return got
