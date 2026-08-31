"""Kimi as a pure reasoning engine.

Verified against kimi 0.39.1. No schema support and no system-prompt flag, so the system
prompt is prepended and the shape is spelled out in words at the end of the prompt. SHAPE
is the only place the action envelope is stated for this provider - the others get it from
their schema - and left to itself kimi invents its own shape, so the reminder is not
optional; when it still gets it wrong we reprompt once with the parse error and give up.

Its NDJSON is keyed on `role`, not `type`, and it reports no usage at all, which leaves
the transcript's character estimate as the only context figure for this provider.
"""

from __future__ import annotations

from panopticon.providers import _cli
from panopticon.providers.base import Fault, TurnRequest, TurnResponse, parse_action

SHAPE = (
    'Reply with only a JSON object of the form {"tool": "<tool name>", "args": {...}, '
    '"note": "<one short line>"}. No prose, no code fence.'
)


class KimiCLI:
    def __init__(
        self,
        *,
        model: str,
        context_window: int = 262_144,
        timeout: float = _cli.DEFAULT_TIMEOUT,
        bin: str = "kimi",
    ) -> None:
        self.context_window = context_window
        self.model = model
        self.timeout = timeout
        self.bin = bin

    def _argv(self, prompt: str) -> list[str]:
        return [self.bin, "-p", prompt, "--output-format", "stream-json", "-m", self.model]

    async def act(self, req: TurnRequest) -> TurnResponse:
        prompt = f"{req.system}\n\n{req.prompt}\n\n{SHAPE}"
        response = await self._once(prompt, req)
        if response.action is not None or response.error is None:
            return response
        retry = f"{prompt}\n\nYour last reply could not be used: {response.error}\n{SHAPE}"
        return await self._once(retry, req)

    async def _once(self, prompt: str, req: TurnRequest) -> TurnResponse:
        done = await _cli.run(self._argv(prompt), cwd=req.cwd, timeout=self.timeout)
        if done.error:
            return TurnResponse(error=_cli.because(done.error, done.stderr))
        return parse_output(done.stdout, req.tools, code=done.code, stderr=done.stderr)

    async def summarize(self, system: str, text: str) -> str | None:
        done = await _cli.run(self._argv(f"{system}\n\n{text}"), timeout=self.timeout)
        if done.error:
            return None
        message = assistant_text(done.stdout)
        return message or None


def parse_output(stdout: str, tools: list, *, code: int = 0, stderr: str = "") -> TurnResponse:
    message = assistant_text(stdout)
    if not message:
        return TurnResponse(
            error=_cli.because(f"kimi exited {code} with no assistant message", stderr or stdout)
        )
    action, error = parse_action(message, tools)
    return TurnResponse(
        action=action,
        error=error and _cli.because(error, message),
        fault=Fault.MALFORMED if error else None,
    )


def assistant_text(stdout: str) -> str:
    parts = [
        str(event.get("content") or "")
        for event in _cli.ndjson(stdout)
        if event.get("role") == "assistant"
    ]
    return "".join(parts).strip()
