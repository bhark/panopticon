"""A scripted provider. What makes a full harness lifecycle testable without spending tokens."""

from __future__ import annotations

from collections.abc import Callable

from panopticon.model import Action, Usage
from panopticon.providers.base import TurnRequest, TurnResponse

Script = dict[str, list[Action]] | list[Action] | Callable[[TurnRequest], Action | None]


class MockProvider:
    """Replays a script, per agent name or as one shared list, or defers to a callable.

    Every request is kept in `calls`, so a test can assert on the prompt an agent was
    actually shown rather than on the action it was handed back.
    """

    def __init__(
        self,
        script: Script | None = None,
        *,
        key: str = "mock",
        context_window: int = 200_000,
        summary: str | None = "summary of earlier turns",
        context_tokens: int = 0,
    ) -> None:
        self.key = key
        self.context_window = context_window
        self.summary = summary
        self.context_tokens = context_tokens
        self.calls: list[TurnRequest] = []
        self.summarized: list[str] = []
        self._script = script if script is not None else {}

    async def act(self, req: TurnRequest) -> TurnResponse:
        self.calls.append(req)
        usage = Usage(context_tokens=self.context_tokens)
        action = self._next(req)
        if action is None:
            return TurnResponse(error=f"mock script exhausted for {req.agent}", usage=usage)
        return TurnResponse(action=action, usage=usage, raw=f"{action.tool} {action.args}")

    async def summarize(self, system: str, text: str) -> str | None:
        self.summarized.append(text)
        return self.summary

    def _next(self, req: TurnRequest) -> Action | None:
        script = self._script
        if callable(script):
            return script(req)
        queue = script.get(req.agent, []) if isinstance(script, dict) else script
        return queue.pop(0) if queue else None
