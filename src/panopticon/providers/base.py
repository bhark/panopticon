"""The provider port. Every provider answers the same question: one action, given a prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from panopticon.model import Action, ToolSpec, Usage


@dataclass(slots=True)
class TurnRequest:
    agent: str
    system: str
    prompt: str
    tools: list[ToolSpec]
    cwd: str | None = None


@dataclass(slots=True)
class TurnResponse:
    """A provider never raises; a failure comes back as `error`."""

    action: Action | None = None
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    raw: str = ""


class Provider(Protocol):
    key: str
    context_window: int

    async def act(self, req: TurnRequest) -> TurnResponse: ...

    async def summarize(self, system: str, text: str) -> str | None:
        """One-shot, no history, no cache write. None on failure."""
        ...


def action_schema(tools: list[ToolSpec]) -> dict[str, Any]:
    """Flat {tool, args} rather than a per-tool oneOf: every provider accepts it."""
    return {
        "type": "object",
        "properties": {
            "note": {"type": "string", "description": "One short line. Optional."},
            "tool": {"type": "string", "enum": [t.name for t in tools]},
            "args": {"type": "object"},
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def strict_action_schema(tools: list[ToolSpec]) -> dict[str, Any]:
    """Same action, for providers on OpenAI strict mode.

    Strict mode demands additionalProperties:false and every property required on every
    object, which cannot express a free-form `args`, so args travels as a JSON string.
    Verified: codex rejects the permissive schema outright with invalid_json_schema.
    """
    return {
        "type": "object",
        "properties": {
            "note": {"type": "string", "description": "One short line, may be empty."},
            "tool": {"type": "string", "enum": [t.name for t in tools]},
            "args": {"type": "string", "description": "Arguments as a JSON object, as a string."},
        },
        "required": ["note", "tool", "args"],
        "additionalProperties": False,
    }


def parse_action(text: str, tools: list[ToolSpec]) -> tuple[Action | None, str | None]:
    """Pull one action out of model output. Returns (action, error)."""
    import json
    import re

    blob = text.strip()
    if not blob:
        return None, "empty response"
    if not blob.startswith("{"):
        # fenced or prose-wrapped: take the last balanced object in the text
        match = None
        for m in re.finditer(r"\{", blob):
            depth, i = 0, m.start()
            for j in range(i, len(blob)):
                if blob[j] == "{":
                    depth += 1
                elif blob[j] == "}":
                    depth -= 1
                    if depth == 0:
                        match = blob[i : j + 1]
                        break
        if match is None:
            return None, "no JSON object in response"
        blob = match
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return None, f"response was not valid JSON: {exc}"
    if not isinstance(data, dict) or "tool" not in data:
        return None, "response had no 'tool' field"
    args = data.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            return None, f"'args' was a string but not valid JSON: {exc}"
    if not isinstance(args, dict):
        return None, "'args' was not an object"
    names = {t.name for t in tools}
    if data["tool"] not in names:
        return None, f"unknown tool {data['tool']!r}; available: {', '.join(sorted(names))}"
    return Action(tool=str(data["tool"]), args=args, note=str(data.get("note") or "")), None
