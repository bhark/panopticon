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


class Provider(Protocol):
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


def _last_object(blob: str) -> str | None:
    """Last balanced top-level {...}, counting braces outside string literals only.

    Nested objects are skipped rather than scanned as candidates of their own, and a brace
    inside a string stays inside it: agents pass shell commands through `args`.
    """
    found, pos = None, 0
    while (start := blob.find("{", pos)) != -1:
        depth, in_string, escaped, end = 0, False, False, -1
        for j in range(start, len(blob)):
            char = blob[j]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end == -1:
            return found
        found, pos = blob[start:end], end
    return found


def parse_action(text: str, tools: list[ToolSpec]) -> tuple[Action | None, str | None]:
    """Pull one action out of model output. Returns (action, error)."""
    import json

    blob = text.strip()
    if not blob:
        return None, "empty response"
    if not blob.startswith("{"):
        # fenced or prose-wrapped: take the last balanced top-level object in the text
        match = _last_object(blob)
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
