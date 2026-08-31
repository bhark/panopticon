"""Tool registry, argument validation and dispatch."""

from __future__ import annotations

from typing import Any

from panopticon.model import Action, ActionResult, ArgSpec, Handler, ToolCtx, ToolSpec

REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, **args: ArgSpec):
    def wrap(handler: Handler) -> Handler:
        REGISTRY[name] = ToolSpec(name=name, description=description, args=args, handler=handler)
        return handler

    return wrap


def specs(names: list[str]) -> list[ToolSpec]:
    return [REGISTRY[n] for n in names if n in REGISTRY]


_CHECK = {
    "string": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "string[]": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
}


def validate(spec: ToolSpec, args: dict[str, Any]) -> list[str]:
    problems = []
    for key, arg in spec.args.items():
        if key not in args or args[key] is None:
            if arg.required:
                problems.append(f"{key}: missing")
            continue
        value = args[key]
        # models routinely send "3" for an int; coerce rather than reject
        if arg.type == "int" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            args[key] = value = int(value)
        if not _CHECK[arg.type](value):
            problems.append(f"{key}: expected {arg.type}")
    for key in args:
        if key not in spec.args:
            problems.append(f"{key}: unknown argument")
    return problems


async def dispatch(ctx: ToolCtx, action: Action, allowed: list[str]) -> ActionResult:
    spec = REGISTRY.get(action.tool)
    if spec is None or action.tool not in allowed:
        return ActionResult.fail(
            f"{action.tool} is not available to you right now. Available: {', '.join(allowed)}"
        )
    args = dict(action.args)
    if problems := validate(spec, args):
        return ActionResult.fail(
            f"{action.tool}: {'; '.join(problems)}. You sent: {action.args!r}"
        )
    try:
        return await spec.handler(ctx, args)
    except Exception as exc:  # a tool bug must reach the agent, not kill its loop
        return ActionResult.fail(f"{action.tool} failed: {type(exc).__name__}: {exc}")


from panopticon.tools import board, comms, jury, knowledge, lifecycle, wait, workspace  # noqa: E402,F401
