"""Tool registry, argument validation and dispatch.

A tool declares where it can be reached from, so availability is a query against the
registry rather than a table someone has to remember to edit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from panopticon.model import (
    Action,
    ActionResult,
    Agent,
    ArgSpec,
    Handler,
    Harness,
    Situation,
    ToolCtx,
    ToolSpec,
)

REGISTRY: dict[str, ToolSpec] = {}

# every situation an agent still takes turns in; relieved and dead reach nothing
EVERYWHERE = (
    Situation.IDLE,
    Situation.WAITING_FOR_SEATS,
    Situation.ON_TASK,
    Situation.JURY,
    Situation.CLOSING_TASK,
    Situation.RELEASED,
)


def tool(
    name: str,
    description: str,
    /,
    *,
    situations: tuple[Situation, ...],
    when: Callable[[Agent, Harness], bool] | None = None,
    **args: ArgSpec,
):
    def wrap(handler: Handler) -> Handler:
        REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            args=args,
            handler=handler,
            situations=situations,
            when=when,
        )
        return handler

    return wrap


def tools_for(agent: Agent, harness: Harness) -> list[ToolSpec]:
    """Which tools the agent can reach, given where it currently is."""
    return [
        spec
        for spec in REGISTRY.values()
        if agent.situation in spec.situations and (spec.when is None or spec.when(agent, harness))
    ]


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
        return ActionResult.fail(f"{action.tool}: {'; '.join(problems)}. You sent: {action.args!r}")
    try:
        return await spec.handler(ctx, args)
    except Exception as exc:  # a tool bug must reach the agent, not kill its loop
        return ActionResult.fail(f"{action.tool} failed: {type(exc).__name__}: {exc}")
