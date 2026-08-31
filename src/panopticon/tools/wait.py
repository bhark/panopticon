"""Doing nothing, on purpose."""

from __future__ import annotations

import math
import time
from typing import Any

from panopticon.model import ActionResult, ArgSpec, Situation, ToolCtx
from panopticon.situations import WAIT
from panopticon.tools import tool

ON_TASK_CAP_MINUTES = 5


@tool(
    WAIT,
    "Do nothing this turn. Use it when there is nothing useful for you to do right now and "
    "you would rather let others act than invent work. Anything arriving for you - a message, "
    "a task board event, jury work - wakes you.",
    minutes=ArgSpec(
        "int",
        "Wake again after this many minutes even if nothing arrives. Omit to sleep until "
        "something does.",
        required=False,
    ),
)
async def wait(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    minutes = args.get("minutes")
    if minutes is None:
        if ctx.agent.situation is Situation.ON_TASK:
            # a task does not move while its holder sleeps, and nothing is bound to wake them
            ctx.agent.wake_at = time.time() + ON_TASK_CAP_MINUTES * 60
            return ActionResult(
                True,
                f"Waiting up to {ON_TASK_CAP_MINUTES}m. You are holding a seat on a task and it "
                "does not move while you wait, so do the work or leave the seat.",
            )
        # inf, not None: None means "not waiting", and the loop would take another turn at once
        ctx.agent.wake_at = math.inf
        return ActionResult(True, "Waiting. Anything arriving for you wakes you.")
    if minutes < 1:
        return ActionResult.fail("minutes must be 1 or more. Omit it to wait indefinitely.")
    ctx.agent.wake_at = time.time() + minutes * 60
    return ActionResult(True, f"Waiting {minutes}m, or until something arrives for you.")
