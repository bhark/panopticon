"""Doing nothing, on purpose."""

from __future__ import annotations

import math
import time
from typing import Any

from panopticon.model import ActionResult, ArgSpec, Situation, ToolCtx
from panopticon.situations import WAIT
from panopticon.tools import tool

CAP_MINUTES = 5
# situations where the agent has a job in front of it and nothing is bound to ever wake it
_HOLDING_WORK = (Situation.ON_TASK, Situation.CLOSING_TASK, Situation.JURY)


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
        if ctx.agent.situation in _HOLDING_WORK:
            ctx.agent.wake_at = time.time() + CAP_MINUTES * 60
            return ActionResult(
                True,
                f"Waiting up to {CAP_MINUTES}m. You are holding work that does not move while "
                "you wait, and nothing else will finish it, so get on with it or hand it back.",
            )
        # inf, not None: None means "not waiting", and the loop would take another turn at once
        ctx.agent.wake_at = math.inf
        return ActionResult(True, "Waiting. Anything arriving for you wakes you.")
    if minutes < 1:
        return ActionResult.fail("minutes must be 1 or more. Omit it to wait indefinitely.")
    ctx.agent.wake_at = time.time() + minutes * 60
    return ActionResult(True, f"Waiting {minutes}m, or until something arrives for you.")
