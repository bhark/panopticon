"""Direct messages and the shoutboard."""

from __future__ import annotations

from typing import Any

from panopticon.model import ActionResult, ArgSpec, QueueItem, ToolCtx
from panopticon.tools.registry import EVERYWHERE, tool

SEND_DM = "send_direct_message"
SHOUT = "send_shoutboard_message"
VIEW_SHOUTBOARD = "view_shoutboard"


@tool(
    SEND_DM,
    "Send one agent a message by name. It lands in their queue and they read it on their next "
    "turn, so do not expect an answer this turn. Write only what they need in order to act: no "
    "greeting, no sign-off, no repeating what they already know. One or two lines.",
    situations=EVERYWHERE,
    to=ArgSpec("string", "The name of the agent you are writing to."),
    body=ArgSpec("string", "The message."),
)
async def send_direct_message(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    wanted = args["to"].strip()
    recipient = harness.agents.get(wanted) or next(
        (a for a in harness.agents.values() if a.name.lower() == wanted.lower()), None
    )
    if recipient is None:
        others = sorted(a.name for a in harness.agents.values() if a.name != agent.name)
        return ActionResult.fail(
            f"There is no agent called {wanted!r}. Here: {', '.join(others) or 'nobody else'}."
        )
    if recipient.name == agent.name:
        return ActionResult.fail("You cannot message yourself.")
    if not recipient.alive:
        return ActionResult.fail(f"{recipient.name} has left the Panopticon and cannot be reached.")

    body = args["body"].strip()
    if not body:
        return ActionResult.fail("body must not be empty.")
    harness.post(recipient.name, QueueItem("dm", f"{agent.name} wrote to you: {body}"))
    return ActionResult(True, f"Sent to {recipient.name}.")


@tool(
    SHOUT,
    "Write to the shoutboard, which every agent can read. Use it for something the whole "
    "harness needs - a finding, a warning, a claim on an area of work - not for chatter or "
    "status. Everyone is notified once the board goes quiet, not once per message, so a burst "
    "of shouts costs everyone one interruption. Keep it to a line or two.",
    situations=EVERYWHERE,
    body=ArgSpec("string", "What you are telling everyone."),
)
async def send_shoutboard_message(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    body = args["body"].strip()
    if not body:
        return ActionResult.fail("body must not be empty.")
    ctx.harness.bus.shout(ctx.agent.name, body)
    return ActionResult(True, "On the shoutboard. Others are notified once the board settles.")


@tool(
    VIEW_SHOUTBOARD,
    "Read the shoutboard: who wrote what, newest first. Old messages fall off it, so read it "
    "when you are told there is something new rather than saving it up.",
    situations=EVERYWHERE,
)
async def view_shoutboard(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    return ActionResult(True, ctx.harness.bus.render_shoutboard())
