"""Reading the knowledge base and putting a truth up for jury."""

from __future__ import annotations

from typing import Any

from panopticon.model import ActionResult, ArgSpec, QueueItem, Situation, ToolCtx
from panopticon.tools.registry import tool
from panopticon.transcript import dedupe_result

VIEW_KB = "view_knowledge_base"
SUBMIT_TRUTH = "submit_truth"

_ASSERTERS = (
    Situation.IDLE,
    Situation.WAITING_FOR_SEATS,
    Situation.ON_TASK,
    Situation.CLOSING_TASK,
)
# a truth needs the knowledge base read first, so nobody duplicates or contradicts one
_READERS = (*_ASSERTERS, Situation.JURY)


@tool(
    VIEW_KB,
    "Read everything the harness has established so far: every accepted truth, title and body. "
    f"Read it before you assume anything. It also unlocks {SUBMIT_TRUTH}, because you cannot "
    "judge whether your own finding is new or contradicts what is already known until you have "
    "seen what is there.",
    situations=_READERS,
)
async def view_knowledge_base(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    ctx.agent.seen_kb = True
    return ActionResult(True, dedupe_result(ctx.agent, VIEW_KB, ctx.harness.kb.render()))


@tool(
    SUBMIT_TRUTH,
    "Put one verified fact to a jury of your peers. Two other agents must call it true before "
    "it enters the knowledge base, and a single agent calling it false kills it outright. So "
    "assert exactly one thing: a submission that bundles several claims almost always fails, "
    "because a juror only has to disbelieve one part of it to reject the whole. The truth "
    "itself belongs in the title, in the most compact practical form you can write it - a "
    "claim, not a topic. The body is what another agent needs in order to use it, plus your "
    "proof. Written for agents, not humans: no prose, no hedging, nothing you have not checked.",
    situations=_ASSERTERS,
    when=lambda agent, harness: agent.seen_kb,
    title=ArgSpec(
        "string",
        "The truth itself, one short line, exact and practical. Not 'notes on the build' but "
        "'the test suite needs PANOPTICON_HOME set or it writes to the real home dir'.",
    ),
    body=ArgSpec(
        "string",
        "Tight expansion and proof: the path, the command, the output that shows it. A few lines.",
    ),
)
async def submit_truth(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    title = args["title"].strip()
    body = args["body"].strip()
    if not title or not body:
        return ActionResult.fail("Both title and body are needed; the body carries your proof.")

    submission = harness.kb.submit(agent.name, title, body)
    harness.broadcast(
        QueueItem(
            "jury",
            f"{agent.name} submitted a truth for jury ({submission.id}): {submission.title}. "
            f"{len(harness.kb.pending)} waiting for jury.",
        ),
        exclude=(agent.name,),
    )
    return ActionResult(
        True,
        f"Submission {submission.id} is with the jury. It needs two 'true' verdicts to enter "
        f"the knowledge base. You cannot judge it yourself.",
    )
