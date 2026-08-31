"""Jury duty: judging a submitted truth."""

from __future__ import annotations

from typing import Any

from panopticon.model import (
    ActionResult,
    ArgSpec,
    QueueItem,
    Situation,
    Submission,
    ToolCtx,
    VerdictCall,
)
from panopticon.prompts import situation_preprompt
from panopticon.situations import CANCEL_JURY, JOIN_JURY, LIST_JURY, SUBMIT_VERDICT
from panopticon.tools import tool

_CALLS = ", ".join(c.value for c in VerdictCall)


@tool(
    LIST_JURY,
    "Show the truths waiting for a jury, with their submitters. Your own submissions are in "
    "there too, and you may not judge those.",
)
async def list_jury_submissions(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    return ActionResult(True, ctx.harness.kb.render_pending())


@tool(
    JOIN_JURY,
    "Take jury duty on one submission. Your context is cleared and you judge that one statement "
    "and nothing else. You cannot judge your own submission. While on the jury your tools are "
    "read-only: read files and run shell commands to gather evidence, and only that. Do not run "
    "tests, do not build, do not change anything. This is a reasoning job against what is "
    "already there.",
    submission_id=ArgSpec("string", "The id of the submission you want to judge."),
)
async def join_jury(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    try:
        submission = harness.kb.join(agent.name, args["submission_id"].strip())
    except ValueError as exc:
        return ActionResult.fail(str(exc))
    agent.submission_id = submission.id
    harness.enter(
        agent,
        Situation.JURY,
        situation_preprompt(agent, harness, "", Situation.JURY),
    )
    harness.post(
        submission.submitted_by,
        QueueItem("jury", f"{agent.name} joined the jury on your submission {submission.id}."),
    )
    return ActionResult(True, f"You are on jury duty for {submission.id}.")


@tool(
    SUBMIT_VERDICT,
    "Decide, once. 'true' if the statement holds as written; two of these accept it. 'false' if "
    "any part of it does not hold; one of these discards it immediately, so use it when the "
    "statement is wrong, not when it is merely awkward. 'restate' when the finding is real but "
    "the statement is not the right one: you write the replacement yourself, as one standalone "
    "claim, and it goes back to the jury from scratch. Your verdict ends your jury duty.",
    verdict=ArgSpec("string", f"One of: {_CALLS}."),
    reasoning=ArgSpec("string", "Short. What decided it, and the evidence if you gathered any."),
    restated_title=ArgSpec(
        "string",
        "Only with 'restate': the replacement claim, one short line, written to stand alone.",
        required=False,
    ),
    restated_body=ArgSpec(
        "string", "Only with 'restate': the replacement body and its proof.", required=False
    ),
)
async def submit_verdict(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    if not agent.submission_id:
        return ActionResult.fail("You are not on jury duty.")
    raw = args["verdict"].strip().lower()
    try:
        call = VerdictCall(raw)
    except ValueError:
        return ActionResult.fail(f"verdict must be one of: {_CALLS}. You sent {raw!r}.")

    restated_title = (args.get("restated_title") or "").strip()
    restated_body = (args.get("restated_body") or "").strip()
    if call is VerdictCall.RESTATE and not (restated_title and restated_body):
        return ActionResult.fail(
            "A 'restate' verdict has to carry the replacement: restated_title and restated_body "
            "are both needed, written as one standalone claim."
        )

    judged = agent.submission_id
    submission, outcome = harness.kb.verdict(
        agent.name, judged, call, args["reasoning"].strip(), restated_title, restated_body
    )
    _announce(ctx, submission, outcome, call)
    # only a resolving verdict strands the others; a pending one leaves them judging
    if outcome != "pending":
        harness.resettle_jury(judged, outcome, submission, exclude=(agent.name,))

    agent.submission_id = None
    harness.enter(
        agent,
        Situation.IDLE,
        situation_preprompt(
            agent,
            harness,
            f"You judged submission {judged} '{call.value}' and your jury duty ended ({outcome}).",
            Situation.IDLE,
        ),
    )
    return ActionResult(True, f"Verdict recorded on {judged}: {outcome}.")


def _announce(ctx: ToolCtx, submission: Submission, outcome: str, call: VerdictCall) -> None:
    harness, agent = ctx.harness, ctx.agent
    if outcome == "accepted":
        harness.broadcast(
            QueueItem(
                "jury",
                f"A truth entered the knowledge base: {submission.title} (submitted by "
                f"{submission.submitted_by}).",
            )
        )
        return
    if outcome == "restated":
        harness.broadcast(
            QueueItem(
                "jury",
                f"{agent.name} restated a submission; it is back with the jury as "
                f"{submission.id}: {submission.title}.",
            ),
            exclude=(agent.name,),
        )
        return
    text = (
        f"Your submission was rejected by {agent.name}: {submission.title}."
        if outcome == "rejected"
        else f"{agent.name} judged your submission '{call.value}'. It is still with the jury."
    )
    harness.post(submission.submitted_by, QueueItem("jury", text))


@tool(
    CANCEL_JURY,
    "Step off the jury without a verdict, leaving the submission for someone else. Use it when "
    "you cannot settle the question with what you can see, not when it is merely hard.",
)
async def cancel_jury(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    if not agent.submission_id:
        return ActionResult.fail("You are not on jury duty.")
    left = agent.submission_id
    try:
        harness.kb.leave(agent.name, left)
    except ValueError as exc:
        return ActionResult.fail(str(exc))
    agent.submission_id = None
    harness.enter(
        agent,
        Situation.IDLE,
        situation_preprompt(
            agent, harness, f"You left jury duty on {left} without a verdict.", Situation.IDLE
        ),
    )
    return ActionResult(True, f"You are off the jury for {left}.")
