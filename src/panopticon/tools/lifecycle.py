"""Leaving, coming back, and closing a task out."""

from __future__ import annotations

from typing import Any

from panopticon.model import ActionResult, ArgSpec, QueueItem, Situation, ToolCtx
from panopticon.prompts import situation_preprompt
from panopticon.situations import MARK_DONE, REJOIN, RELIEVE_SELF, VOTE_GOAL_REACHED
from panopticon.tools import tool


@tool(
    VOTE_GOAL_REACHED,
    "Vote that the goal set for this harness is reached. Your session closes: you keep only the "
    f"message tools and {REJOIN}, and you take no more work until you retract. The whole "
    "session ends when every agent has voted. Vote when the goal is actually met, not when your "
    "own part of it is and not to get out of work you find hard.",
    note=ArgSpec("string", "One line: why you hold that the goal is reached."),
)
async def vote_goal_reached(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    if agent.task_id:
        return ActionResult.fail(
            f"You still hold a seat on task {agent.task_id}. Finish or leave it first."
        )
    note = args["note"].strip()
    agent.voted_goal_reached = True
    agent.situation = Situation.RELEASED
    agent.wake_at = None
    harness.broadcast(
        QueueItem(
            "system",
            f"{agent.name} voted the goal reached and is released from the Panopticon: {note} "
            f"They can still be messaged, and can rejoin.",
        ),
        exclude=(agent.name,),
    )
    voters = [a for a in harness.agents.values() if a.counts_toward_goal]
    counted = sum(1 for a in voters if a.voted_goal_reached)
    harness.tally_goal()
    return ActionResult(
        True,
        f"Your vote is counted ({counted} of {len(voters)}). You are released and out of the "
        f"work until you rejoin.",
    )


@tool(
    REJOIN,
    "Retract your goal-reached vote and come back to the work with a clean context. Do this if "
    "you learn the goal is not reached after all, or if another agent needs you.",
)
async def rejoin(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    agent.voted_goal_reached = False
    agent.wake_at = None
    harness.broadcast(
        QueueItem("system", f"{agent.name} retracted their goal-reached vote and rejoined."),
        exclude=(agent.name,),
    )
    harness.enter(
        agent,
        Situation.IDLE,
        situation_preprompt(
            agent,
            harness,
            "You had voted the goal reached and were released. You retracted that vote and are "
            "back in the work.",
            Situation.IDLE,
        ),
    )
    return ActionResult(True, "You are back, and your vote is retracted.")


@tool(
    RELIEVE_SELF,
    "Leave the Panopticon without voting that the goal is reached. Available only because the "
    "human has ended the session. Your loop stops for good: no rejoining, no messages. Use it "
    "once your work is wrapped up and handed over.",
)
async def relieve_self(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    if not harness.force_ending:
        return ActionResult.fail(
            "The session has not been ended by the human, so you cannot relieve yourself."
        )
    agent.situation = Situation.RELIEVED
    agent.wake_at = None
    harness.broadcast(
        QueueItem("system", f"{agent.name} relieved themselves and has left the Panopticon."),
        exclude=(agent.name,),
    )
    harness.retire(agent)
    return ActionResult(True, "You are relieved. This is your last turn.")


@tool(
    MARK_DONE,
    "Call this when the task you are closing out is fully integrated: its git worktree is dealt "
    "with, and everyone who needed to know has been told. It archives the task and ends your "
    "run. Do not call it with anything left open.",
    summary=ArgSpec(
        "string",
        "One or two lines for the archive: what happened to the work and to the worktree.",
    ),
)
async def mark_integration_done(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    task = harness.board.get(agent.task_id or "")
    if task is None:
        return ActionResult.fail("You are not closing out a task.")
    summary = args["summary"].strip()
    harness.board.archive_task(task, summary)
    agent.task_id = None
    agent.situation = Situation.RELIEVED
    harness.broadcast(
        QueueItem("board", f"Task {task.id} ({task.title}) is closed out and archived: {summary}"),
        exclude=(agent.name,),
    )
    harness.retire(agent)
    return ActionResult(True, f"Task {task.id} is archived. Your run ends here.")
