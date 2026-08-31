"""The shared task board: seats, assignment, finalization."""

from __future__ import annotations

from typing import Any

from panopticon.model import ActionResult, ArgSpec, QueueItem, Situation, ToolCtx
from panopticon.situations import (
    ASSIGN_SELF,
    CANCEL_FINALIZE,
    CREATE_TASK,
    FINALIZE_TASK,
    UNASSIGN_SELF,
    VIEW_BOARD,
)
from panopticon.tools import tool


@tool(
    VIEW_BOARD,
    "Show the task board: open tasks with their seats, the role of each seat and who holds it, "
    "plus the most recent archived tasks. Read it before you create a task, so you do not "
    "duplicate one that already exists.",
)
async def view_task_board(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    return ActionResult(True, ctx.harness.board.render())


@tool(
    CREATE_TASK,
    "Put a task on the board for anyone to take. One role per seat: the number of roles is the "
    "number of agents the task needs, and it cannot start until every seat is filled. Ask for "
    "the fewest seats that can actually do the work - one is usually right, and every extra "
    "seat is an agent the task has to wait for. Creating a task does not put you on it.",
    title=ArgSpec("string", "One line naming the work."),
    description=ArgSpec(
        "string",
        "A few lines at most: what has to be true for this to be done. Facts and constraints, "
        "no preamble.",
    ),
    roles=ArgSpec(
        "string[]",
        'One short role per seat, e.g. ["implementer"] or ["implementer", "reviewer"].',
    ),
)
async def create_task(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    roles = [r.strip() for r in args["roles"] if r.strip()]
    if not roles:
        return ActionResult.fail("roles must hold at least one role; one role is one seat.")
    title = args["title"].strip()
    if not title:
        return ActionResult.fail("title must not be empty.")
    task = ctx.harness.board.create(ctx.agent.name, title, args["description"].strip(), roles)
    ctx.harness.broadcast(
        QueueItem(
            "board",
            f"{ctx.agent.name} put task {task.id} on the board: {task.title} "
            f"({len(roles)} seats: {', '.join(roles)}).",
        ),
        exclude=(ctx.agent.name,),
    )
    return ActionResult(
        True,
        f"Task {task.id} created with seats: {', '.join(roles)}. You are not on it; take a "
        f"seat with {ASSIGN_SELF} if you mean to do the work.",
    )


@tool(
    ASSIGN_SELF,
    "Take an open seat on a task, by the role of that seat. If yours fills the last seat the "
    "task starts at once: everyone on it gets a fresh context and a git worktree of their own. "
    "If seats are still open you wait, and you can keep acting while you do.",
    task_id=ArgSpec("string", "The task id as it appears on the board."),
    role=ArgSpec("string", "The role of the seat you want, exactly as it appears on the board."),
)
async def assign_self(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    if agent.task_id:
        return ActionResult.fail(
            f"You already hold a seat on task {agent.task_id}. Leave it first with {UNASSIGN_SELF}."
        )
    role = args["role"].strip()
    try:
        task, ready = harness.board.assign(agent.name, args["task_id"].strip(), role)
    except ValueError as exc:
        return ActionResult.fail(str(exc))

    agent.task_id = task.id
    for name in task.holders:
        if name != agent.name:
            harness.post(
                name,
                QueueItem(
                    "task",
                    f"{agent.name} took the {role} seat on task {task.id} alongside you.",
                ),
            )
    if not ready:
        agent.situation = Situation.WAITING_FOR_SEATS
        open_roles = [s.role for s in task.seats if not s.holder]
        return ActionResult(
            True,
            f"You hold the {role} seat on task {task.id}. It starts once these seats fill: "
            f"{', '.join(open_roles)}. You will be told when someone joins.",
        )

    await harness.launch_task(task)
    return ActionResult(True, f"Task {task.id} is fully seated and has started.")


@tool(
    UNASSIGN_SELF,
    "Give up your seat. If the task was still waiting for seats it goes back to open. If it was "
    "already running it stops for everyone on it until your seat is filled again, so tell them "
    "why before you do this.",
    task_id=ArgSpec("string", "The task you are leaving."),
)
async def unassign_self(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    task_id = args["task_id"].strip()
    task = harness.board.get(task_id)
    if task is None:
        return ActionResult.fail(f"There is no task {task_id}.")
    if task.seat_of(agent.name) is None:
        return ActionResult.fail(f"You hold no seat on task {task_id}.")
    harness.leave_task(agent.name, task, "unassigned themselves")
    return ActionResult(True, f"You are off task {task.id}.")


@tool(
    FINALIZE_TASK,
    "Declare the task finished from your seat. It only ends when every seat has finalized; the "
    "others are told that you did and can follow or not. Until they all do you can take it "
    f"back with {CANCEL_FINALIZE}. Both fields are read by the agent that closes the task out, "
    "so make them short and factual.",
    task_id=ArgSpec("string", "The task you are finalizing."),
    reason=ArgSpec("string", "One line: why this is done."),
    conclusion=ArgSpec("string", "One line: what is now true that was not before."),
)
async def finalize_task(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    try:
        task, agreed = harness.board.finalize(
            agent.name, args["task_id"].strip(), args["reason"].strip(), args["conclusion"].strip()
        )
    except ValueError as exc:
        return ActionResult.fail(str(exc))

    if not agreed:
        waiting = [s.role for s in task.seats if not s.finalization]
        for name in task.holders:
            if name != agent.name:
                harness.post(
                    name,
                    QueueItem(
                        "task",
                        f"{agent.name} finalized their seat on task {task.id}: "
                        f"{args['conclusion'].strip()}. Still to finalize: {', '.join(waiting)}.",
                    ),
                )
        return ActionResult(
            True,
            f"Finalized from your seat. The task ends when these seats do too: "
            f"{', '.join(waiting)}.",
        )

    await harness.close_task(task)
    return ActionResult(True, f"Every seat agreed. Task {task.id} is finished.")


@tool(
    CANCEL_FINALIZE,
    "Withdraw your finalization, because the task is not done after all. Say why to the others; "
    "they finalized or are about to, and they cannot see your reasoning.",
    task_id=ArgSpec("string", "The task you are un-finalizing."),
)
async def cancel_finalize(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    try:
        task = harness.board.cancel_finalize(agent.name, args["task_id"].strip())
    except ValueError as exc:
        return ActionResult.fail(str(exc))
    for name in task.holders:
        if name != agent.name:
            harness.post(
                name,
                QueueItem(
                    "task",
                    f"{agent.name} withdrew their finalization on task {task.id}. It is not "
                    f"ending yet.",
                ),
            )
    return ActionResult(True, f"Your finalization on task {task.id} is withdrawn.")
