"""The shared task board: seats, assignment, finalization."""

from __future__ import annotations

from typing import Any

from panopticon.model import (
    ActionResult,
    Agent,
    ArgSpec,
    Harness,
    QueueItem,
    Situation,
    Task,
    ToolCtx,
)
from panopticon.tools.registry import tool
from panopticon.transcript import dedupe_result

VIEW_BOARD = "view_task_board"
CREATE_TASK = "create_task"
ASSIGN_SELF = "assign_self"
UNASSIGN_SELF = "unassign_self"
FINALIZE_TASK = "finalize_task"
CANCEL_FINALIZE = "cancel_finalize"

_SEATED = (Situation.WAITING_FOR_SEATS, Situation.ON_TASK)
_BOARD_READERS = (*_SEATED, Situation.IDLE, Situation.CLOSING_TASK)
# a seat can be taken from idle, or given up for a better one while still waiting
_SEAT_TAKERS = (Situation.IDLE, Situation.WAITING_FOR_SEATS)


def _seatable(harness: Harness) -> int:
    """Agents that could ever fill a seat. A released one counts; it can rejoin."""
    return sum(1 for a in harness.agents.values() if a.alive and not a.transient)


def _has_finalized(agent: Agent, harness: Harness) -> bool:
    task = harness.board.get(agent.task_id or "")
    seat = task.seat_of(agent.name) if task else None
    return bool(seat and seat.finalization)


def _roster(harness: Harness) -> str:
    """Who else is here and what they are. The only place an agent learns the head count."""
    live = sorted((a for a in harness.agents.values() if a.alive), key=lambda a: a.name)
    return f"agents ({len(live)})\n" + "\n".join(
        f"  {a.name} [{a.level}] {a.situation}" for a in live
    )


def _tell_mates(ctx: ToolCtx, task: Task, text: str) -> None:
    for name in task.holders:
        if name != ctx.agent.name:
            ctx.harness.post(name, QueueItem("task", text))


@tool(
    VIEW_BOARD,
    "Show the task board: open tasks with their seats, the role of each seat and who holds it, "
    "the most recent archived tasks, and every agent here with the level it runs at. Read it "
    "before you create a task, so you do not duplicate one that already exists, and before you "
    "take a seat, so you know who else could take it.",
    situations=_BOARD_READERS,
)
async def view_task_board(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    board = f"{ctx.harness.board.render()}\n\n{_roster(ctx.harness)}"
    return ActionResult(True, dedupe_result(ctx.agent, VIEW_BOARD, board))


@tool(
    CREATE_TASK,
    "Put a task on the board for anyone to take. One role per seat: the number of roles is the "
    "number of agents the task needs, and it cannot start until every seat is filled. Ask for "
    "the fewest seats that can actually do the work - one is usually right, and every extra "
    "seat is an agent the task has to wait for. Creating a task does not put you on it. Look at "
    "the board first: if the work is already there, take a seat on it instead.",
    situations=(Situation.IDLE,),
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
    seatable = _seatable(ctx.harness)
    if len(roles) > seatable:
        return ActionResult.fail(
            f"You asked for {len(roles)} seats and there are only {seatable} agents here, so "
            f"the task could never start. Ask for {seatable} seats or fewer."
        )
    title = args["title"].strip()
    if not title:
        return ActionResult.fail("title must not be empty.")
    if twin := ctx.harness.board.duplicate_of(title):
        return ActionResult.fail(
            f"{twin.id} is already on the board saying the same thing: {twin.title}. "
            f"Take a seat on it instead, or say something different."
        )
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
    "If seats are still open you wait, and you can keep acting while you do. While you are "
    "waiting you can call this again for a seat on another task: you give up the seat you were "
    "waiting on and take the new one, which is how you break a standoff where everyone is "
    "waiting and nobody is left to fill a seat.",
    situations=_SEAT_TAKERS,
    task_id=ArgSpec("string", "The task id as it appears on the board."),
    role=ArgSpec("string", "The role of the seat you want, exactly as it appears on the board."),
)
async def assign_self(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    role = args["role"].strip()
    old = harness.board.get(agent.task_id or "")
    try:
        task, ready = harness.board.assign(agent.name, args["task_id"].strip(), role)
    except ValueError as exc:
        return ActionResult.fail(str(exc))

    if old is not None:
        # idle in effect, between seats: leave_task must not clear the transcript for a swap
        agent.situation = Situation.IDLE
        harness.leave_task(agent.name, old, f"moved to the {role} seat on {task.id}", notify=False)
    agent.task_id = task.id
    _tell_mates(ctx, task, f"{agent.name} took the {role} seat on task {task.id} alongside you.")
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
    situations=_SEATED,
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
    situations=(Situation.ON_TASK,),
    when=lambda agent, harness: not _has_finalized(agent, harness),
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
        _tell_mates(
            ctx,
            task,
            f"{agent.name} finalized their seat on task {task.id}: "
            f"{args['conclusion'].strip()}. Still to finalize: {', '.join(waiting)}.",
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
    situations=(Situation.ON_TASK,),
    when=_has_finalized,
    task_id=ArgSpec("string", "The task you are un-finalizing."),
)
async def cancel_finalize(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    harness, agent = ctx.harness, ctx.agent
    try:
        task = harness.board.cancel_finalize(agent.name, args["task_id"].strip())
    except ValueError as exc:
        return ActionResult.fail(str(exc))
    _tell_mates(
        ctx,
        task,
        f"{agent.name} withdrew their finalization on task {task.id}. It is not ending yet.",
    )
    return ActionResult(True, f"Your finalization on task {task.id} is withdrawn.")
