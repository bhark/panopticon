"""Rich renderables shared by the screens. Everything here is read-only on harness state."""

from __future__ import annotations

import time

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from panopticon.model import Agent, Entry, Event, Shout, Submission, Task, Truth
from panopticon.tui.format import (
    ACCENT,
    AMBER,
    BLUE,
    DIM,
    ENTRY_STYLE,
    FAINT,
    GREEN,
    INK,
    RED,
    SITUATION_LABEL,
    SITUATION_STYLE,
    VERDICT_STYLE,
    VIOLET,
    agent_mark,
    clip,
    clock,
    context_cell,
    oneline,
    since,
    tally,
    task_state,
)

AGENT_COLUMNS = (
    ("", 1),
    ("agent", 12),
    ("provider", 14),
    ("situation", 17),
    ("turns", 5),
    ("context", 14),
    ("last action", 26),
    ("up", 7),
)

TASK_COLUMNS = (
    ("task", 34),
    ("seats", 6),
    ("state", 16),
    ("holders", 28),
    ("age", 7),
)


def kv(rows: list[tuple[str, RenderableType]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=DIM, justify="right", width=12)
    table.add_column(style=INK)
    for label, value in rows:
        table.add_row(label, value)
    return table


def agent_row(agent: Agent, window: int, now: float) -> list[Text]:
    mark, style = agent_mark(agent)
    ctx_text, ctx_style = context_cell(agent.usage.context_tokens, window)
    faded = FAINT if not agent.alive else INK
    return [
        Text(mark, style=style),
        Text(agent.name, style=f"bold {style}" if agent.alive else style),
        Text(clip(agent.provider, 14), style=DIM),
        Text(SITUATION_LABEL[agent.situation], style=SITUATION_STYLE[agent.situation]),
        Text(str(agent.turns), style=DIM, justify="right"),
        Text(ctx_text, style=ctx_style),
        Text(oneline(agent.last_action or "-", 26), style=faded),
        Text(since(agent.born_at, now), style=DIM, justify="right"),
    ]


def task_row(task: Task, now: float) -> list[Text]:
    label, style = task_state(task)
    filled = sum(1 for s in task.seats if s.holder)
    holders = ", ".join(task.holders) or "-"
    return [
        Text(clip(task.title, 34), style=INK if not task.archived_at else FAINT),
        Text(f"{filled}/{len(task.seats)}", style=DIM, justify="right"),
        Text(label, style=style),
        Text(clip(holders, 28), style=DIM),
        Text(since(task.created_at, now), style=DIM, justify="right"),
    ]


def shout_block(shouts: list[Shout], limit: int) -> RenderableType:
    if not shouts:
        return Text("nothing shouted yet", style=FAINT)
    lines: list[RenderableType] = []
    for shout in shouts[-limit:]:
        head = Text(f"{clock(shout.at)} ", style=FAINT)
        head.append(shout.sender, style=f"bold {ACCENT if shout.sender == 'human' else BLUE}")
        lines.append(head)
        lines.append(Text(f"  {oneline(shout.body, 400)}", style=INK))
    return Group(*lines)


def event_block(events: list[Event], limit: int) -> RenderableType:
    if not events:
        return Text("quiet", style=FAINT)
    lines = []
    for event in list(events)[-limit:]:
        line = Text(f"{clock(event.at)} ", style=FAINT)
        if event.agent:
            line.append(f"{event.agent} ", style=ACCENT)
        line.append(oneline(event.text, 200), style=INK)
        lines.append(line)
    return Group(*lines)


def transcript_block(entries: list[Entry], hidden: int) -> RenderableType:
    lines: list[RenderableType] = []
    if hidden:
        lines.append(Text(f"… {hidden} earlier entries not shown", style=FAINT))
        lines.append(Text())
    for entry in entries:
        style = ENTRY_STYLE.get(entry.kind, INK)
        head = Text(f"t{entry.turn:<4}", style=FAINT)
        head.append(f"{entry.kind:<8}", style=style)
        head.append(clock(entry.at), style=FAINT)
        lines.append(head)
        lines.append(Text(clip(entry.text, 3000), style=INK))
        lines.append(Text())
    if not lines:
        lines.append(Text("no transcript yet", style=FAINT))
    return Group(*lines)


def seats_block(task: Task, now: float) -> RenderableType:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=DIM, width=18)
    table.add_column(width=12)
    table.add_column(style=DIM, width=8)
    table.add_column(style=INK)
    table.add_row(
        Text("role", style=FAINT),
        Text("holder", style=FAINT),
        Text("held", style=FAINT),
        Text("finalization", style=FAINT),
    )
    for seat in task.seats:
        held = since(seat.assigned_at, now) if seat.assigned_at else "-"
        if seat.finalization:
            final = Text(f"{seat.finalization.reason}", style=GREEN)
            final.append(f"\n{seat.finalization.conclusion}", style=DIM)
        else:
            final = Text("not finalized", style=FAINT)
        table.add_row(
            Text(clip(seat.role, 18), style=INK),
            Text(seat.holder or "empty", style=ACCENT if seat.holder else FAINT),
            Text(held),
            final,
        )
    return table


def truths_block(truths: list[Truth], now: float) -> RenderableType:
    if not truths:
        return Text("nothing accepted yet", style=FAINT)
    lines: list[RenderableType] = []
    for truth in truths:
        head = Text("+ ", style=GREEN)
        head.append(truth.title, style=f"bold {INK}")
        lines.append(head)
        lines.append(Text(f"  {truth.body}", style=DIM))
        lines.append(
            Text(f"  {truth.submitted_by} · {since(truth.accepted_at, now)} ago", style=FAINT)
        )
        lines.append(Text())
    return Group(*lines)


def pending_block(pending: list[Submission], now: float) -> RenderableType:
    if not pending:
        return Text("no submissions waiting", style=FAINT)
    lines: list[RenderableType] = []
    for sub in pending:
        head = Text("? ", style=AMBER)
        head.append(sub.title, style=f"bold {AMBER}")
        lines.append(head)
        lines.append(Text(f"  {sub.body}", style=DIM))
        meta = Text(f"  {sub.submitted_by} · {since(sub.submitted_at, now)} ago · ", style=FAINT)
        meta.append(tally(sub), style=INK)
        if sub.restated_from:
            meta.append("  restated", style=VIOLET)
        lines.append(meta)
        jurors = ", ".join(sub.jurors) or "none seated"
        lines.append(Text(f"  jurors: {jurors}", style=VIOLET if sub.jurors else FAINT))
        for verdict in sub.verdicts:
            line = Text(f"  {verdict.juror} ", style=DIM)
            line.append(str(verdict.call), style=VERDICT_STYLE[str(verdict.call)])
            line.append(f"  {oneline(verdict.reasoning, 160)}", style=DIM)
            lines.append(line)
        lines.append(Text())
    return Group(*lines)


def goal_bar(goal: str, started_at: float, agents: list[Agent], flag: str) -> Text:
    now = time.time()
    live = sum(1 for a in agents if a.alive)
    released = sum(1 for a in agents if a.voted_goal_reached)
    line = Text("PANOPTICON", style=f"bold {ACCENT}")
    line.append("  ")
    line.append(oneline(goal, 160), style=f"bold {INK}")
    line.append(f"   {since(started_at, now)}", style=DIM)
    line.append(f"   {live}/{len(agents)} live", style=DIM)
    if released:
        line.append(f"   {released} voted goal reached", style=ACCENT)
    if flag:
        line.append(f"   {flag}", style=f"bold {RED}")
    return line
