"""One agent, or one task, in full."""

from __future__ import annotations

import time
from typing import ClassVar

from rich.console import Group
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.widgets import Footer, Static

from panopticon.tui.base import LiveScreen
from panopticon.tui.format import (
    ACCENT,
    AMBER,
    DIM,
    FAINT,
    GREEN,
    INK,
    SITUATION_LABEL,
    SITUATION_STYLE,
    agent_mark,
    clip,
    describe_wait,
    since,
    tail_entries,
    task_state,
)
from panopticon.tui.render import agent_stats, kv, seats_block, transcript_block

BACK = Binding("escape", "app.pop_screen", "back")


class AgentScreen(LiveScreen):
    BINDINGS: ClassVar[list[BindingType]] = [BACK]

    def __init__(self, name: str) -> None:
        super().__init__()
        self.agent_name = name
        self._entries_seen = -1

    def compose(self) -> ComposeResult:
        yield Static(id="detailbar")
        yield Static(id="agentstats")
        with VerticalScroll(id="transcript"):
            yield Static(id="entries")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#agentstats", Static).border_title = "agent"
        self.query_one("#transcript").border_title = "transcript"
        self.query_one("#transcript").focus()
        self.refresh_data()

    def refresh_data(self) -> None:
        agent = self.harness.agents.get(self.agent_name)
        if agent is None:
            self.query_one("#detailbar", Static).update(
                Text(f"{self.agent_name} is no longer in the harness", style=FAINT)
            )
            return
        now = time.time()
        mark, style = agent_mark(agent)
        bar = Text(f"{mark} {agent.name}", style=f"bold {style}")
        bar.append(f"   {SITUATION_LABEL[agent.situation]}", style=SITUATION_STYLE[agent.situation])
        bar.append(
            f"   {agent.provider}   {agent.turns} turns   {since(agent.born_at, now)}", style=DIM
        )
        self.query_one("#detailbar", Static).update(bar)

        task = self.harness.board.tasks.get(agent.task_id or "")
        submission = self.harness.kb.pending.get(agent.submission_id or "")
        window = self.app.context_window(agent)
        self.query_one("#agentstats", Static).update(
            agent_stats(
                agent,
                window,
                self.harness.model(agent),
                describe_wait(agent, task, submission, now),
                now,
            )
        )

        if len(agent.entries) == self._entries_seen:
            return
        scroller = self.query_one("#transcript", VerticalScroll)
        at_end = scroller.scroll_offset.y >= scroller.max_scroll_y - 1
        entries, hidden = tail_entries(agent.entries)
        self.query_one("#entries", Static).update(transcript_block(entries, hidden))
        self._entries_seen = len(agent.entries)
        if at_end:
            self.call_after_refresh(scroller.scroll_end, animate=False)


class TaskScreen(LiveScreen):
    BINDINGS: ClassVar[list[BindingType]] = [BACK]

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id

    def compose(self) -> ComposeResult:
        yield Static(id="detailbar")
        with VerticalScroll(id="taskbody"):
            yield Static(id="taskmeta")
            yield Static(id="taskseats")
            yield Static(id="taskarchive")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#taskmeta", Static).border_title = "task"
        self.query_one("#taskseats", Static).border_title = "seats"
        self.query_one("#taskarchive", Static).border_title = "archive"
        self.query_one("#taskbody").focus()
        self.refresh_data()

    def refresh_data(self) -> None:
        task = self.harness.board.tasks.get(self.task_id)
        if task is None:
            self.query_one("#detailbar", Static).update(Text("task is gone", style=FAINT))
            return
        now = time.time()
        label, style = task_state(task)
        bar = Text(clip(task.title, 120), style=f"bold {INK}")
        bar.append(f"   {label}", style=style)
        self.query_one("#detailbar", Static).update(bar)

        rows: list[tuple[str, Text]] = [
            ("id", Text(task.id, style=DIM)),
            ("brief", Text(task.description or "-", style=INK)),
            ("created", Text(f"{task.created_by} · {since(task.created_at, now)} ago", style=DIM)),
            (
                "started",
                Text(
                    f"{since(task.started_at, now)} ago" if task.started_at else "not yet",
                    style=DIM if task.started_at else FAINT,
                ),
            ),
            (
                "worktree",
                Text(task.worktree or "none cut yet", style=ACCENT if task.worktree else FAINT),
            ),
        ]
        self.query_one("#taskmeta", Static).update(kv(rows))
        self.query_one("#taskseats", Static).update(seats_block(task, now))

        archive = self.query_one("#taskarchive", Static)
        if not task.archived_at:
            archive.update(Text("open; nothing archived yet", style=FAINT))
            return
        closed = Text(f"closed {since(task.archived_at, now)} ago", style=GREEN)
        closer = Text(f"closer: {task.closer or 'none'}", style=AMBER if task.closer else FAINT)
        archive.update(
            Group(closed, closer, Text(), Text(task.outcome or "no outcome recorded", style=INK))
        )
