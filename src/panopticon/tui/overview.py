"""Looking across the panopticon: every agent, the board, the jury, the shouting."""

from __future__ import annotations

import time

from rich.console import Group
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Static

from panopticon.model import Agent, Situation
from panopticon.tui.base import LiveScreen, repaint_table
from panopticon.tui.detail import AgentScreen, TaskScreen
from panopticon.tui.format import ACCENT, AMBER, DIM, FAINT, GREEN, INK, RED, since
from panopticon.tui.render import (
    AGENT_COLUMNS,
    TASK_COLUMNS,
    agent_row,
    event_block,
    goal_bar,
    kv,
    shout_block,
    task_row,
)

ARCHIVE_SHOWN = 8
SHOUT_TAIL = 12
EVENT_TAIL = 40


class OverviewScreen(LiveScreen):
    def compose(self) -> ComposeResult:
        yield Static(id="goalbar")
        yield DataTable(id="agents", cursor_type="row", zebra_stripes=True)
        with Horizontal(id="lower"):
            yield DataTable(id="tasks", cursor_type="row")
            with Vertical(id="right"):
                yield Static(id="state")
                with VerticalScroll(id="shoutpanel"):
                    yield Static(id="shouts")
                with VerticalScroll(id="activitypanel"):
                    yield Static(id="activity")
        yield Footer()

    def on_mount(self) -> None:
        agents = self.query_one("#agents", DataTable)
        agents.border_title = "agents"
        for label, width in AGENT_COLUMNS:
            agents.add_column(Text(label, style=FAINT), width=width)
        tasks = self.query_one("#tasks", DataTable)
        tasks.border_title = "task board"
        for label, width in TASK_COLUMNS:
            tasks.add_column(Text(label, style=FAINT), width=width)
        self.query_one("#state", Static).border_title = "state"
        self.query_one("#shoutpanel").border_title = "shoutboard"
        self.query_one("#activitypanel").border_title = "activity"
        agents.focus()
        self.refresh_data()

    def refresh_data(self) -> None:
        now = time.time()
        app = self.app
        harness = self.harness
        agents = sorted(harness.agents.values(), key=lambda a: (not a.alive, a.name))

        self.query_one("#goalbar", Static).update(
            goal_bar(harness.goal, app.started_at, agents, app.banner())
        )
        repaint_table(
            self.query_one("#agents", DataTable),
            [(a.name, agent_row(a, app.context_window(a.provider), now)) for a in agents],
        )
        repaint_table(self.query_one("#tasks", DataTable), self._task_rows(now))
        self.query_one("#state", Static).update(self._state(agents, now))
        self.query_one("#shouts", Static).update(shout_block(harness.bus.shouts, SHOUT_TAIL))
        self.query_one("#activity", Static).update(event_block(app.events, EVENT_TAIL))

    def _task_rows(self, now: float) -> list[tuple[str, list[Text]]]:
        tasks = list(self.harness.board.tasks.values())
        live = sorted((t for t in tasks if not t.archived_at), key=lambda t: t.created_at)
        closed = sorted(
            (t for t in tasks if t.archived_at), key=lambda t: t.archived_at or 0, reverse=True
        )
        return [(t.id, task_row(t, now)) for t in [*live, *closed[:ARCHIVE_SHOWN]]]

    def _state(self, agents: list[Agent], now: float) -> Group:
        harness = self.harness
        tasks = list(harness.board.tasks.values())
        open_tasks = [t for t in tasks if not t.archived_at]
        seats = [s for t in open_tasks for s in t.seats]
        filled = sum(len(t.holders) for t in open_tasks)
        pending = list(harness.kb.pending.values())
        unseated = sum(1 for s in pending if not s.jurors)

        live = sum(1 for a in agents if a.alive)
        released = sum(1 for a in agents if a.situation is Situation.RELEASED)
        dead = sum(1 for a in agents if a.situation is Situation.DEAD)
        running = sum(1 for t in open_tasks if t.running)

        rows: list[tuple[str, Text]] = [
            ("elapsed", Text(since(self.app.started_at, now), style=INK)),
            (
                "agents",
                Text.assemble(
                    (f"{live} live", GREEN),
                    ("  ", ""),
                    (f"{released} released", ACCENT if released else FAINT),
                    ("  ", ""),
                    (f"{dead} dead", RED if dead else FAINT),
                ),
            ),
            (
                "tasks",
                Text.assemble(
                    (f"{running} running", GREEN if running else FAINT),
                    ("  ", ""),
                    (f"{len(open_tasks) - running} open", DIM),
                    ("  ", ""),
                    (f"{len(tasks) - len(open_tasks)} closed", FAINT),
                ),
            ),
            ("seats", Text(f"{filled}/{len(seats)} filled", style=DIM)),
            ("knowledge", Text(f"{len(harness.kb.truths)} accepted truths", style=DIM)),
            (
                "jury",
                Text(
                    f"{len(pending)} pending" + (f", {unseated} unseated" if unseated else ""),
                    style=f"bold {AMBER}" if pending else FAINT,
                ),
            ),
        ]
        body = kv(rows)
        if not pending:
            return Group(body)
        alert = Text(
            f" {len(pending)} submission{'s' if len(pending) > 1 else ''} awaiting jury ",
            style=f"bold {AMBER} reverse",
        )
        return Group(body, Text(), alert)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = event.row_key.value
        if key is None:
            return
        if event.data_table.id == "agents":
            self.app.push_screen(AgentScreen(str(key)))
        elif event.data_table.id == "tasks":
            self.app.push_screen(TaskScreen(str(key)))
