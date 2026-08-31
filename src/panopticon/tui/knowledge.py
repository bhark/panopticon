"""Accepted truths on one side, the jury's open business on the other."""

from __future__ import annotations

import time
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Static

from panopticon.tui.base import LiveScreen
from panopticon.tui.format import AMBER, DIM, GREEN
from panopticon.tui.render import pending_block, truths_block


class KnowledgeScreen(LiveScreen):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "app.pop_screen", "back")]

    def compose(self) -> ComposeResult:
        yield Static(id="detailbar")
        with Horizontal(id="kbbody"):
            with VerticalScroll(id="truthpanel"):
                yield Static(id="truths")
            with VerticalScroll(id="jurypanel"):
                yield Static(id="pending")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#truthpanel").border_title = "knowledge base"
        self.query_one("#jurypanel").border_title = "jury"
        self.query_one("#truthpanel").focus()
        self.refresh_data()

    def refresh_data(self) -> None:
        now = time.time()
        kb = self.harness.kb
        pending = sorted(kb.pending.values(), key=lambda s: s.submitted_at)
        bar = Text(f"{len(kb.truths)} accepted", style=GREEN)
        bar.append("   ", style=DIM)
        bar.append(
            f"{len(pending)} awaiting jury", style=f"bold {AMBER}" if pending else DIM
        )
        self.query_one("#detailbar", Static).update(bar)
        self.query_one("#truths", Static).update(truths_block(kb.truths, now))
        self.query_one("#pending", Static).update(pending_block(pending, now))
