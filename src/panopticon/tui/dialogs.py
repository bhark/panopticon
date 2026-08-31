"""The three things the human can actually do, plus the key map."""

from __future__ import annotations

from typing import ClassVar

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from panopticon.tui.format import ACCENT, AMBER, DIM, FAINT, INK, RED
from panopticon.tui.render import shout_block

SHOUT_TAIL = 40

KEYS = [
    ("up / down", "move between rows"),
    ("enter", "open the agent or task under the cursor"),
    ("tab", "move between panels"),
    ("escape", "back"),
    ("s", "shoutbox; your only voice in here"),
    ("k", "knowledge base and jury"),
    ("p", "pause: drain the harness and save for `panopticon resume`"),
    ("f", "force end: tell every agent to wrap up"),
    ("q", "quit"),
    ("?", "this"),
]


class ShoutboxScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "dismiss", "close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="shoutdialog"):
            with VerticalScroll(id="shouthistory"):
                yield Static(id="shoutlog")
            yield Input(placeholder="shout at the panopticon", id="shoutinput")
            yield Static(
                Text("enter sends · escape closes", style=FAINT), id="shouthint"
            )

    def on_mount(self) -> None:
        self.query_one("#shoutdialog").border_title = "shoutbox"
        self.refresh_data()
        self.query_one("#shoutinput", Input).focus()
        self.query_one("#shouthistory", VerticalScroll).scroll_end(animate=False)

    def refresh_data(self) -> None:
        history = self.query_one("#shouthistory", VerticalScroll)
        at_end = history.scroll_offset.y >= history.max_scroll_y - 1
        self.query_one("#shoutlog", Static).update(
            shout_block(self.app.harness.bus.shouts, SHOUT_TAIL, self.app.human_name)
        )
        if at_end:
            self.call_after_refresh(history.scroll_end, animate=False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        body = event.value.strip()
        if not body:
            return
        event.input.value = ""
        self.app.on_shout(body)
        self.app.paint.note()
        self.notify(body, title="shouted", timeout=3)


class ConfirmScreen(ModalScreen[bool]):
    """Pause and force-end cannot be taken back, so they are asked for twice."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "cancel"),
        Binding("n", "cancel", "no"),
        Binding("y", "confirm", "yes"),
    ]

    def __init__(self, title: str, question: str, detail: str) -> None:
        super().__init__()
        self.dialog_title = title
        self.question = question
        self.detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="confirmdialog"):
            yield Static(Text(self.question, style=f"bold {INK}"), id="confirmquestion")
            yield Static(Text(self.detail, style=DIM), id="confirmdetail")
            with Horizontal(id="confirmbuttons"):
                yield Button("cancel  (n)", id="cancel")
                yield Button("confirm  (y)", id="confirm", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#confirmdialog").border_title = self.dialog_title
        self.query_one("#cancel", Button).focus()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class HelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss", "close"),
        Binding("question_mark", "dismiss", "close"),
    ]

    def compose(self) -> ComposeResult:
        table = Table.grid(padding=(0, 3))
        table.add_column(style=ACCENT, justify="right", width=10)
        table.add_column(style=INK)
        for key, what in KEYS:
            table.add_row(key, what)
        with Vertical(id="helpdialog"):
            yield Static(table)
            yield Static(
                Text(
                    "\nthe shoutbox is the only channel from you to the agents",
                    style=AMBER,
                )
            )
            yield Static(
                Text("a force end cannot be undone; agents then vote out or relieve themselves",
                     style=RED)
            )

    def on_mount(self) -> None:
        self.query_one("#helpdialog").border_title = "keys"
