"""The human's window into the panopticon.

The orchestrator owns the state; this only reads it. Two things come the other
way: `harness_event`, which the orchestrator calls for anything worth showing,
and the three callbacks it sets for the human's actions. Events never paint
directly - they mark the board dirty and a timer decides, so twelve agents
taking turns at once still cost one repaint.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import ClassVar

from textual.app import App
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.theme import Theme

from panopticon.model import Agent, Event, Harness
from panopticon.tui.dialogs import ConfirmScreen, HelpScreen, ShoutboxScreen
from panopticon.tui.format import Coalescer
from panopticon.tui.knowledge import KnowledgeScreen
from panopticon.tui.overview import OverviewScreen

EVENT_LOG = 300
PAINT_INTERVAL = 0.2

PANOPTICON_THEME = Theme(
    name="panopticon",
    primary="#4fd6be",
    secondary="#7cc5ff",
    accent="#b48ead",
    warning="#e8b657",
    error="#e26d6d",
    success="#7fd88f",
    foreground="#c8d0da",
    background="#0e1116",
    surface="#141920",
    panel="#1b2129",
    dark=True,
)


class PanopticonApp(App[None]):
    CSS_PATH = "panopticon.tcss"
    TITLE = "panopticon"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s", "shout", "shout"),
        Binding("k", "knowledge", "knowledge"),
        Binding("p", "pause", "pause"),
        Binding("f", "force_end", "force end"),
        Binding("question_mark", "help", "help", key_display="?"),
        Binding("q", "leave", "quit"),
    ]

    def __init__(self, harness: Harness) -> None:
        super().__init__()
        self.harness = harness
        self.on_shout: Callable[[str], None] = lambda body: None
        self.on_pause: Callable[[], None] = lambda: None
        self.on_force_end: Callable[[], None] = lambda: None
        self.human_name = "human"
        self.started_at = harness.started_at
        self.events: deque[Event] = deque(maxlen=EVENT_LOG)
        self.paint = Coalescer()
        self.pausing = False

    def get_default_screen(self) -> Screen[None]:
        return OverviewScreen()

    def on_mount(self) -> None:
        self.register_theme(PANOPTICON_THEME)
        self.theme = "panopticon"
        self.set_interval(PAINT_INTERVAL, self._paint)

    # from the orchestrator

    def harness_event(self, event: Event) -> None:
        """Anything worth showing the human. Safe to call before the app is mounted."""
        self.events.append(event)
        self.paint.note()

    # reads

    def context_window(self, agent: Agent) -> int:
        return self.harness.context_window(agent)

    def banner(self) -> str:
        if self.harness.force_ending:
            return "force ending · draining"
        if self.pausing:
            return "pausing · draining"
        return ""

    def _paint(self) -> None:
        if not self.paint.due(time.monotonic()):
            return
        refresh = getattr(self.screen, "refresh_data", None)
        if callable(refresh):
            refresh()

    # human actions

    def action_shout(self) -> None:
        self._open(ShoutboxScreen())

    def action_knowledge(self) -> None:
        self._open(KnowledgeScreen())

    def action_help(self) -> None:
        self._open(HelpScreen())

    def action_pause(self) -> None:
        self.push_screen(
            ConfirmScreen(
                "pause",
                "pause the harness?",
                "every agent finishes what it is doing, then the run is saved "
                "and you can pick it up with `panopticon resume`.",
            ),
            self._pause_confirmed,
        )

    def action_force_end(self) -> None:
        self.push_screen(
            ConfirmScreen(
                "force end",
                "force the session to end?",
                "every agent is told to wrap up now, and each one either votes goal "
                "reached or relieves itself. this cannot be taken back.",
            ),
            self._force_end_confirmed,
        )

    def action_leave(self) -> None:
        self.push_screen(
            ConfirmScreen(
                "quit",
                "close the interface?",
                "the run stops with it. use pause instead if you want it back.",
            ),
            self._leave_confirmed,
        )

    def _pause_confirmed(self, ok: bool | None) -> None:
        if not ok:
            return
        self.pausing = True
        self.paint.note()
        self.on_pause()
        self.notify("draining; state is saved for `panopticon resume`", title="pausing")

    def _force_end_confirmed(self, ok: bool | None) -> None:
        if not ok:
            return
        self.paint.note()
        self.on_force_end()
        self.notify(
            "every agent has been told to wrap up", title="force ending", severity="warning"
        )

    def _leave_confirmed(self, ok: bool | None) -> None:
        if ok:
            self.exit()

    def _open(self, screen: Screen[None]) -> None:
        if type(self.screen) is type(screen):
            return
        self.push_screen(screen)
