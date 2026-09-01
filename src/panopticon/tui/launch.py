"""Opening a panopticon: what the flags did not say is asked here, not in the terminal."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Static

from panopticon.config import MIN_AGENTS
from panopticon.model import Harness, Level
from panopticon.tui.format import AMBER, DIM, FAINT

if TYPE_CHECKING:
    from panopticon.tui.app import PanopticonApp


@dataclass(slots=True)
class Launch:
    """What the interface needs to open a panopticon of its own."""

    roster: dict[Level, int]
    providers: list[str]
    open: Callable[[str, dict[Level, int]], Harness]
    resume: Callable[[], Harness] | None = None


class LaunchScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "app.quit", "quit")]

    @property
    def panopticon(self) -> PanopticonApp:
        app: PanopticonApp = self.app  # type: ignore[assignment]
        return app

    @property
    def launch(self) -> Launch:
        assert self.panopticon.launch is not None
        return self.panopticon.launch

    def compose(self) -> ComposeResult:
        with Vertical(id="launchdialog"):
            yield Static(Text("goal", style=FAINT))
            yield Input(placeholder="what should they work toward?", id="goal")
            with Horizontal(id="launchroster"):
                yield Static(Text("roster", style=FAINT), id="rosterlabel")
                for level, count in self.launch.roster.items():
                    yield Static(Text(str(level), style=DIM), classes="levellabel")
                    yield Input(str(count), type="integer", id=f"n-{level}", classes="levelcount")
            yield Static(id="launchtally")
            yield Static(id="launchhint")
            with Horizontal(id="launchbuttons"):
                if self.launch.resume:
                    yield Button("resume the saved run", id="resume")
                yield Button("open  (enter)", id="open", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#launchdialog").border_title = "panopticon"
        self.query_one("#goal", Input).focus()
        self.refresh_data()

    def refresh_data(self) -> None:
        count = sum(self._roster().values())
        across = ", ".join(self.launch.providers)
        self.query_one("#launchtally", Static).update(
            Text(f"{count} agents across {across}", style=DIM)
        )
        note = self.panopticon.update_note
        self.query_one("#launchhint", Static).update(
            Text(note or "escape quits", style=AMBER if note else FAINT)
        )

    def on_input_changed(self) -> None:
        self.refresh_data()

    def on_input_submitted(self) -> None:
        self._open()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "resume":
            self.panopticon.show(self.launch.resume())
        else:
            self._open()

    def _roster(self) -> dict[Level, int]:
        return {level: self._count(level) for level in self.launch.roster}

    def _count(self, level: Level) -> int:
        raw = self.query_one(f"#n-{level}", Input).value.strip()
        return max(0, int(raw)) if raw.lstrip("-").isdigit() else 0

    def _open(self) -> None:
        goal = self.query_one("#goal", Input).value.strip()
        roster = self._roster()
        if not goal:
            self.notify("a panopticon needs a goal", severity="warning")
        elif sum(roster.values()) < MIN_AGENTS:
            self.notify(f"at least {MIN_AGENTS} agents are needed for a jury", severity="warning")
        else:
            self.panopticon.show(self.launch.open(goal, roster))
