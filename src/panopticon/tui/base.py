"""The screen contract: state is pulled on the app's paint tick, never pushed per event."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable
from textual.widgets.data_table import CellDoesNotExist, RowDoesNotExist

if TYPE_CHECKING:
    from panopticon.model import Harness
    from panopticon.tui.app import PanopticonApp


class LiveScreen(Screen[None]):
    # every one of these reads or pokes the harness, so they hang off the screens that have one
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s", "app.shout", "shout"),
        Binding("k", "app.knowledge", "knowledge"),
        Binding("p", "app.pause", "pause"),
        Binding("f", "app.force_end", "force end"),
        Binding("question_mark", "app.help", "help", key_display="?"),
        Binding("q", "app.leave", "quit"),
    ]

    @property
    def harness(self) -> Harness:
        app: PanopticonApp = self.app  # type: ignore[assignment]
        return app.harness

    def refresh_data(self) -> None:
        """Redraw from current harness state. Called at most a few times a second."""


def repaint_table(table: DataTable, rows: list[tuple[str, list[Text]]]) -> None:
    """Rebuild a table without stealing the cursor the human parked on a row."""
    parked = cursor_key(table)
    table.clear()
    for key, cells in rows:
        table.add_row(*cells, key=key)
    if parked is None:
        return
    with contextlib.suppress(RowDoesNotExist):
        table.move_cursor(row=table.get_row_index(parked), scroll=False)


def cursor_key(table: DataTable) -> str | None:
    try:
        return str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
    except CellDoesNotExist:
        return None
