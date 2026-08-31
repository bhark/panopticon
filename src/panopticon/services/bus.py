"""Direct messages and the shoutboard."""

from __future__ import annotations

from collections.abc import Callable

from panopticon.model import DirectMessage, Shout


class Bus:
    DEBOUNCE_SECONDS = 180
    MAX_AGE_SECONDS = 3600
    MAX_TOKENS = 10_000

    def __init__(self, flush: Callable[[list[str]], None]) -> None:
        """`flush` is called with the sender names once a shout burst settles."""
        self.direct: list[DirectMessage] = []
        self.shouts: list[Shout] = []

    def send_direct(self, sender: str, recipient: str, body: str) -> DirectMessage: ...
    def shout(self, sender: str, body: str) -> Shout: ...
    def render_shoutboard(self) -> str:
        """Newest first, older than MAX_AGE_SECONDS dropped, truncated to MAX_TOKENS."""
        ...

    async def run(self) -> None:
        """The debounce timer. Cancelled on shutdown."""
        ...
