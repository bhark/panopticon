"""Direct messages and the shoutboard."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from panopticon import store
from panopticon.model import DirectMessage, Shout


class Bus:
    DEBOUNCE_SECONDS = 180
    MAX_AGE_SECONDS = 3600
    MAX_TOKENS = 10_000

    def __init__(self, flush: Callable[[list[str]], None]) -> None:
        """`flush` is called with the sender names once a shout burst settles."""
        self.direct: list[DirectMessage] = []
        self.shouts: list[Shout] = []
        self._flush = flush
        self._batch: list[str] = []
        self._shouted = asyncio.Event()

    def send_direct(self, sender: str, recipient: str, body: str) -> DirectMessage:
        message = DirectMessage(sender=sender, recipient=recipient, body=body)
        self.direct.append(message)
        return message

    def shout(self, sender: str, body: str) -> Shout:
        shout = Shout(sender=sender, body=body)
        self.shouts.append(shout)
        if sender not in self._batch:
            self._batch.append(sender)
        self._shouted.set()
        return shout

    def render_shoutboard(self) -> str:
        """Newest first, older than MAX_AGE_SECONDS dropped, truncated to MAX_TOKENS."""
        cutoff = time.time() - self.MAX_AGE_SECONDS
        budget = self.MAX_TOKENS * 4  # no tokenizer here, chars/4 is close enough
        lines: list[str] = []
        used = 0
        for shout in reversed(self.shouts):
            if shout.at < cutoff:
                break
            at = time.strftime("%H:%M", time.localtime(shout.at))
            line = f"{at} {shout.sender}: {shout.body}"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line) + 1
        if not lines:
            return "shoutboard: empty"
        return "shoutboard (newest first, last hour)\n" + "\n".join(lines)

    def snapshot(self) -> list[dict]:
        return [store.dump(s) for s in self.shouts]

    def restore(self, raw: list[dict]) -> None:
        self.shouts = [store.load(Shout, s) for s in raw]

    async def run(self) -> None:
        """The debounce timer. Cancelled on shutdown."""
        while True:
            await self._shouted.wait()
            self._shouted.clear()
            while True:
                try:
                    await asyncio.wait_for(self._shouted.wait(), self.DEBOUNCE_SECONDS)
                except TimeoutError:
                    break
                self._shouted.clear()  # another shout, so the window starts over
            senders, self._batch = self._batch, []
            if senders:
                self._flush(senders)
