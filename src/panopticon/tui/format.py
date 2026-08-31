"""Formatting, colour and repaint-throttling rules for the interface.

Deliberately free of Textual and Rich so the parts worth testing - the paint
throttle, the context-window arithmetic, transcript trimming - stay plain
functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from panopticon.model import Agent, Entry, Situation, Submission, Task

# palette; a calm slate board with one accent, kept here so tables and CSS agree
INK = "#c8d0da"
DIM = "#6b7785"
FAINT = "#4a5058"
ACCENT = "#4fd6be"
BLUE = "#7cc5ff"
AMBER = "#e8b657"
RED = "#e26d6d"
GREEN = "#7fd88f"
VIOLET = "#b48ead"

SITUATION_STYLE: dict[Situation, str] = {
    Situation.IDLE: DIM,
    Situation.WAITING_FOR_SEATS: AMBER,
    Situation.ON_TASK: GREEN,
    Situation.JURY: VIOLET,
    Situation.CLOSING_TASK: BLUE,
    Situation.RELEASED: f"bold {ACCENT}",
    Situation.RELIEVED: FAINT,
    Situation.DEAD: f"bold {RED}",
}

SITUATION_LABEL: dict[Situation, str] = {
    Situation.IDLE: "idle",
    Situation.WAITING_FOR_SEATS: "waiting for seats",
    Situation.ON_TASK: "on task",
    Situation.JURY: "jury duty",
    Situation.CLOSING_TASK: "closing task",
    Situation.RELEASED: "released",
    Situation.RELIEVED: "relieved",
    Situation.DEAD: "dead",
}

ENTRY_STYLE: dict[str, str] = {
    "inbox": BLUE,
    "action": ACCENT,
    "result": INK,
    "summary": VIOLET,
    "note": DIM,
}

VERDICT_STYLE = {"true": GREEN, "false": RED, "restate": AMBER}

TRANSCRIPT_TAIL = 250  # entries kept on screen; older ones stay in the log, not the eye
ENTRY_CLIP = 3000


def elapsed(seconds: float) -> str:
    """Coarse on purpose: a monitoring board should not tick every second."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def since(at: float, now: float | None = None) -> str:
    return elapsed((now if now is not None else time.time()) - at)


def clock(at: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(at))


def context_pct(used: int, window: int) -> float | None:
    """None when the provider has not reported a size yet, so we can say so."""
    if window <= 0 or used <= 0:
        return None
    return min(100.0, used / window * 100.0)


def bar(fraction: float, width: int = 8) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def context_cell(used: int, window: int) -> tuple[str, str]:
    """(text, style) for the context column; amber then red as the window fills."""
    pct = context_pct(used, window)
    if pct is None:
        return "  -", FAINT
    style = RED if pct >= 90 else AMBER if pct >= 70 else DIM
    return f"{pct:3.0f}% {bar(pct / 100)}", style


def clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def oneline(text: str, limit: int) -> str:
    return clip(" ".join(text.split()), limit)


def tail_entries(entries: list[Entry], limit: int = TRANSCRIPT_TAIL) -> tuple[list[Entry], int]:
    """Newest `limit` entries plus how many were left behind."""
    if len(entries) <= limit:
        return list(entries), 0
    return entries[-limit:], len(entries) - limit


def agent_mark(agent: Agent) -> tuple[str, str]:
    """Glyph and style. Voted-out and dead agents have to read at a glance."""
    style = SITUATION_STYLE[agent.situation]
    if agent.situation is Situation.DEAD:
        return "x", style
    if agent.voted_goal_reached or agent.situation is Situation.RELEASED:
        return "+", f"bold {ACCENT}"
    if agent.situation is Situation.RELIEVED:
        return "-", style
    return " ", style


def task_state(task: Task) -> tuple[str, str]:
    filled = sum(1 for s in task.seats if s.holder)
    finalized = sum(1 for s in task.seats if s.finalization)
    if task.archived_at:
        return "closed", FAINT
    if finalized:
        return f"finalizing {finalized}/{len(task.seats)}", VIOLET
    if task.running:
        return "running", GREEN
    if filled == len(task.seats):
        return "ready", BLUE
    return "open", AMBER


def tally(submission: Submission) -> str:
    counts = {"true": 0, "false": 0, "restate": 0}
    for verdict in submission.verdicts:
        counts[str(verdict.call)] += 1
    return f"{counts['true']} true / {counts['false']} false / {counts['restate']} restate"


def describe_wait(
    agent: Agent, task: Task | None, submission: Submission | None, now: float | None = None
) -> str:
    """What this agent is actually blocked on right now, in one line."""
    now = now if now is not None else time.time()
    if agent.situation is Situation.DEAD:
        return "nothing; it died"
    if agent.situation is Situation.RELIEVED:
        return "nothing; relieved"
    if agent.inbox:
        return f"nothing; {len(agent.inbox)} queued to drain next turn"
    if agent.situation is Situation.WAITING_FOR_SEATS and task:
        filled = sum(1 for s in task.seats if s.holder)
        return f"seats on {task.title!r}: {filled}/{len(task.seats)} filled"
    if agent.situation is Situation.ON_TASK and task:
        return f"work on {task.title!r}"
    if agent.situation is Situation.JURY and submission:
        return f"verdict on {submission.title!r}"
    if agent.situation is Situation.RELEASED:
        return "release; voted goal reached"
    if agent.wake_at:
        return f"timer, {elapsed(agent.wake_at - now)} left"
    return "anything that wakes it"


@dataclass(slots=True)
class Coalescer:
    """A dozen agents acting at once must still cost one repaint.

    An event only marks the board dirty; the paint tick decides. `max_stale`
    keeps clocks and counters honest when nothing is emitting at all.
    """

    min_interval: float = 0.35
    max_stale: float = 2.0
    dirty: bool = True
    last_paint: float = field(default=-1e9)

    def note(self) -> None:
        self.dirty = True

    def due(self, now: float) -> bool:
        waited = now - self.last_paint
        if waited < self.min_interval:
            return False
        if not self.dirty and waited < self.max_stale:
            return False
        self.dirty = False
        self.last_paint = now
        return True
