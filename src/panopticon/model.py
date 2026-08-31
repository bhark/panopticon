"""Core data shapes. Every other module keys off these."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from panopticon.services.bus import Bus
    from panopticon.services.knowledge import Knowledge
    from panopticon.services.taskboard import TaskBoard
    from panopticon.services.worktrees import Worktrees


class Situation(StrEnum):
    IDLE = "idle"
    WAITING_FOR_SEATS = "waiting_for_seats"
    ON_TASK = "on_task"
    JURY = "jury"
    CLOSING_TASK = "closing_task"
    RELEASED = "released"
    RELIEVED = "relieved"
    DEAD = "dead"


class VerdictCall(StrEnum):
    TRUE = "true"
    FALSE = "false"
    RESTATE = "restate"


# transcript

@dataclass(slots=True)
class Entry:
    """One line of an agent's transcript. Appended, never rewritten."""

    kind: str  # inbox | action | result | summary | note
    text: str
    turn: int = 0
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    context_tokens: int = 0  # provider-reported total, 0 when unknown
    measured_entries: int = 0  # entries the figure above covered, for the hybrid estimate


# actions

@dataclass(frozen=True, slots=True)
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass(slots=True)
class ActionResult:
    ok: bool
    text: str

    @staticmethod
    def fail(text: str) -> ActionResult:
        return ActionResult(False, text)


@dataclass(slots=True)
class QueueItem:
    """Anything waiting for an agent: its last result, a DM, a board event, a timer."""

    kind: str  # result | dm | shout | board | task | jury | system | timer
    text: str
    at: float = field(default_factory=time.time)


# task board

@dataclass(slots=True)
class Finalization:
    reason: str
    conclusion: str
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Seat:
    role: str
    holder: str | None = None
    assigned_at: float | None = None
    nudges_sent: int = 0
    finalization: Finalization | None = None


@dataclass(slots=True)
class Task:
    id: str
    title: str
    description: str
    seats: list[Seat]
    created_by: str
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    archived_at: float | None = None
    worktree: str | None = None
    closer: str | None = None
    outcome: str | None = None

    @property
    def ready(self) -> bool:
        return all(s.holder for s in self.seats)

    @property
    def running(self) -> bool:
        return self.started_at is not None and self.archived_at is None

    @property
    def holders(self) -> list[str]:
        return [s.holder for s in self.seats if s.holder]

    def seat_of(self, agent: str) -> Seat | None:
        return next((s for s in self.seats if s.holder == agent), None)


# knowledge base

@dataclass(slots=True)
class Truth:
    id: str
    title: str
    body: str
    submitted_by: str
    accepted_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Verdict:
    juror: str
    call: VerdictCall
    reasoning: str
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Submission:
    id: str
    title: str
    body: str
    submitted_by: str
    submitted_at: float = field(default_factory=time.time)
    jurors: list[str] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    restated_from: str | None = None


# messages

@dataclass(slots=True)
class DirectMessage:
    sender: str
    recipient: str
    body: str
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Shout:
    sender: str
    body: str
    at: float = field(default_factory=time.time)


# agents

@dataclass(slots=True)
class Agent:
    name: str
    provider: str
    situation: Situation = Situation.IDLE
    born_at: float = field(default_factory=time.time)
    turns: int = 0
    task_id: str | None = None
    submission_id: str | None = None
    seen_kb: bool = False
    wake_at: float | None = None
    voted_goal_reached: bool = False
    transient: bool = False  # task closer; excluded from vote tally and jury minimum
    consecutive_failures: int = 0
    last_action: str = ""
    entries: list[Entry] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # runtime only, never persisted; a list rather than a Queue so it can be read without draining
    inbox: list[QueueItem] = field(default_factory=list, repr=False)
    wakeup: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def alive(self) -> bool:
        return self.situation not in (Situation.RELIEVED, Situation.DEAD)

    @property
    def counts_toward_goal(self) -> bool:
        return not self.transient and self.situation is not Situation.DEAD


@dataclass(slots=True)
class Event:
    """Anything worth showing the human or writing to the audit log."""

    kind: str
    text: str
    agent: str | None = None
    at: float = field(default_factory=time.time)


class Harness(Protocol):
    """What a tool handler is allowed to reach. Implemented by the orchestrator."""

    goal: str
    agents: dict[str, Agent]
    board: TaskBoard
    bus: Bus
    kb: Knowledge
    worktrees: Worktrees
    force_ending: bool

    def post(self, recipient: str, item: QueueItem) -> None: ...

    def broadcast(self, item: QueueItem, exclude: tuple[str, ...] = ()) -> None: ...

    def emit(self, event: Event) -> None: ...

    def enter(self, agent: Agent, situation: Situation, preprompt: str) -> None:
        """Move an agent to a new situation, clearing its transcript."""
        ...

    async def launch_task(self, task: Task) -> None:
        """All seats filled: cut the worktree and put every holder on the task."""
        ...

    def leave_task(self, name: str, task: Task, why: str, notify: bool = True) -> None:
        """Release a seat. The only correct way to do it: it also settles everyone left behind."""
        ...

    async def close_task(self, task: Task) -> None:
        """All holders agreed to finalize: release them and spin up the closer."""
        ...

    def retire(self, agent: Agent) -> None:
        """End an agent's loop: a transient closer that is done, or a relieved agent."""
        ...

    def tally_goal(self) -> None:
        """Recount goal-reached votes and end the session if they are unanimous."""
        ...


Handler = Callable[["ToolCtx", dict[str, Any]], Awaitable[ActionResult]]


@dataclass(slots=True)
class ToolCtx:
    agent: Agent
    harness: Harness


@dataclass(frozen=True, slots=True)
class ArgSpec:
    type: str  # string | int | string[]
    description: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args: dict[str, ArgSpec]
    handler: Handler
