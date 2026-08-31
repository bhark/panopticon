"""State on disk, under .panopticon/ in the working directory."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from panopticon.model import (
    Agent,
    Entry,
    Event,
    Finalization,
    QueueItem,
    Seat,
    Shout,
    Situation,
    Submission,
    Task,
    Truth,
    Usage,
    Verdict,
    VerdictCall,
)

STATE_DIRNAME = ".panopticon"


def _drop_runtime(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    return {k: v for k, v in pairs if k != "inbox"}


def _dump(obj: Any) -> Any:
    return asdict(obj, dict_factory=_drop_runtime) if is_dataclass(obj) else obj


def _load[T](cls: type[T], data: dict[str, Any]) -> T:
    """Construct a dataclass from a dict, ignoring fields a newer version dropped."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.transcripts = root / "transcripts"
        self.worktrees = root / "worktrees"
        self.state_file = root / "state.json"
        self.events_file = root / "events.jsonl"

    def prepare(self) -> None:
        self.transcripts.mkdir(parents=True, exist_ok=True)
        self.worktrees.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.state_file.exists()

    # events and transcripts are append-only; the snapshot is rewritten

    def append_event(self, event: Event) -> None:
        with self.events_file.open("a") as fh:
            fh.write(json.dumps(_dump(event)) + "\n")

    def append_entry(self, agent: str, entry: Entry) -> None:
        with (self.transcripts / f"{agent}.jsonl").open("a") as fh:
            fh.write(json.dumps(_dump(entry)) + "\n")

    def save(self, state: dict[str, Any]) -> None:
        """Atomic: a crash mid-write must not leave an unreadable snapshot."""
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, self.state_file)

    def load(self) -> dict[str, Any]:
        return json.loads(self.state_file.read_text())


def snapshot(harness: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "goal": harness.goal,
        "started_at": harness.started_at,
        "force_ending": harness.force_ending,
        "saved_at": time.time(),
        "agents": [
            _dump(a) | {"pending": [_dump(i) for i in _drain(a)]} for a in harness.agents.values()
        ],
        "tasks": [_dump(t) for t in harness.board.tasks.values()],
        "truths": [_dump(t) for t in harness.kb.truths],
        "pending_submissions": [_dump(s) for s in harness.kb.pending.values()],
        "shouts": [_dump(s) for s in harness.bus.shouts],
    }


def _drain(agent: Agent) -> list[QueueItem]:
    """A paused agent's undelivered inbox has to survive, or resume loses its DMs."""
    items = []
    while not agent.inbox.empty():
        items.append(agent.inbox.get_nowait())
    return items


def restore_agent(data: dict[str, Any]) -> Agent:
    entries = [_load(Entry, e) for e in data.pop("entries", [])]
    usage = _load(Usage, data.pop("usage", {}) or {})
    pending = [_load(QueueItem, i) for i in data.pop("pending", [])]
    agent = _load(Agent, data)
    agent.entries = entries
    agent.usage = usage
    agent.situation = Situation(data["situation"])
    for item in pending:
        agent.inbox.put_nowait(item)
    return agent


def restore_task(data: dict[str, Any]) -> Task:
    seats = []
    for raw in data.pop("seats", []):
        fin = raw.pop("finalization", None)
        seat = _load(Seat, raw)
        seat.finalization = _load(Finalization, fin) if fin else None
        seats.append(seat)
    task = _load(Task, data)
    task.seats = seats
    return task


def restore_submission(data: dict[str, Any]) -> Submission:
    verdicts = []
    for raw in data.pop("verdicts", []):
        verdict = _load(Verdict, raw)
        verdict.call = VerdictCall(raw["call"])
        verdicts.append(verdict)
    submission = _load(Submission, data)
    submission.verdicts = verdicts
    return submission


def restore_truth(data: dict[str, Any]) -> Truth:
    return _load(Truth, data)


def restore_shout(data: dict[str, Any]) -> Shout:
    return _load(Shout, data)
