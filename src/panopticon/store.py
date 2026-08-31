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
    Harness,
    QueueItem,
    Seat,
    Situation,
    Submission,
    Task,
    Usage,
    Verdict,
    VerdictCall,
)

STATE_DIRNAME = ".panopticon"


RUNTIME_FIELDS = ("inbox", "wakeup")


def dump(obj: Any) -> Any:
    return asdict(obj) if is_dataclass(obj) else obj


def _dump_agent(agent: Agent) -> dict[str, Any]:
    """asdict deep-copies every field before any filter runs, and an asyncio.Event cannot be
    copied, so the runtime fields have to be skipped before they are ever touched."""
    out: dict[str, Any] = {}
    for f in fields(agent):
        if f.name in RUNTIME_FIELDS:
            continue
        out[f.name] = dump(getattr(agent, f.name))
    out["entries"] = [dump(e) for e in agent.entries]
    return out


def load[T](cls: type[T], data: dict[str, Any]) -> T:
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
            fh.write(json.dumps(dump(event)) + "\n")

    def append_entry(self, agent: str, entry: Entry) -> None:
        with (self.transcripts / f"{agent}.jsonl").open("a") as fh:
            fh.write(json.dumps(dump(entry)) + "\n")

    def save(self, state: dict[str, Any]) -> None:
        """Atomic: a crash mid-write must not leave an unreadable snapshot."""
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, self.state_file)

    def load(self) -> dict[str, Any]:
        return json.loads(self.state_file.read_text())


def snapshot(harness: Harness) -> dict[str, Any]:
    return {
        "version": 2,
        "goal": harness.goal,
        "started_at": harness.started_at,
        "force_ending": harness.force_ending,
        "saved_at": time.time(),
        "agents": [
            _dump_agent(a) | {"pending": [dump(i) for i in a.inbox]}
            for a in harness.agents.values()
        ],
        "board": harness.board.snapshot(),
        "knowledge": harness.kb.snapshot(),
        "bus": harness.bus.snapshot(),
    }


def restore_agent(data: dict[str, Any]) -> Agent:
    entries = [load(Entry, e) for e in data.pop("entries", [])]
    usage = load(Usage, data.pop("usage", {}) or {})
    pending = [load(QueueItem, i) for i in data.pop("pending", [])]
    agent = load(Agent, data)
    agent.entries = entries
    agent.usage = usage
    agent.situation = Situation(data["situation"])
    agent.inbox.extend(pending)  # undelivered DMs must survive a pause
    return agent


def restore_task(data: dict[str, Any]) -> Task:
    seats = []
    for raw in data.pop("seats", []):
        fin = raw.pop("finalization", None)
        seat = load(Seat, raw)
        seat.finalization = load(Finalization, fin) if fin else None
        seats.append(seat)
    task = load(Task, {**data, "seats": []})  # seats is required positionally
    task.seats = seats
    return task


def restore_submission(data: dict[str, Any]) -> Submission:
    verdicts = []
    for raw in data.pop("verdicts", []):
        verdict = load(Verdict, raw)
        verdict.call = VerdictCall(raw["call"])
        verdicts.append(verdict)
    submission = load(Submission, data)
    submission.verdicts = verdicts
    return submission

