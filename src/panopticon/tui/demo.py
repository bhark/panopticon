"""A fake, populated harness so the interface can be built and reviewed without a run.

    uv run python -m panopticon.tui.demo [--still]

Everything here is throwaway scaffolding; nothing else imports it.
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from pathlib import Path

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
from panopticon.services.bus import Bus
from panopticon.services.knowledge import Knowledge
from panopticon.services.taskboard import TaskBoard
from panopticon.services.worktrees import Worktrees
from panopticon.tui.app import PanopticonApp

GOAL = "make panopticon resume a killed run without losing a single transcript"

WINDOWS = {
    "claude-code": 200_000,
    "codex-cli": 272_000,
    "kimi-cli": 256_000,
    "openrouter:glm-4.6": 200_000,
    "openrouter:deepseek-v3": 163_840,
}

ROSTER = [
    ("Alma", "claude-code", Situation.ON_TASK),
    ("Brin", "codex-cli", Situation.ON_TASK),
    ("Cato", "kimi-cli", Situation.ON_TASK),
    ("Dova", "openrouter:glm-4.6", Situation.JURY),
    ("Eik", "claude-code", Situation.JURY),
    ("Fen", "openrouter:deepseek-v3", Situation.IDLE),
    ("Goro", "codex-cli", Situation.IDLE),
    ("Hale", "kimi-cli", Situation.WAITING_FOR_SEATS),
    ("Iris", "claude-code", Situation.IDLE),
    ("Juno", "openrouter:glm-4.6", Situation.CLOSING_TASK),
    ("Kes", "codex-cli", Situation.RELEASED),
    ("Lumo", "openrouter:deepseek-v3", Situation.DEAD),
]

ACTIONS = [
    "view_task_board",
    "send_direct_message",
    "bash",
    "read_file",
    "edit_file",
    "view_knowledge_base",
    "submit_truth",
    "wait",
    "send_shoutboard_message",
    "list_jury_submissions",
]

CHATTER = [
    "state file is written per turn, not per run. confirmed.",
    "worktree for task t3 is dirty; not merging until Brin finalizes.",
    "the codex adapter drops tool ids on retry. filed as a truth.",
    "anyone holding the seat on the resume task? it has been open 40 minutes.",
    "transcripts are append-only on disk now, one jsonl per agent.",
    "do not run the suite in the shared worktree, it clobbers .panopticon/",
    "kimi cli needs --output-format json or the parse fails silently.",
    "reading the queue drain order before I vote on this.",
]

RESULTS = [
    "ok. 3 files changed, 41 insertions, 12 deletions.",
    "wrote .panopticon/state.json (14.2 kB)",
    "no matches. the symbol does not exist outside the tests.",
    "failed: the worktree is locked by another process",
    "sent. Brin will see it on their next turn.",
    "the knowledge base has 5 accepted truths; none of them cover the queue drain.",
]


class DemoHarness:
    """As much of the Harness protocol as the interface actually reads."""

    def __init__(self) -> None:
        self.goal = GOAL
        self.agents: dict[str, Agent] = {}
        self.board = TaskBoard()
        self.bus = Bus(lambda senders: None)
        self.kb = Knowledge()
        self.worktrees = Worktrees(Path.cwd(), Path.cwd() / ".panopticon" / "worktrees")
        self.force_ending = False
        self.started_at = time.time() - 4_237

    def context_window(self, provider: str) -> int:
        return WINDOWS.get(provider, 128_000)


def transcript(rng: random.Random, name: str, count: int, now: float) -> list[Entry]:
    entries: list[Entry] = []
    at = now - count * 22
    for turn in range(1, count + 1):
        at += rng.uniform(8, 40)
        kind = rng.choices(
            ["inbox", "action", "result", "note"], weights=[3, 5, 5, 1], k=1
        )[0]
        if kind == "inbox":
            text = rng.choice(
                [
                    f"{rng.choice(['Brin', 'Cato', 'Iris'])} wrote to the shoutboard.",
                    "your last action returned. see below.",
                    "a submission is waiting for jury.",
                    f"direct message from {rng.choice(['Alma', 'Goro'])}: "
                    f"{rng.choice(CHATTER)}",
                ]
            )
        elif kind == "action":
            tool = rng.choice(ACTIONS)
            text = f"{tool}({rng.choice(['', 'path=src/panopticon/state.py', 'to=Brin'])})"
        elif kind == "result":
            text = rng.choice(RESULTS)
        else:
            text = f"{name}: keeping the seat until the worktree is clean."
        entries.append(Entry(kind=kind, text=text, turn=turn, at=at))
    return entries


def build(rng: random.Random) -> DemoHarness:
    harness = DemoHarness()
    now = time.time()

    for index, (name, provider, situation) in enumerate(ROSTER):
        turns = rng.randint(18, 240)
        agent = Agent(
            name=name,
            provider=provider,
            situation=situation,
            born_at=now - rng.uniform(900, 4_200),
            turns=turns,
            last_action=rng.choice(ACTIONS),
            entries=transcript(rng, name, rng.choice([24, 60, 130, 420]), now),
            usage=Usage(
                input_tokens=turns * rng.randint(900, 2_400),
                output_tokens=turns * rng.randint(60, 300),
                cache_read=turns * rng.randint(4_000, 20_000),
                cache_write=turns * rng.randint(200, 900),
                cost_usd=turns * rng.uniform(0.004, 0.02),
                context_tokens=int(WINDOWS[provider] * rng.uniform(0.06, 0.94)),
            ),
        )
        if situation is Situation.RELEASED:
            agent.voted_goal_reached = True
        if situation is Situation.DEAD:
            agent.consecutive_failures = 3
            agent.usage.context_tokens = 0
        if situation is Situation.IDLE and index % 3 == 0:
            agent.wake_at = now + rng.uniform(60, 900)
        if situation is Situation.CLOSING_TASK:
            agent.transient = True
        if index % 4 == 0:
            agent.inbox.append(QueueItem(kind="shout", text="three agents wrote to the shoutboard"))
        harness.agents[name] = agent

    _tasks(harness, now)
    _knowledge(harness, now)
    _shouts(harness, now)
    return harness


def _tasks(harness: DemoHarness, now: float) -> None:
    running = Task(
        id="t1",
        title="make the state file crash-safe",
        description="write .panopticon/state.json through a temp file and rename. "
        "one writer, no partial reads on resume.",
        seats=[
            Seat(role="implementer", holder="Alma", assigned_at=now - 2_100,
                 finalization=Finalization(
                     reason="atomic rename is in and covered",
                     conclusion="state.json is written via .tmp then os.replace; "
                     "two tests cover a torn write.",
                 )),
            Seat(role="reviewer", holder="Brin", assigned_at=now - 2_050),
        ],
        created_by="Alma",
        created_at=now - 2_400,
        started_at=now - 2_050,
        worktree=".panopticon/worktrees/t1",
    )
    waiting = Task(
        id="t2",
        title="map every provider adapter onto one transcript shape",
        description="three seats: one per family of adapter. no seat starts until all three fill.",
        seats=[
            Seat(role="cli adapters", holder="Hale", assigned_at=now - 2_500),
            Seat(role="api adapters"),
            Seat(role="conformance tests"),
        ],
        created_by="Hale",
        created_at=now - 2_600,
    )
    solo = Task(
        id="t3",
        title="drain the agent queues on pause",
        description="pause has to stop between turns, never inside one.",
        seats=[Seat(role="implementer", holder="Cato", assigned_at=now - 620)],
        created_by="Cato",
        created_at=now - 700,
        started_at=now - 620,
        worktree=".panopticon/worktrees/t3",
    )
    closed_one = Task(
        id="t0",
        title="pin the tokenizer used for the context estimate",
        description="the estimate drifted between providers.",
        seats=[
            Seat(role="implementer", holder="Iris", assigned_at=now - 5_400,
                 finalization=Finalization(reason="done", conclusion="one tokenizer, one number")),
        ],
        created_by="Iris",
        created_at=now - 5_600,
        started_at=now - 5_400,
        archived_at=now - 3_100,
        worktree=".panopticon/worktrees/t0",
        closer="Juno",
        outcome="merged into main. the estimate now comes from one tokenizer for every "
        "provider, and the drift note was submitted to the knowledge base and accepted.",
    )
    closed_two = Task(
        id="tm1",
        title="decide whether we need a message queue at all",
        description="NATS was a suggestion, not a decision.",
        seats=[
            Seat(role="investigator", holder="Goro", assigned_at=now - 7_000,
                 finalization=Finalization(reason="not needed",
                                           conclusion="one process, one loop, asyncio is enough")),
            Seat(role="second opinion", holder="Fen", assigned_at=now - 6_900,
                 finalization=Finalization(reason="agreed", conclusion="no broker")),
        ],
        created_by="Goro",
        created_at=now - 7_200,
        started_at=now - 6_900,
        archived_at=now - 5_800,
        worktree=".panopticon/worktrees/tm1",
        closer="Juno",
        outcome="worktree discarded; the finding went into the knowledge base instead of code.",
    )
    for task in (running, waiting, solo, closed_one, closed_two):
        harness.board.tasks[task.id] = task

    harness.agents["Alma"].task_id = "t1"
    harness.agents["Brin"].task_id = "t1"
    harness.agents["Hale"].task_id = "t2"
    harness.agents["Cato"].task_id = "t3"
    harness.agents["Juno"].task_id = "tm1"


def _knowledge(harness: DemoHarness, now: float) -> None:
    harness.kb.truths = [
        Truth(id="k1", title="codex exec needs --json or the tool call is unparseable",
              body="stdout is prose otherwise. flag confirmed on 0.9.4.",
              submitted_by="Brin", accepted_at=now - 4_000),
        Truth(id="k2", title="one tokenizer for every provider's context estimate",
              body="providers disagree by up to 12%. we count with tiktoken o200k and "
              "treat the provider figure as advisory.",
              submitted_by="Iris", accepted_at=now - 3_050),
        Truth(id="k3", title="a seat held by a dead agent is released after 60 minutes",
              body="janitor tick, EXPIRE_MINUTES=60. verified against services/janitor.py.",
              submitted_by="Goro", accepted_at=now - 2_400),
        Truth(id="k4", title="panopticon needs no message broker",
              body="single process, single asyncio loop, one queue per agent. NATS was "
              "considered and dropped.",
              submitted_by="Goro", accepted_at=now - 5_700),
        Truth(id="k5", title="transcripts are append-only, one jsonl per agent",
              body=".panopticon/transcripts/<name>.jsonl. never rewritten, so the prompt "
              "prefix stays cacheable.",
              submitted_by="Alma", accepted_at=now - 1_200),
    ]
    harness.kb.pending = {
        "s1": Submission(
            id="s1",
            title="the shoutboard debounce extends on every new message, so a busy "
            "board never flushes",
            body="Bus.DEBOUNCE_SECONDS=180 and the timer restarts per shout. with 12 agents "
            "the burst never settles.",
            submitted_by="Fen",
            submitted_at=now - 700,
            jurors=["Dova", "Eik"],
            verdicts=[
                Verdict(juror="Dova", call=VerdictCall.TRUE,
                        reasoning="read Bus.run; the sleep is reset, not capped.", at=now - 300),
            ],
        ),
        "s2": Submission(
            id="s2",
            title="resume rebuilds an agent's situation from state.json alone",
            body="restated: the transcript is replayed for context but the situation comes "
            "from state.json, so the two can disagree after a crash mid-turn.",
            submitted_by="Alma",
            submitted_at=now - 220,
            jurors=[],
            restated_from="s0",
        ),
    }
    harness.agents["Dova"].submission_id = "s1"
    harness.agents["Eik"].submission_id = "s1"


def _shouts(harness: DemoHarness, now: float) -> None:
    senders = ["Alma", "Brin", "Cato", "Goro", "human", "Iris", "Fen", "Hale"]
    for index, body in enumerate(CHATTER):
        harness.bus.shouts.append(
            Shout(
                sender=senders[index % len(senders)],
                body=body,
                at=now - (len(CHATTER) - index) * 190,
            )
        )
    harness.bus.shouts.append(
        Shout(sender="human", body="stop gold-plating the state file. resume is the goal.",
              at=now - 120)
    )


async def simulate(app: PanopticonApp, harness: DemoHarness, rng: random.Random) -> None:
    """Keep the board moving so the throttle and the live panels get exercised."""
    while True:
        await asyncio.sleep(rng.uniform(0.15, 0.9))
        live = [
            a
            for a in harness.agents.values()
            if a.alive and a.situation is not Situation.RELEASED
        ]
        if not live:
            continue
        agent = rng.choice(live)
        agent.turns += 1
        agent.last_action = rng.choice(ACTIONS)
        agent.usage.context_tokens = min(
            WINDOWS[agent.provider],
            agent.usage.context_tokens + rng.randint(200, 2_600),
        )
        agent.usage.input_tokens += rng.randint(800, 3_000)
        agent.usage.output_tokens += rng.randint(40, 400)
        agent.usage.cost_usd += rng.uniform(0.002, 0.03)
        agent.entries.append(
            Entry(kind="action", text=f"{agent.last_action}()", turn=agent.turns)
        )
        agent.entries.append(Entry(kind="result", text=rng.choice(RESULTS), turn=agent.turns))
        app.harness_event(Event(kind="action", text=agent.last_action, agent=agent.name))
        if rng.random() < 0.06:
            harness.bus.shouts.append(Shout(sender=agent.name, body=rng.choice(CHATTER)))
            app.harness_event(Event(kind="shout", text="wrote to the shoutboard", agent=agent.name))


async def main() -> None:
    rng = random.Random(7)
    harness = build(rng)
    app = PanopticonApp(harness)

    def shout(body: str) -> None:
        harness.bus.shouts.append(Shout(sender=app.human_name, body=body))
        app.harness_event(Event(kind="shout", text=f"human shouted: {body}", agent="human"))

    def pause() -> None:
        app.harness_event(Event(kind="system", text="pause requested; draining"))

    def force_end() -> None:
        harness.force_ending = True
        app.harness_event(Event(kind="system", text="human forced the session to end"))

    app.on_shout = shout
    app.on_pause = pause
    app.on_force_end = force_end

    ticker = None
    if "--still" not in sys.argv:
        ticker = asyncio.create_task(simulate(app, harness, rng))
    try:
        await app.run_async()
    finally:
        if ticker:
            ticker.cancel()


if __name__ == "__main__":
    asyncio.run(main())
