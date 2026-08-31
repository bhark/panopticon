"""The turn loop, and everything that owns it."""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from collections.abc import Callable

from panopticon import prompts, situations, transcript
from panopticon import store as store_mod
from panopticon.config import Config
from panopticon.model import (
    Agent,
    Event,
    QueueItem,
    Situation,
    Submission,
    Task,
    ToolCtx,
)
from panopticon.names import generate
from panopticon.providers.base import Provider, TurnRequest
from panopticon.services.bus import Bus
from panopticon.services.janitor import Janitor
from panopticon.services.knowledge import Knowledge
from panopticon.services.taskboard import TaskBoard
from panopticon.services.worktrees import Worktrees
from panopticon.store import Store
from panopticon.tools import dispatch

MAX_CONSECUTIVE_FAILURES = 4
AUTOSAVE_SECONDS = 30
IDLE_POKE_SECONDS = 900


class Orchestrator:
    HUMAN = "The human"

    def __init__(
        self,
        *,
        goal: str,
        agents: list[Agent],
        providers: dict[str, Provider],
        store: Store,
        worktrees: Worktrees,
        config: Config,
        started_at: float | None = None,
    ) -> None:
        self.goal = goal
        self.agents: dict[str, Agent] = {a.name: a for a in agents}
        self.providers = providers
        self.store = store
        self.worktrees = worktrees
        self.config = config
        self.started_at = started_at or time.time()
        self.board = TaskBoard()
        self.kb = Knowledge()
        self.bus = Bus(self._shout_flushed)
        self.janitor = Janitor(self)
        self.force_ending = False
        self.stopped_because = ""
        self.subscribers: list[Callable[[Event], None]] = []
        self._sem = {k: asyncio.Semaphore(config.max_concurrent_turns) for k in providers}
        self._loops: dict[str, asyncio.Task] = {}
        self._stop = asyncio.Event()
        self._pausing = False
        self._last_activity = time.time()
        self._provider_cycle = itertools.cycle(providers)
        for agent in self.agents.values():
            # a resume against a config that dropped a provider must not wedge on a KeyError
            if agent.provider not in providers:
                agent.situation = Situation.DEAD

    # lifecycle

    async def run(self) -> None:
        self.store.prepare()
        self.emit(Event("session", f"panopticon opened: {self.goal}"))
        background = [
            asyncio.create_task(self.bus.run(), name="bus"),
            asyncio.create_task(self.janitor.run(), name="janitor"),
            asyncio.create_task(self._autosave(), name="autosave"),
            asyncio.create_task(self._idle_watchdog(), name="watchdog"),
        ]
        for agent in self.agents.values():
            self._launch_loop(agent)
        try:
            await self._stop.wait()
        finally:
            for task in [*background, *self._loops.values()]:
                task.cancel()
            await asyncio.gather(*background, *self._loops.values(), return_exceptions=True)
            self.save()

    def stop(self, reason: str) -> None:
        if not self._stop.is_set():
            self.stopped_because = reason
            self.emit(Event("session", f"panopticon closed: {reason}"))
            self._stop.set()

    async def pause(self) -> None:
        """Drain rather than cut: every agent finishes the turn it is in, then we save."""
        self._pausing = True
        self.broadcast(
            QueueItem("system", "The human paused the session. Your work is being saved.")
        )
        await asyncio.gather(*self._loops.values(), return_exceptions=True)
        self.stop("paused by the human")

    def force_end(self) -> None:
        if self.force_ending:
            return
        self.force_ending = True
        self.broadcast(
            QueueItem(
                "system",
                "The human is ending this session. Finish up as quickly as you can. You can now "
                "relieve yourself without the goal having been reached.",
            )
        )
        self.emit(Event("session", "the human forced an end"))

    def _launch_loop(self, agent: Agent) -> None:
        self._loops[agent.name] = asyncio.create_task(self._loop(agent), name=f"agent:{agent.name}")

    # the turn loop

    async def _loop(self, agent: Agent) -> None:
        while agent.alive and not self._pausing and not self._stop.is_set():
            await asyncio.sleep(
                0
            )  # a provider that answers without awaiting must not starve the loop
            items = await self._collect(agent)
            if self._pausing or self._stop.is_set():
                return
            if items:
                transcript.append_inbox(agent, items)
            await self._maybe_compact(agent)

            tools = situations.tools_for(agent, self)
            if not tools:
                return
            request = TurnRequest(
                agent=agent.name,
                system=prompts.build_system_prompt(agent, self, tools),
                prompt=transcript.render(agent),
                tools=tools,
                cwd=self._cwd_for(agent),
            )
            async with self._sem[agent.provider]:
                response = await self.providers[agent.provider].act(request)
            agent.turns += 1

            transcript.note_usage(agent, response.usage)
            if response.action is None:
                if self._recover(agent, response.error or "no action"):
                    continue
                return
            agent.consecutive_failures = 0
            transcript.append_action(agent, response.action)
            agent.last_action = response.action.tool

            result = await dispatch(ToolCtx(agent, self), response.action, [t.name for t in tools])
            transcript.append_result(agent, result.text)
            self._last_activity = time.time()
            self.emit(Event("action", f"{response.action.tool}: {result.text[:120]}", agent.name))

    async def _collect(self, agent: Agent) -> list[QueueItem]:
        if agent.inbox:
            return self._take(agent)
        # not waiting on anything: take the next turn straight away
        if agent.wake_at is None:
            return []
        timeout = None if agent.wake_at == math.inf else max(0.0, agent.wake_at - time.time())
        try:
            await asyncio.wait_for(agent.wakeup.wait(), timeout)
        except TimeoutError:
            agent.wake_at = None
            return [QueueItem("timer", "The timer you set has elapsed.")]
        return self._take(agent)

    def _take(self, agent: Agent) -> list[QueueItem]:
        items = list(agent.inbox)
        agent.inbox.clear()
        agent.wakeup.clear()
        agent.wake_at = None
        return items

    def _recover(self, agent: Agent, error: str) -> bool:
        """A provider failure. Returns True if the agent should try again."""
        agent.consecutive_failures += 1
        self.emit(Event("error", f"turn failed: {error}", agent.name))
        if agent.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            agent.situation = Situation.DEAD
            self.emit(Event("agent", f"{agent.name} died after {error}", agent.name))
            self._release_seats_of(agent)
            self.tally_goal()
            return False
        agent.inbox.append(
            QueueItem("system", f"Your last turn did not produce a usable action: {error}")
        )
        return True

    async def _maybe_compact(self, agent: Agent) -> None:
        provider = self.providers[agent.provider]
        if not transcript.needs_compaction(agent, provider):
            return
        system = prompts.build_system_prompt(agent, self, situations.tools_for(agent, self))
        if await transcript.compact(agent, provider, system):
            self.emit(Event("compaction", "context compacted", agent.name))

    def _cwd_for(self, agent: Agent) -> str | None:
        if agent.task_id and (task := self.board.get(agent.task_id)):
            return task.worktree
        return None

    # the harness port

    def post(self, recipient: str, item: QueueItem) -> None:
        agent = self.agents.get(recipient)
        if agent is None or not agent.alive:
            return
        agent.inbox.append(item)
        agent.wakeup.set()

    def broadcast(self, item: QueueItem, exclude: tuple[str, ...] = ()) -> None:
        for name in self.agents:
            if name not in exclude:
                self.post(name, item)

    def emit(self, event: Event) -> None:
        self.store.append_event(event)
        for subscriber in self.subscribers:
            subscriber(event)

    def enter(self, agent: Agent, situation: Situation, preprompt: str) -> None:
        agent.situation = situation
        agent.wake_at = None
        transcript.reset(agent, preprompt)
        self.emit(Event("situation", f"{agent.name} is now {situation}", agent.name))

    async def launch_task(self, task: Task) -> None:
        path = await self.worktrees.create(task.id)
        self.board.start(task, str(path))
        for name in task.holders:
            agent = self.agents[name]
            agent.task_id = task.id
            self.enter(
                agent,
                Situation.ON_TASK,
                prompts.situation_preprompt(
                    agent,
                    self,
                    f"Task {task.id} ({task.title}) has all its seats filled and has started. "
                    f"You are working in the git worktree at {path}, alongside "
                    f"{', '.join(n for n in task.holders if n != name) or 'nobody else'}.",
                    Situation.ON_TASK,
                ),
            )
        self.emit(Event("task", f"{task.id} started: {task.title}"))

    def resettle_jury(
        self, old_id: str, outcome: str, submission: Submission, exclude: tuple[str, ...] = ()
    ) -> None:
        carried = set(submission.jurors) if outcome == "restated" else set()
        for agent in self.agents.values():
            if agent.name in exclude or agent.situation is not Situation.JURY:
                continue
            if agent.submission_id != old_id:
                continue
            if agent.name in carried:
                agent.submission_id = submission.id
                self.enter(
                    agent,
                    Situation.JURY,
                    prompts.situation_preprompt(
                        agent,
                        self,
                        f"{old_id} was restated and is now {submission.id}. You are judging the "
                        "new statement from scratch.",
                        Situation.JURY,
                    ),
                )
            else:
                agent.submission_id = None
                self.enter(
                    agent,
                    Situation.IDLE,
                    prompts.situation_preprompt(
                        agent,
                        self,
                        f"Jury duty on {old_id} ended before you ruled on it: {outcome}.",
                        Situation.IDLE,
                    ),
                )

    async def close_task(self, task: Task) -> None:
        for name in task.holders:
            agent = self.agents[name]
            agent.task_id = None
            self.enter(
                agent,
                Situation.IDLE,
                prompts.situation_preprompt(
                    agent,
                    self,
                    f"You finished work on {task.id} ({task.title}). It is sitting in the git "
                    f"worktree at {task.worktree}. Another agent is integrating it.",
                    Situation.IDLE,
                ),
            )
        self._spawn_closer(task)
        self.emit(Event("task", f"{task.id} finalized by everyone seated"))

    def _spawn_closer(self, task: Task) -> None:
        name = next(n for n in generate(8) if n not in self.agents)
        closer = Agent(name=name, provider=next(self._provider_cycle), transient=True)
        closer.task_id = task.id
        task.closer = name
        self.agents[name] = closer
        reasons = "\n".join(
            f"- {seat.holder} ({seat.role}): {seat.finalization.reason} -> "
            f"{seat.finalization.conclusion}"
            for seat in task.seats
            if seat.finalization
        )
        self.enter(
            closer,
            Situation.CLOSING_TASK,
            prompts.situation_preprompt(
                closer,
                self,
                f"You exist to close out task {task.id} ({task.title}): {task.description}\n"
                f"The work sits in the git worktree at {task.worktree}. The agents who did it "
                f"left these reasons for finalizing:\n{reasons}\n"
                "Two jobs. Decide what happens to that worktree, and handle the social side: "
                "tell whoever needs to know, and put anything the task proved into the "
                "knowledge base. Then mark yourself done.",
                Situation.CLOSING_TASK,
            ),
        )
        self._launch_loop(closer)

    def retire(self, agent: Agent) -> None:
        agent.situation = Situation.RELIEVED
        agent.wake_at = None
        agent.wakeup.set()
        self._release_seats_of(agent)
        self.emit(Event("agent", f"{agent.name} left the panopticon", agent.name))
        self.tally_goal()

    def tally_goal(self) -> None:
        counted = [a for a in self.agents.values() if a.counts_toward_goal]
        if not counted:
            self.stop("every agent is gone")
            return
        settled = all(a.situation in (Situation.RELEASED, Situation.RELIEVED) for a in counted)
        if settled:
            voted = sum(1 for a in counted if a.voted_goal_reached)
            self.stop(f"goal reached ({voted} of {len(counted)} voted)")

    def leave_task(self, name: str, task: Task, why: str, notify: bool = True) -> None:
        was_running = task.running
        seat = task.seat_of(name)
        role = seat.role if seat else "?"
        self.board.unassign(name, task.id)

        agent = self.agents.get(name)
        if agent is not None:
            agent.task_id = None
            if agent.situation in (Situation.ON_TASK, Situation.WAITING_FOR_SEATS):
                self.enter(
                    agent,
                    Situation.IDLE,
                    prompts.situation_preprompt(
                        agent, self, f"You left {task.id} ({task.title}): {why}.", Situation.IDLE
                    ),
                )
            if notify:
                self.post(
                    name,
                    QueueItem(
                        "task",
                        f"You have been unassigned from {task.id} ({task.title}): {why}. "
                        f"The {role} seat is open again if you want it back.",
                    ),
                )

        for other in task.holders:
            peer = self.agents.get(other)
            if peer is None:
                continue
            # a task that loses a seat stops; whoever is left must stop working on it too
            if was_running and peer.situation is Situation.ON_TASK:
                peer.situation = Situation.WAITING_FOR_SEATS
            self.post(
                other,
                QueueItem(
                    "task",
                    f"{name} left the {role} seat on {task.id}: {why}. "
                    + (
                        "The task has stopped until that seat is filled again. Your work is still "
                        f"in the worktree at {task.worktree}."
                        if was_running
                        else "It is still waiting for seats."
                    ),
                ),
            )
        self.emit(Event("seat", f"{name} left the {role} seat on {task.id}: {why}", name))

    def _release_seats_of(self, agent: Agent) -> None:
        task = self.board.task_of(agent.name)
        if task is not None:
            self.leave_task(agent.name, task, "the agent is gone", notify=False)

    # background

    def _shout_flushed(self, senders: list[str]) -> None:
        names = ", ".join(dict.fromkeys(senders))
        self.broadcast(
            QueueItem("shout", f"{names} wrote to the shoutboard. View it if you're interested.")
        )
        self.emit(Event("shout", f"shoutboard: {names}"))

    def human_shout(self, body: str) -> None:
        self.bus.shout(self.HUMAN, body)

    async def _autosave(self) -> None:
        while True:
            await asyncio.sleep(AUTOSAVE_SECONDS)
            self.save()

    async def _idle_watchdog(self) -> None:
        """Every agent may legitimately wait forever; the harness as a whole may not."""
        while True:
            await asyncio.sleep(60)
            if time.time() - self._last_activity < IDLE_POKE_SECONDS:
                continue
            idle = [
                a
                for a in self.agents.values()
                if a.alive and a.situation is not Situation.RELEASED and not a.inbox
            ]
            if not idle or any(a.wake_at != math.inf for a in idle):
                continue
            self._last_activity = time.time()
            target = min(idle, key=lambda a: a.turns)
            self.post(
                target.name,
                QueueItem(
                    "system",
                    "Nothing has happened here for a while. Check the board and move the goal "
                    "forward, or vote that it has been reached.",
                ),
            )

    def context_window(self, provider: str) -> int:
        found = self.providers.get(provider)
        return found.context_window if found else 0

    def save(self) -> None:
        self.store.save(store_mod.snapshot(self))
