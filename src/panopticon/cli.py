"""panopticon start | resume | config"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from panopticon import config as config_mod
from panopticon import names
from panopticon import store as store_mod
from panopticon.config import Config
from panopticon.model import Agent, Situation
from panopticon.orchestrator import Orchestrator
from panopticon.providers.registry import build
from panopticon.services.worktrees import Worktrees
from panopticon.store import STATE_DIRNAME, Store

MIN_AGENTS = 3  # a truth needs two 'true' verdicts and cannot be judged by its submitter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="panopticon")
    sub = parser.add_subparsers(dest="command")

    start = sub.add_parser("start", help="open a panopticon in this directory")
    start.add_argument("--goal")
    start.add_argument("--agents", type=int)
    start.add_argument("--headless", action="store_true")
    start.add_argument("--provider", action="append", help="restrict to these providers")

    resume = sub.add_parser("resume", help="reopen the panopticon saved here")
    resume.add_argument("--headless", action="store_true")

    cfg = sub.add_parser("config", help="inspect and change providers")
    cfg.add_argument("--list", action="store_true")
    cfg.add_argument("--connect-provider", metavar="NAME")
    cfg.add_argument("--disconnect-provider", metavar="NAME")

    args = parser.parse_args(argv)
    match args.command:
        case "config":
            return _config(args)
        case "resume":
            return _resume(args)
        case _:
            return _start(args)


def _config(args: argparse.Namespace) -> int:
    cfg = config_mod.load()
    if name := args.connect_provider or args.disconnect_provider:
        if name not in cfg.providers:
            print(f"no such provider: {name}. Known: {', '.join(cfg.providers)}", file=sys.stderr)
            return 1
        cfg.providers[name]["enabled"] = bool(args.connect_provider)
        config_mod.save(cfg)
        print(f"{name} {'connected' if args.connect_provider else 'disconnected'}")
        return 0
    usable, skipped = cfg.usable()
    for name, entry in cfg.providers.items():
        state = "on" if name in usable else ("off" if not entry.get("enabled") else "unusable")
        print(f"{state:9} {name:12} {entry['kind']:12} {entry.get('model', '')}")
    for reason in skipped:
        print(f"  ! {reason}")
    print(f"\nconfig: {config_mod.CONFIG_FILE}")
    return 0


def _providers(cfg: Config, only: list[str] | None) -> tuple[dict, int]:
    usable, skipped = cfg.usable()
    if only:
        usable = {k: v for k, v in usable.items() if k in only}
    if not usable:
        print("no usable providers.", file=sys.stderr)
        for reason in skipped:
            print(f"  {reason}", file=sys.stderr)
        print("try: panopticon config --list", file=sys.stderr)
        return {}, 1
    return usable, 0


def _start(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    if not Worktrees.is_git_repo(cwd):
        print(f"{cwd} is not a git repository; tasks need worktrees.", file=sys.stderr)
        return 1

    cfg = config_mod.load()
    usable, code = _providers(cfg, args.provider)
    if code:
        return code

    goal = args.goal or input("goal: ").strip()
    if not goal:
        print("a panopticon needs a goal.", file=sys.stderr)
        return 1
    count = args.agents or int(input(f"agents [{MIN_AGENTS}]: ").strip() or MIN_AGENTS)
    if count < MIN_AGENTS:
        print(f"at least {MIN_AGENTS} agents are needed for a jury.", file=sys.stderr)
        return 1

    assigned = config_mod.spread(count, sorted(usable))
    agents = [
        Agent(name=name, provider=provider)
        for name, provider in zip(names.generate(count), assigned, strict=True)
    ]
    store = Store(cwd / STATE_DIRNAME)
    orch = Orchestrator(
        goal=goal,
        agents=agents,
        providers={k: build(k, v) for k, v in usable.items()},
        store=store,
        worktrees=Worktrees(cwd, store.worktrees),
        config=cfg,
    )
    return _run(orch, args.headless)


def _resume(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    store = Store(cwd / STATE_DIRNAME)
    if not store.exists():
        print(f"no saved panopticon in {cwd / STATE_DIRNAME}", file=sys.stderr)
        return 1
    state = store.load()
    cfg = config_mod.load()
    agents = [store_mod.restore_agent(a) for a in state["agents"]]
    usable, code = _providers(cfg, sorted({a.provider for a in agents}))
    if code:
        return code

    orch = Orchestrator(
        goal=state["goal"],
        agents=agents,
        providers={k: build(k, v) for k, v in usable.items()},
        store=store,
        worktrees=Worktrees(cwd, store.worktrees),
        config=cfg,
        started_at=state["started_at"],
    )
    orch.force_ending = state.get("force_ending", False)
    for raw in state.get("tasks", []):
        task = store_mod.restore_task(raw)
        orch.board.tasks[task.id] = task
    orch.kb.truths = [store_mod.restore_truth(t) for t in state.get("truths", [])]
    for raw in state.get("pending_submissions", []):
        submission = store_mod.restore_submission(raw)
        orch.kb.pending[submission.id] = submission
    orch.bus.shouts = [store_mod.restore_shout(s) for s in state.get("shouts", [])]

    # an agent that was mid-turn when we paused resumes by taking that turn again
    for agent in agents:
        if agent.situation is Situation.DEAD:
            continue
        agent.wake_at = None
    return _run(orch, args.headless)


def _run(orch: Orchestrator, headless: bool) -> int:
    asyncio.run(_headless(orch) if headless else _interactive(orch))
    print(orch.stopped_because or "closed")
    return 0


async def _headless(orch: Orchestrator) -> None:
    await orch.run()


async def _interactive(orch: Orchestrator) -> None:
    from panopticon.tui.app import PanopticonApp

    app = PanopticonApp(orch)
    app.on_shout = orch.human_shout
    app.on_pause = lambda: asyncio.create_task(orch.pause())
    app.on_force_end = orch.force_end
    orch.subscribers.append(app.on_event)
    runner = asyncio.create_task(orch.run())
    try:
        await app.run_async()
    finally:
        orch.stop("the human closed the interface")
        await asyncio.gather(runner, return_exceptions=True)
