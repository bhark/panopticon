"""panopticon start | resume | config | update"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from panopticon import __version__, names
from panopticon import config as config_mod
from panopticon import store as store_mod
from panopticon import update as update_mod
from panopticon.config import Config
from panopticon.model import Agent, Level, Situation
from panopticon.orchestrator import Orchestrator
from panopticon.providers.base import Provider
from panopticon.providers.registry import build_levels
from panopticon.services.worktrees import Worktrees
from panopticon.store import STATE_DIRNAME, Store

MIN_AGENTS = 3  # a truth needs two 'true' verdicts and cannot be judged by its submitter


def main(argv: list[str] | None = None) -> int:
    # child processes write to this same fd, so our own lines must not sit in a block buffer
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(prog="panopticon")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    start = sub.add_parser("start", help="open a panopticon in this directory")
    start.add_argument("--goal")
    start.add_argument("--agents", type=int)
    start.add_argument(
        "--mix", help="levels to open with, e.g. fast=3,balanced=2,capable=1; sets the head count"
    )
    start.add_argument("--headless", action="store_true")
    start.add_argument("--provider", action="append", help="restrict to these providers")

    resume = sub.add_parser("resume", help="reopen the panopticon saved here")
    resume.add_argument("--headless", action="store_true")

    cfg = sub.add_parser("config", help="inspect and change providers")
    cfg.add_argument("--list", action="store_true")
    cfg.add_argument("--connect-provider", metavar="NAME")
    cfg.add_argument("--disconnect-provider", metavar="NAME")

    sub.add_parser("update", help="install the latest panopticon")

    args = parser.parse_args(argv)
    match args.command:
        case "update":
            return _update()
        case "config":
            return _nudge(_config(args))
        case "resume":
            return _resume(args)
        case _:
            return _start(args)


def _nudge(code: int) -> int:
    """A command that never opens the interface says it in a line instead."""
    if note := update_mod.note():
        print(note, file=sys.stderr)
    return code


def _update() -> int:
    print(f"panopticon {__version__}, looking for a newer one...")
    try:
        tag = update_mod.latest()
        update_mod.remember(tag)
        if not update_mod.behind(tag):
            print("already the latest release")
            return 0
        print(f"installing {tag.removeprefix('v')}...")
        update_mod.install(tag)
    except update_mod.UpdateError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"panopticon {tag.removeprefix('v')} installed")
    return 0


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
        levels = entry.get("levels") or {}
        for level in Level:
            overlay = levels.get(str(level)) or {}
            settings = ", ".join(f"{k}={v}" for k, v in overlay.items()) or "as above"
            print(f"{'':9} {'':12} {level:12} {settings}")
    for reason in skipped:
        print(f"  ! {reason}")
    print(f"\nconfig: {config_mod.CONFIG_FILE}")
    return 0


def _mix(raw: str) -> dict[Level, int]:
    """fast=3,balanced=2,capable=1. Raises ValueError on anything else."""
    out: dict[Level, int] = {}
    for part in raw.split(","):
        name, _, count = part.partition("=")
        out[Level(name.strip())] = int(count)
    if any(n < 0 for n in out.values()) or not sum(out.values()):
        raise ValueError("a mix needs at least one agent")
    return out


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
    mix = None
    if args.mix:
        try:
            mix = _mix(args.mix)
        except ValueError as exc:
            levels = ", ".join(str(level) for level in Level)
            print(f"bad --mix: {exc}. Levels are {levels}.", file=sys.stderr)
            return 1
        count = sum(mix.values())
        if args.agents is not None and args.agents != count:
            print(
                f"--mix asks for {count} agents but --agents says {args.agents}.", file=sys.stderr
            )
            return 1
    else:
        count = args.agents or int(input(f"agents [{MIN_AGENTS}]: ").strip() or MIN_AGENTS)
    if count < MIN_AGENTS:
        print(f"at least {MIN_AGENTS} agents are needed for a jury.", file=sys.stderr)
        return 1

    assigned = config_mod.spread(count, sorted(usable), mix)
    agents = [
        Agent(name=name, provider=provider, level=level)
        for name, (provider, level) in zip(names.generate(count), assigned, strict=True)
    ]
    store = Store(cwd / STATE_DIRNAME)
    orch = Orchestrator(
        goal=goal,
        agents=agents,
        providers={k: build_levels(k, v) for k, v in usable.items()},
        store=store,
        worktrees=Worktrees(cwd, store.worktrees),
        config=cfg,
    )
    return _run(orch, args.headless)


def rebuild(
    store: Store, cfg: Config, providers: dict[str, dict[Level, Provider]], repo: Path
) -> Orchestrator:
    """Put a saved panopticon back together. Shared by `resume` and its tests."""
    state = store.load()
    agents = [store_mod.restore_agent(a) for a in state["agents"]]
    orch = Orchestrator(
        goal=state["goal"],
        agents=agents,
        providers=providers,
        store=store,
        worktrees=Worktrees(repo, store.worktrees),
        config=cfg,
        started_at=state["started_at"],
    )
    orch.force_ending = state.get("force_ending", False)
    orch.board.restore(state.get("board", []))
    orch.kb.restore(state.get("knowledge", {}))
    orch.bus.restore(state.get("bus", []))
    # whoever was mid-turn when we paused simply takes that turn again
    for agent in agents:
        if agent.situation is not Situation.DEAD:
            agent.wake_at = None
    return orch


def _resume(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    store = Store(cwd / STATE_DIRNAME)
    if not store.exists():
        print(f"no saved panopticon in {cwd / STATE_DIRNAME}", file=sys.stderr)
        return 1
    cfg = config_mod.load()
    wanted = sorted({a["provider"] for a in store.load()["agents"]})
    usable, code = _providers(cfg, wanted)
    if code:
        return code
    orch = rebuild(store, cfg, {k: build_levels(k, v) for k, v in usable.items()}, cwd)
    return _run(orch, args.headless)


def _run(orch: Orchestrator, headless: bool) -> int:
    asyncio.run(_headless(orch) if headless else _interactive(orch))
    print(orch.stopped_because or "closed")
    return _nudge(0) if headless else 0


async def _headless(orch: Orchestrator) -> None:
    check = asyncio.create_task(update_mod.refresh())
    try:
        await orch.run()
    finally:
        await _drop(check)


async def _drop(task: asyncio.Task) -> None:
    """The update check must never hold the exit open."""
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _interactive(orch: Orchestrator) -> None:
    from panopticon.tui.app import PanopticonApp

    app = PanopticonApp(orch)
    side: set[asyncio.Task] = set()

    def spawn(coro) -> None:
        task = asyncio.create_task(coro)
        side.add(task)
        task.add_done_callback(side.discard)

    async def pause_then_exit() -> None:
        await orch.pause()
        app.exit()

    app.on_shout = orch.human_shout
    app.human_name = orch.HUMAN
    app.on_force_end = orch.force_end
    app.on_pause = lambda: spawn(pause_then_exit())
    orch.subscribers.append(app.harness_event)

    async def check_for_update() -> None:
        """Beside the harness, never before it - the interface opens on the cached answer."""
        await update_mod.refresh()
        app.update_note = update_mod.note()
        app.paint.note()

    check = asyncio.create_task(check_for_update())
    runner = asyncio.create_task(orch.run())

    async def close_when_done() -> None:
        await asyncio.gather(runner, return_exceptions=True)
        app.exit()

    spawn(close_when_done())
    try:
        await app.run_async()
    finally:
        orch.stop("the human closed the interface")
        await _drop(check)
        await asyncio.gather(runner, *side, return_exceptions=True)
