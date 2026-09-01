"""panopticon | resume | config | update"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

from panopticon import __version__, names
from panopticon import config as config_mod
from panopticon import store as store_mod
from panopticon import update as update_mod
from panopticon.config import MIN_AGENTS, Config
from panopticon.model import Agent, Level, Situation
from panopticon.orchestrator import Orchestrator
from panopticon.providers.base import Provider
from panopticon.providers.registry import build_levels
from panopticon.services.worktrees import Worktrees
from panopticon.store import STATE_DIRNAME, Store


def main(argv: list[str] | None = None) -> int:
    # child processes write to this same fd, so our own lines must not sit in a block buffer
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="panopticon", description="open a panopticon in this directory"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--goal")
    parser.add_argument("--agents", type=int)
    parser.add_argument(
        "--mix", help="levels to open with, e.g. fast=3,balanced=2,capable=1; sets the head count"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--provider", action="append", help="restrict to these providers")
    sub = parser.add_subparsers(dest="command")

    resume = sub.add_parser("resume", help="reopen the panopticon saved here")
    # the same flag as above, so an absent one here must not overwrite what came before the command
    resume.add_argument("--headless", action="store_true", default=argparse.SUPPRESS)

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
    try:
        for part in raw.split(","):
            name, _, count = part.partition("=")
            out[Level(name.strip())] = int(count)
    except ValueError:
        levels = ", ".join(str(level) for level in Level)
        raise ValueError(f"bad --mix: {raw!r}. Levels are {levels}.") from None
    if any(n < 0 for n in out.values()) or not sum(out.values()):
        raise ValueError("a mix needs at least one agent")
    return out


def _wanted(args: argparse.Namespace) -> dict[Level, int]:
    """The roster the flags asked for; the interface asks for it when they did not."""
    mix = _mix(args.mix) if args.mix else None
    count = sum(mix.values()) if mix else (args.agents or MIN_AGENTS)
    if mix and args.agents is not None and args.agents != count:
        raise ValueError(f"--mix asks for {count} agents but --agents says {args.agents}.")
    if (args.agents or args.mix) and count < MIN_AGENTS:
        raise ValueError(f"at least {MIN_AGENTS} agents are needed for a jury.")
    return config_mod.roster(count, mix)


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
    try:
        wanted = _wanted(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    store = Store(cwd / STATE_DIRNAME)
    providers = {k: build_levels(k, v) for k, v in usable.items()}

    def build(goal: str, roster: dict[Level, int]) -> Orchestrator:
        assigned = config_mod.spread(sum(roster.values()), sorted(usable), roster)
        agents = [
            Agent(name=name, provider=provider, level=level)
            for name, (provider, level) in zip(names.generate(len(assigned)), assigned, strict=True)
        ]
        return Orchestrator(
            goal=goal,
            agents=agents,
            providers=providers,
            store=store,
            worktrees=Worktrees(cwd, store.worktrees),
            config=cfg,
        )

    def saved() -> Orchestrator:
        return rebuild(store, cfg, providers, cwd)

    if args.headless:
        if not args.goal:
            print("--headless needs a --goal; there is no interface to ask in.", file=sys.stderr)
            return 1
        return _run(build(args.goal, wanted), headless=True)
    if args.goal:
        return _run(build(args.goal, wanted))

    return _ask(build, saved if store.exists() else None, wanted, sorted(usable))


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
    return _run(orch, headless=getattr(args, "headless", False))


def _run(orch: Orchestrator, headless: bool = False) -> int:
    if headless:
        asyncio.run(_headless(orch))
        print(orch.stopped_because or "closed")
        return _nudge(0)
    return _closed(asyncio.run(_interactive(orch)))


def _ask(
    build: Callable[[str, dict[Level, int]], Orchestrator],
    saved: Callable[[], Orchestrator] | None,
    roster: dict[Level, int],
    providers: list[str],
) -> int:
    """Nothing is assembled yet: the interface asks for what the flags did not say."""
    return _closed(
        asyncio.run(_interactive(build=build, saved=saved, roster=roster, providers=providers))
    )


def _closed(opened: Orchestrator | None) -> int:
    reason = opened.stopped_because if opened else ""
    print(reason or "closed")
    return 0


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


async def _interactive(
    ready: Orchestrator | None = None,
    *,
    build: Callable[[str, dict[Level, int]], Orchestrator] | None = None,
    saved: Callable[[], Orchestrator] | None = None,
    roster: dict[Level, int] | None = None,
    providers: list[str] | None = None,
) -> Orchestrator | None:
    from panopticon.tui.app import PanopticonApp
    from panopticon.tui.launch import Launch

    side: set[asyncio.Task] = set()
    live: list[Orchestrator] = []

    def spawn(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        side.add(task)
        task.add_done_callback(side.discard)
        return task

    async def pause_then_exit(orch: Orchestrator) -> None:
        await orch.pause()
        app.exit()

    async def close_when_done(runner: asyncio.Task) -> None:
        await asyncio.gather(runner, return_exceptions=True)
        app.exit()

    def adopt(orch: Orchestrator) -> Orchestrator:
        """Everything that only makes sense once there is something to watch."""
        live.append(orch)
        app.on_shout = orch.human_shout
        app.human_name = orch.HUMAN
        app.on_force_end = orch.force_end
        app.on_pause = lambda: spawn(pause_then_exit(orch))
        orch.subscribers.append(app.harness_event)
        runner = spawn(orch.run())
        spawn(close_when_done(runner))
        return orch

    launch = None
    if build is not None:
        launch = Launch(
            roster=roster or {},
            providers=providers or [],
            open=lambda goal, picked: adopt(build(goal, picked)),
            resume=(lambda: adopt(saved())) if saved else None,
        )
    app = PanopticonApp(launch=launch)
    if ready is not None:
        app.attach(adopt(ready))

    async def check_for_update() -> None:
        """Beside the harness, never before it - the interface opens on the cached answer."""
        await update_mod.refresh()
        app.update_note = update_mod.note()
        app.paint.note()

    check = asyncio.create_task(check_for_update())
    try:
        await app.run_async()
    finally:
        for orch in live:
            orch.stop("the human closed the interface")
        await _drop(check)
        await asyncio.gather(*side, return_exceptions=True)
    return live[-1] if live else None
