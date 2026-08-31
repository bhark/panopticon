"""The shell and the file tools, rooted in whatever tree the agent is entitled to."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from typing import Any

from panopticon.model import ActionResult, ArgSpec, Situation, ToolCtx
from panopticon.situations import BASH, EDIT_FILE, READ_FILE, WRITE_FILE
from panopticon.tools import tool

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_OUTPUT = 16_000
MAX_LINES = 2_000


def _root(ctx: ToolCtx) -> Path:
    """The one tree this agent's file tools may touch."""
    agent = ctx.agent
    if agent.situation is Situation.ON_TASK and agent.task_id:
        return ctx.harness.worktrees.path_for(agent.task_id).resolve()
    # jury reads the main repo; the closer works across it, worktrees included
    return ctx.harness.worktrees.repo.resolve()


def _resolve(root: Path, raw: str) -> Path:
    """Resolve first, then check containment, so a symlink pointing out lands outside and fails."""
    wanted = Path(raw.strip()).expanduser()
    target = (wanted if wanted.is_absolute() else root / wanted).resolve()
    if target != root and root not in target.parents:
        raise ValueError(
            f"{raw} resolves to {target}, outside {root}. That tree is the only one these tools "
            f"will touch."
        )
    return target


def _shown(root: Path, target: Path) -> str:
    return str(target.relative_to(root)) if target != root else "."


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    cut = len(text) - limit
    return f"{text[:head]}\n\n[... {cut} characters cut from the middle ...]\n\n{text[-tail:]}"


@tool(
    BASH,
    "Run a shell command in your working tree. You get the exit code and stdout and stderr "
    "combined, truncated in the middle if it is long, so pipe through grep or head rather than "
    "dumping whole files. On jury duty this is for gathering evidence only: look at what is "
    "there, do not run tests, do not build, do not change anything.",
    command=ArgSpec("string", "The command line, run through the shell."),
    timeout=ArgSpec(
        "int",
        f"Seconds before the command is killed. Default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT}.",
        required=False,
    ),
)
async def bash(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    command = args["command"].strip()
    if not command:
        return ActionResult.fail("command must not be empty.")
    timeout = args.get("timeout") or DEFAULT_TIMEOUT
    if not 1 <= timeout <= MAX_TIMEOUT:
        return ActionResult.fail(f"timeout must be between 1 and {MAX_TIMEOUT} seconds.")

    root = _root(ctx)
    # a guardrail against accident, not a sandbox: the shell can still reach outside by absolute path
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(root),
        start_new_session=True,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # start_new_session made it the group leader
        await proc.wait()
        return ActionResult.fail(
            f"Command was killed after {timeout}s. Nothing it printed was captured. Run "
            f"something shorter, or pass a larger timeout if it genuinely needs one."
        )

    body = _truncate(out.decode(errors="replace")).rstrip()
    return ActionResult(
        proc.returncode == 0, f"exit {proc.returncode}\n{body or '(no output)'}"
    )


@tool(
    READ_FILE,
    "Read a file in your working tree, with line numbers. Long files come back truncated, so "
    "use offset and limit to walk a big one rather than pulling it all in.",
    path=ArgSpec("string", "Path to the file, relative to your working tree."),
    offset=ArgSpec("int", "First line to show, 1-based. Default 1.", required=False),
    limit=ArgSpec("int", f"How many lines to show. Default {MAX_LINES}.", required=False),
)
async def read_file(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    root = _root(ctx)
    try:
        target = _resolve(root, args["path"])
    except ValueError as exc:
        return ActionResult.fail(str(exc))
    if not target.is_file():
        return ActionResult.fail(f"{_shown(root, target)} is not a file.")

    offset = max(1, args.get("offset") or 1)
    limit = args.get("limit") or MAX_LINES
    if limit < 1:
        return ActionResult.fail("limit must be 1 or more.")
    limit = min(limit, MAX_LINES)

    try:
        lines = target.read_text(errors="replace").splitlines()
    except OSError as exc:
        return ActionResult.fail(f"Could not read {_shown(root, target)}: {exc}")
    if offset > len(lines):
        return ActionResult.fail(
            f"{_shown(root, target)} has {len(lines)} lines; offset {offset} is past the end."
        )

    window = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{offset + i:>6}\t{line}" for i, line in enumerate(window))
    last = offset + len(window) - 1
    head = f"{_shown(root, target)} lines {offset}-{last} of {len(lines)}"
    return ActionResult(True, f"{head}\n{_truncate(numbered)}")


@tool(
    WRITE_FILE,
    "Write a file in your working tree, creating it and any missing directories, replacing it "
    "whole if it is already there. Read it first if it exists; you are overwriting work that "
    "may not be yours.",
    path=ArgSpec("string", "Path to the file, relative to your working tree."),
    content=ArgSpec("string", "The full new contents of the file."),
)
async def write_file(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    root = _root(ctx)
    try:
        target = _resolve(root, args["path"])
    except ValueError as exc:
        return ActionResult.fail(str(exc))
    if target.is_dir():
        return ActionResult.fail(f"{_shown(root, target)} is a directory.")

    existed = target.exists()
    content = args["content"]
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    except OSError as exc:
        return ActionResult.fail(f"Could not write {_shown(root, target)}: {exc}")
    what = "Overwrote" if existed else "Created"
    return ActionResult(
        True, f"{what} {_shown(root, target)}, {len(content.splitlines())} lines."
    )


@tool(
    EDIT_FILE,
    "Replace one exact stretch of text in a file. `old` has to match the file byte for byte, "
    "whitespace included, and has to appear exactly once - if it appears twice the edit is "
    "refused rather than guessed at, so include the surrounding lines that make it unique.",
    path=ArgSpec("string", "Path to the file, relative to your working tree."),
    old=ArgSpec("string", "The exact text to replace, unique within the file."),
    new=ArgSpec("string", "What to put in its place."),
)
async def edit_file(ctx: ToolCtx, args: dict[str, Any]) -> ActionResult:
    root = _root(ctx)
    try:
        target = _resolve(root, args["path"])
    except ValueError as exc:
        return ActionResult.fail(str(exc))
    if not target.is_file():
        return ActionResult.fail(f"{_shown(root, target)} is not a file.")

    old, new = args["old"], args["new"]
    if not old:
        return ActionResult.fail("old must not be empty; there is nothing to match on.")
    if old == new:
        return ActionResult.fail("old and new are identical, so this edit would change nothing.")

    try:
        content = target.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        return ActionResult.fail(f"Could not read {_shown(root, target)}: {exc}")

    found = content.count(old)
    shown = _shown(root, target)
    if found == 0:
        return ActionResult.fail(
            f"That text does not appear in {shown}. It has to match exactly, whitespace and all. "
            f"Read the file and copy the lines you mean."
        )
    if found > 1:
        return ActionResult.fail(
            f"That text appears {found} times in {shown}, so the edit is ambiguous. Include more "
            f"of the surrounding lines until it matches exactly one place."
        )

    try:
        target.write_text(content.replace(old, new, 1))
    except OSError as exc:
        return ActionResult.fail(f"Could not write {shown}: {exc}")
    return ActionResult(True, f"Edited {shown}.")
