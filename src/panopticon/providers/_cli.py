"""Subprocess plumbing shared by the three CLI adapters.

The CLIs spawn children of their own, so every run gets its own process group and a
timeout kills the group, not just the wrapper. Nothing here raises: a failure to spawn
reads the same as a non-zero exit.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Iterator
from dataclasses import dataclass

DEFAULT_TIMEOUT = 180.0


@dataclass(slots=True)
class Completed:
    stdout: str
    stderr: str
    code: int
    error: str | None = None


async def run(argv: list[str], *, cwd: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> Completed:
    """stdin is closed: codex appends a piped stdin to the prompt as a <stdin> block."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        return Completed("", "", -1, f"could not start {argv[0]}: {exc}")

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        _kill_group(proc)
        await proc.wait()
        return Completed("", "", -1, f"{argv[0]} timed out after {timeout:.0f}s")
    except asyncio.CancelledError:
        _kill_group(proc)
        raise

    return Completed(out.decode(errors="replace"), err.decode(errors="replace"), proc.returncode or 0)


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def ndjson(text: str) -> Iterator[dict]:
    """Every CLI here streams NDJSON with the occasional non-JSON line mixed in."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def tail(text: str, limit: int = 400) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text
