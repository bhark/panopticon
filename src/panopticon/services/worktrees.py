"""One git worktree per task, under .panopticon/worktrees/."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path


class Worktrees:
    def __init__(self, repo: Path, root: Path) -> None:
        self.repo = repo
        self.root = root

    @staticmethod
    def is_git_repo(path: Path) -> bool:
        done = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--git-dir"],
            capture_output=True,
            check=False,
        )
        return done.returncode == 0

    async def create(self, task_id: str) -> Path:
        path = self.path_for(task_id)
        if path.exists():
            return path
        self.root.mkdir(parents=True, exist_ok=True)
        code, output = await self._git(
            "worktree", "add", "-b", f"panopticon/{task_id}", str(path)
        )
        if code != 0:
            raise RuntimeError(f"could not cut a worktree for {task_id}: {output}")
        return path

    async def remove(self, task_id: str) -> None:
        path = self.path_for(task_id)
        code, _ = await self._git("worktree", "remove", "--force", str(path))
        if code != 0:
            # the closer may have left it in any state; cleanup failing must not stop the harness
            shutil.rmtree(path, ignore_errors=True)
            await self._git("worktree", "prune")

    def path_for(self, task_id: str) -> Path:
        return self.root / task_id

    async def _git(self, *args: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(self.repo),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await proc.communicate()
        return proc.returncode or 0, output.decode(errors="replace").strip()
