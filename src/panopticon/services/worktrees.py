"""One git worktree per task, under .panopticon/worktrees/."""

from __future__ import annotations

from pathlib import Path


class Worktrees:
    def __init__(self, repo: Path, root: Path) -> None:
        self.repo = repo
        self.root = root

    @staticmethod
    def is_git_repo(path: Path) -> bool: ...

    async def create(self, task_id: str) -> Path: ...
    async def remove(self, task_id: str) -> None: ...
    def path_for(self, task_id: str) -> Path: ...
