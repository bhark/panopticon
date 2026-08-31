"""Worktrees against a real temporary repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from panopticon.services.worktrees import Worktrees


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def worktrees(git_repo: Path) -> Worktrees:
    return Worktrees(git_repo, git_repo / ".panopticon" / "worktrees")


async def test_create_gives_the_task_its_own_branch_and_checkout(worktrees: Worktrees):
    path = await worktrees.create("t1-fix-parser")

    assert path == worktrees.path_for("t1-fix-parser")
    assert (path / "README.md").read_text() == "seed\n"
    assert git(path, "rev-parse", "--abbrev-ref", "HEAD") == "panopticon/t1-fix-parser"

    assert await worktrees.create("t1-fix-parser") == path  # already cut, not an error


async def test_a_dirty_worktree_is_still_removed(worktrees: Worktrees):
    path = await worktrees.create("t2-messy")
    (path / "README.md").write_text("half-finished work\n")
    (path / "scratch.txt").write_text("untracked\n")

    await worktrees.remove("t2-messy")

    assert not path.exists()
    assert "t2-messy" not in git(worktrees.repo, "worktree", "list")


async def test_removing_something_that_is_not_there_stays_quiet(worktrees: Worktrees):
    await worktrees.remove("t3-never-existed")


def test_is_git_repo_guards_the_working_directory(worktrees: Worktrees, tmp_path: Path):
    assert Worktrees.is_git_repo(worktrees.repo) is True
    assert Worktrees.is_git_repo(tmp_path / "not-a-repo") is False
