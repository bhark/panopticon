"""Shared fixtures: the one temp git repo both the worktree and lifecycle tests need."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def make_git_repo(path: Path) -> Path:
    """An initialised repo on `main` with one seed commit, signing off."""
    path.mkdir(parents=True, exist_ok=True)
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "harness@panopticon.test"],
        ["git", "config", "user.name", "panopticon"],
        ["git", "config", "commit.gpgsign", "false"],
    ):
        subprocess.run(argv, cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed"], cwd=path, check=True, capture_output=True
    )
    return path


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return make_git_repo(tmp_path / "repo")
