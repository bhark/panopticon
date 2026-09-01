"""The way in: what the flags mean, and what is left for the interface to ask."""

from __future__ import annotations

import pytest

from panopticon import cli
from panopticon import config as config_mod
from panopticon.model import Level


class FakeArgs:
    def __init__(self, **kw):
        self.__dict__.update({"goal": None, "agents": None, "mix": None} | kw)


class TestParsing:
    def test_bare_panopticon_opens_a_new_one(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cli, "_start", lambda args: seen.append(args) or 0)
        assert cli.main([]) == 0
        assert seen[0].command is None
        assert seen[0].goal is None and seen[0].headless is False

    def test_headless_is_the_same_flag_on_either_side_of_resume(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cli, "_resume", lambda args: seen.append(args.headless) or 0)
        cli.main(["resume", "--headless"])
        cli.main(["--headless", "resume"])
        cli.main(["resume"])
        assert seen == [True, True, False]


class TestRoster:
    def test_the_default_roster_is_a_jury_of_three(self):
        assert cli._wanted(FakeArgs()) == {Level.CAPABLE: 1, Level.BALANCED: 1, Level.FAST: 1}

    def test_a_mix_is_taken_as_written(self):
        wanted = cli._wanted(FakeArgs(mix="fast=3,capable=1"))
        assert wanted == {Level.CAPABLE: 1, Level.BALANCED: 0, Level.FAST: 3}

    def test_a_mix_that_fights_the_head_count_is_refused(self):
        with pytest.raises(ValueError, match="asks for 4"):
            cli._wanted(FakeArgs(mix="fast=3,capable=1", agents=5))

    def test_a_roster_too_small_for_a_jury_is_refused(self):
        with pytest.raises(ValueError, match="jury"):
            cli._wanted(FakeArgs(agents=2))

    def test_an_unknown_level_says_which_ones_exist(self):
        with pytest.raises(ValueError, match="Levels are fast, balanced, capable"):
            cli._wanted(FakeArgs(mix="quick=3"))


def test_headless_without_a_goal_says_so(capsys, git_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "none.json")
    monkeypatch.chdir(git_repo)
    assert cli.main(["--headless"]) == 1
    assert "needs a --goal" in capsys.readouterr().err
