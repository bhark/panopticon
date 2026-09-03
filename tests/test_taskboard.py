"""Seat lifecycle, finalization, and the janitor that polices seats."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from panopticon.model import Situation
from panopticon.services.janitor import Janitor
from panopticon.services.taskboard import TaskBoard
from tests.fakes import FakeHarness


def board_with_task(roles: list[str]) -> tuple[TaskBoard, str]:
    board = TaskBoard()
    task = board.create("ada", "Fix the parser", "empty input crashes", roles)
    return board, task.id


def test_task_starts_exactly_when_the_last_seat_fills():
    board, task_id = board_with_task(["impl", "review"])

    task, ready = board.assign("ada", task_id, "impl")
    assert ready is False
    assert task.ready is False

    task, ready = board.assign("bo", task_id, "review")
    assert ready is True
    assert task.holders == ["ada", "bo"]


def test_single_seat_task_is_ready_on_the_first_assignment():
    board, task_id = board_with_task(["impl"])
    _, ready = board.assign("ada", task_id, "impl")
    assert ready is True


def test_a_filled_seat_cannot_be_taken_twice():
    board, task_id = board_with_task(["impl", "review"])
    board.assign("ada", task_id, "impl")

    with pytest.raises(ValueError):
        board.assign("bo", task_id, "impl")
    with pytest.raises(ValueError):
        board.assign("ada", task_id, "review")  # already holds a seat here
    with pytest.raises(ValueError):
        board.assign("bo", task_id, "designer")  # no such role


def test_unassign_reopens_the_seat_and_stops_the_run():
    board, task_id = board_with_task(["impl", "review"])
    board.assign("ada", task_id, "impl")
    task, _ = board.assign("bo", task_id, "review")
    board.start(task, "wt")
    board.finalize("bo", task_id, "done", "shipped")

    task = board.unassign("ada", task_id)
    assert task.running is False
    assert task.ready is False
    assert board.task_of("ada") is None
    assert task.seat_of("bo").finalization is not None  # the ones who stayed keep their vote

    _, ready = board.assign("cy", task_id, "impl")
    assert ready is True


def test_finalization_needs_every_seat():
    board, task_id = board_with_task(["impl", "review"])
    board.assign("ada", task_id, "impl")
    task, _ = board.assign("bo", task_id, "review")
    board.start(task, "wt")

    _, agreed = board.finalize("ada", task_id, "tests pass", "parser fixed")
    assert agreed is False
    _, agreed = board.finalize("bo", task_id, "reviewed", "looks right")
    assert agreed is True


def test_cancel_finalize_unblocks_a_task_one_vote_from_done():
    board, task_id = board_with_task(["impl", "review"])
    board.assign("ada", task_id, "impl")
    task, _ = board.assign("bo", task_id, "review")
    board.start(task, "wt")
    board.finalize("ada", task_id, "tests pass", "parser fixed")
    board.cancel_finalize("ada", task_id)

    _, agreed = board.finalize("bo", task_id, "reviewed", "looks right")
    assert agreed is False

    with pytest.raises(ValueError):
        board.cancel_finalize("ada", task_id)  # nothing to cancel any more


def test_finalizing_needs_a_seat_on_a_running_task():
    board, task_id = board_with_task(["impl", "review"])
    board.assign("ada", task_id, "impl")

    with pytest.raises(ValueError):
        board.finalize("ada", task_id, "r", "c")  # not started, one seat still open
    with pytest.raises(ValueError):
        board.finalize("zed", task_id, "r", "c")  # not seated


def test_ids_stay_short_readable_and_unique():
    board = TaskBoard()
    first = board.create("ada", "Fix the parser", "", ["impl"])
    second = board.create("bo", "Fix the parser", "", ["impl"])

    assert first.id.startswith("t1-fix-the-parser")
    assert second.id != first.id
    assert len(second.id) < 32


def test_render_shows_open_seats_and_caps_the_archive():
    board = TaskBoard()
    for n in range(TaskBoard.ARCHIVE_SHOWN + 5):
        task = board.create("ada", f"Old job {n}", "", ["impl"])
        board.archive_task(task, "merged")
    task = board.create("ada", "Fix the parser", "empty input crashes", ["impl", "review"])
    board.assign("bo", task.id, "impl")

    view = board.render()
    assert "impl=bo" in view
    assert "review=OPEN" in view
    assert view.count("Old job") == TaskBoard.ARCHIVE_SHOWN
    assert "empty input crashes" in view


# janitor


def waiting_seat(tmp_path: Path, minutes: float) -> tuple[FakeHarness, str, float]:
    board, task_id = board_with_task(["impl", "review"])
    harness = FakeHarness(tmp_path / "repo", names=("ada", "bo"), board=board)
    harness.agents["ada"].situation = Situation.WAITING_FOR_SEATS
    board.assign("ada", task_id, "impl")
    now = time.time()
    board.get(task_id).seat_of("ada").assigned_at = now - minutes * 60
    return harness, task_id, now


def test_nudges_fire_once_per_threshold_then_the_seat_expires(tmp_path):
    harness, task_id, now = waiting_seat(tmp_path, minutes=15)
    janitor = Janitor(harness)

    janitor.tick(now)
    janitor.tick(now + 120)  # still inside the same window
    assert len(harness.posted) == 1
    assert "15 minutes" in harness.posted[0][1].text

    janitor.tick(now + 16 * 60)  # 31 minutes seated
    janitor.tick(now + 31 * 60)  # 46 minutes seated
    assert len(harness.posted) == 3
    assert harness.board.get(task_id).seat_of("ada").nudges_sent == 3

    janitor.tick(now + 46 * 60)  # an hour seated
    assert harness.board.get(task_id).seats[0].holder is None
    assert "unassigned" in harness.posted[-1][1].text
    assert harness.events


def test_a_running_task_is_checked_in_on_rather_than_nudged(tmp_path):
    harness, task_id, now = waiting_seat(tmp_path, minutes=50)
    task = harness.board.get(task_id)
    harness.board.assign("bo", task_id, "review")
    harness.board.start(task, "wt")

    Janitor(harness).tick(now + 61 * 60)

    assert task.holders == ["ada", "bo"]
    assert all("open seats" not in item.text for _, item in harness.posted)
    assert all("Check-in" in item.text for _, item in harness.posted)


def test_a_dead_agents_seat_is_released_without_nudging_it(tmp_path):
    harness, task_id, now = waiting_seat(tmp_path, minutes=1)
    harness.agents["ada"].situation = Situation.DEAD

    Janitor(harness).tick(now)

    assert harness.board.get(task_id).seats[0].holder is None
    assert harness.posted == []
    assert harness.events[0].agent == "ada"


def test_the_board_catches_the_same_task_filed_twice_but_not_a_neighbouring_one():
    """Agents acting at the same instant all file the task nobody has filed yet: in a live
    run three agents opened three tasks for one job before any board event reached them."""
    board = TaskBoard()
    board.create("ada", "Add subtract(a, b) to calc.py and test it", "d", ["impl"])

    assert board.duplicate_of("Add and test subtract") is not None
    assert board.duplicate_of("Add subtract(a,b) to calc") is not None
    # a different change to the same file is not the same task
    assert board.duplicate_of("Add multiply to calc.py") is None
    assert board.duplicate_of("Fix the parser crash") is None
    assert board.duplicate_of("") is None

    # once it is out of the way it stops blocking
    task = board.get(next(iter(board.tasks)))
    board.archive_task(task, "done")
    assert board.duplicate_of("Add and test subtract") is None


def test_leaving_a_long_running_task_restarts_the_wait_for_the_others():
    board, task_id = board_with_task(["impl", "review"])
    task = board.get(task_id)
    board.assign("ada", task_id, "impl")
    board.assign("bo", task_id, "review")
    board.start(task, "wt")
    for seat in task.seats:
        seat.assigned_at -= 3 * 3600

    board.unassign("ada", task_id)

    # bo spent those three hours working, not waiting; its hour starts from the stop
    assert [minutes for _, _, minutes in board.waiting_seats(time.time())] == [0]


def test_the_janitor_breaks_a_standoff_where_every_agent_is_waiting(tmp_path):
    board = TaskBoard()
    harness = FakeHarness(tmp_path / "repo", names=("ada", "bo"), board=board)
    now = time.time()
    for name, waited in (("ada", 5), ("bo", 12)):
        task = board.create(name, f"job for {name}", "d", ["impl", "review"])
        board.assign(name, task.id, "impl")
        task.seat_of(name).assigned_at = now - waited * 60
        harness.agents[name].situation = Situation.WAITING_FOR_SEATS

    Janitor(harness).tick(now)

    assert harness.agents["bo"].situation is Situation.IDLE  # the longest wait gives way
    assert harness.agents["ada"].situation is Situation.WAITING_FOR_SEATS


def test_a_task_being_closed_out_is_left_alone():
    board, task_id = board_with_task(["impl"])
    task = board.get(task_id)
    board.assign("ada", task_id, "impl")
    board.start(task, "wt")
    task.closer = "zed"

    assert board.task_of("ada") is None  # the seat is history, not a live holding
    assert board.held_seats() == []
    assert "being closed out by zed" in board.render()


def test_a_gone_closer_is_replaced_once_then_the_task_is_archived(tmp_path):
    board, task_id = board_with_task(["impl"])
    harness = FakeHarness(tmp_path / "repo", names=("ada",), board=board)
    task = board.get(task_id)
    board.assign("ada", task_id, "impl")
    board.start(task, "wt")
    harness.spawn_closer(task)
    janitor = Janitor(harness)

    harness.agents[task.closer].situation = Situation.DEAD
    janitor.tick()
    assert harness.closers == ["closer1", "closer2"]
    assert task.archived_at is None

    harness.agents[task.closer].situation = Situation.DEAD
    janitor.tick()
    assert task.archived_at is not None
    assert "wt" in (task.outcome or "")
    assert any("worktree" in item.text for item in harness.broadcasts)


def running_seat(tmp_path: Path) -> tuple[FakeHarness, str, float]:
    board, task_id = board_with_task(["impl"])
    harness = FakeHarness(tmp_path / "repo", names=("ada",), board=board)
    harness.agents["ada"].situation = Situation.ON_TASK
    board.assign("ada", task_id, "impl")
    task = board.get(task_id)
    board.start(task, "wt")
    return harness, task_id, task.started_at or 0.0


def test_a_silent_seat_is_checked_in_on_twice_then_released(tmp_path):
    harness, task_id, now = running_seat(tmp_path)
    seats = harness.board.get(task_id).seats
    janitor = Janitor(harness)

    janitor.tick(now + 61 * 60)
    janitor.tick(now + 62 * 60)  # the next one is an hour off, not every tick
    assert len(harness.posted) == 1
    assert "Check-in" in harness.posted[0][1].text

    janitor.tick(now + 121 * 60)
    assert len(harness.posted) == 2
    assert "your last one" in harness.posted[1][1].text

    janitor.tick(now + 181 * 60)
    assert seats[0].holder is None


def test_a_check_in_counts_only_when_it_is_answered_inside_the_window(tmp_path):
    harness, task_id, now = running_seat(tmp_path)
    seat = harness.board.get(task_id).seats[0]
    janitor = Janitor(harness)

    janitor.tick(now + 61 * 60)
    harness.agents["ada"].last_turn_at = now + 70 * 60
    janitor.tick(now + 71 * 60)
    assert seat.checkins == 0

    # a turn 24 minutes after the check-in is too late to clear it
    janitor.tick(now + 121 * 60)
    harness.agents["ada"].last_turn_at = now + 145 * 60
    janitor.tick(now + 146 * 60)
    assert seat.checkins == 1
