"""Seat lifecycle, finalization, and the janitor that polices seats."""

from __future__ import annotations

import time

import pytest

from panopticon.model import Agent, Event, QueueItem, Situation
from panopticon.services.janitor import Janitor
from panopticon.services.taskboard import TaskBoard


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


class FakeHarness:
    def __init__(self, board: TaskBoard) -> None:
        self.board = board
        self.agents: dict[str, Agent] = {}
        self.posted: list[tuple[str, QueueItem]] = []
        self.events: list[Event] = []

    def add(self, name: str, situation: Situation = Situation.WAITING_FOR_SEATS) -> Agent:
        self.agents[name] = Agent(name=name, provider="test", situation=situation)
        return self.agents[name]

    def post(self, recipient: str, item: QueueItem) -> None:
        self.posted.append((recipient, item))

    def emit(self, event: Event) -> None:
        self.events.append(event)


def waiting_seat(minutes: float) -> tuple[FakeHarness, str, float]:
    board, task_id = board_with_task(["impl", "review"])
    harness = FakeHarness(board)
    harness.add("ada")
    board.assign("ada", task_id, "impl")
    now = time.time()
    board.get(task_id).seat_of("ada").assigned_at = now - minutes * 60
    return harness, task_id, now


def test_nudges_fire_once_per_threshold_then_the_seat_expires():
    harness, task_id, now = waiting_seat(minutes=15)
    janitor = Janitor(harness)

    janitor.tick(now)
    janitor.tick(now + 120)  # still inside the same window
    assert len(harness.posted) == 1
    assert "15 minutes" in harness.posted[0][1].text

    janitor.tick(now + 31 * 60)
    janitor.tick(now + 46 * 60)
    assert len(harness.posted) == 3
    assert harness.board.get(task_id).seat_of("ada").nudges_sent == 3

    janitor.tick(now + 61 * 60)
    assert harness.board.get(task_id).seats[0].holder is None
    assert "unassigned" in harness.posted[-1][1].text
    assert harness.events


def test_a_running_task_is_never_nudged():
    harness, task_id, now = waiting_seat(minutes=50)
    task = harness.board.get(task_id)
    harness.add("bo")
    harness.board.assign("bo", task_id, "review")
    harness.board.start(task, "wt")

    Janitor(harness).tick(now + 61 * 60)

    assert harness.posted == []
    assert task.holders == ["ada", "bo"]


def test_a_dead_agents_seat_is_released_without_nudging_it():
    harness, task_id, now = waiting_seat(minutes=1)
    harness.agents["ada"].situation = Situation.DEAD

    Janitor(harness).tick(now)

    assert harness.board.get(task_id).seats[0].holder is None
    assert harness.posted == []
    assert harness.events[0].agent == "ada"
