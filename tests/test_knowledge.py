"""The jury: verdict races, restatement lineage, and who may judge what."""

from __future__ import annotations

import json

import pytest

from panopticon.model import VerdictCall
from panopticon.services.knowledge import Knowledge

TRUE = VerdictCall.TRUE
FALSE = VerdictCall.FALSE
RESTATE = VerdictCall.RESTATE


def submitted(*jurors: str) -> tuple[Knowledge, str]:
    kb = Knowledge()
    sub = kb.submit("ada", "parser rejects empty input", "src/parse.py:31 raises on b''")
    for juror in jurors:
        kb.join(juror, sub.id)
    return kb, sub.id


def test_two_true_verdicts_accept_the_truth():
    kb, sid = submitted("bo", "cy")

    _, outcome = kb.verdict("bo", sid, TRUE, "checked the source")
    assert outcome == "pending"
    _, outcome = kb.verdict("cy", sid, TRUE, "same here")

    assert outcome == "accepted"
    assert kb.pending == {}
    assert [t.title for t in kb.truths] == ["parser rejects empty input"]
    assert kb.truths[0].submitted_by == "ada"


def test_one_false_discards_before_a_second_true_can_land():
    kb, sid = submitted("bo", "cy", "di")
    kb.verdict("bo", sid, TRUE, "looks right")

    _, outcome = kb.verdict("cy", sid, FALSE, "line 31 does not raise")

    assert outcome == "rejected"
    assert kb.pending == {}
    assert kb.truths == []
    with pytest.raises(ValueError):
        kb.verdict("di", sid, TRUE, "too late")


def test_true_then_restate_then_false_discards_because_false_came_first():
    kb, sid = submitted("bo", "cy", "di")
    kb.verdict("bo", sid, TRUE, "looks right")

    replacement, outcome = kb.verdict(
        "cy", sid, RESTATE, "too broad", "parse('') raises ValueError", "src/parse.py:31"
    )
    assert outcome == "restated"
    assert replacement.restated_from == sid
    assert replacement.verdicts == []
    assert list(kb.pending) == [replacement.id]

    _, outcome = kb.verdict("di", replacement.id, FALSE, "it returns None")

    assert outcome == "rejected"
    assert kb.truths == []


def test_a_restatement_is_judged_from_scratch():
    kb, sid = submitted("bo", "cy")
    kb.verdict("bo", sid, TRUE, "looks right")
    replacement, _ = kb.verdict(
        "cy", sid, RESTATE, "too broad", "parse('') raises ValueError", "src/parse.py:31"
    )

    # bo's earlier true is gone with the old statement, so one true is not enough
    _, outcome = kb.verdict("bo", replacement.id, TRUE, "still holds")
    assert outcome == "pending"
    assert kb.truths == []

    kb.join("di", replacement.id)
    _, outcome = kb.verdict("di", replacement.id, TRUE, "confirmed")
    assert outcome == "accepted"
    assert kb.truths[0].title == "parse('') raises ValueError"
    assert kb.truths[0].submitted_by == "cy"  # the restater owns the wording


def test_nobody_judges_their_own_statement_or_its_restatement():
    kb, sid = submitted("bo", "cy")
    with pytest.raises(ValueError):
        kb.join("ada", sid)

    replacement, _ = kb.verdict(
        "cy", sid, RESTATE, "too broad", "parse('') raises ValueError", "src/parse.py:31"
    )

    with pytest.raises(ValueError):
        kb.join("ada", replacement.id)  # ada's claim, reworded
    with pytest.raises(ValueError):
        kb.join("cy", replacement.id)  # cy wrote the replacement
    assert "cy" not in replacement.jurors
    assert replacement.jurors == ["bo"]  # the other juror stays seated on the new wording


def test_a_juror_rules_once_and_only_from_the_bench():
    kb, sid = submitted("bo")

    with pytest.raises(ValueError):
        kb.verdict("cy", sid, TRUE, "not seated")

    kb.verdict("bo", sid, TRUE, "checked")
    with pytest.raises(ValueError):
        kb.verdict("bo", sid, TRUE, "checked twice")
    assert kb.truths == []


def test_a_restate_without_a_replacement_is_refused_and_leaves_no_trace():
    kb, sid = submitted("bo")

    with pytest.raises(ValueError):
        kb.verdict("bo", sid, RESTATE, "too broad")

    assert kb.pending[sid].verdicts == []
    _, outcome = kb.verdict("bo", sid, TRUE, "fine as it stands")
    assert outcome == "pending"


def test_leaving_the_jury_frees_the_seat_without_touching_verdicts():
    kb, sid = submitted("bo", "cy")
    kb.verdict("bo", sid, TRUE, "checked")

    kb.leave("cy", sid)

    assert kb.pending[sid].jurors == ["bo"]
    with pytest.raises(ValueError):
        kb.leave("cy", sid)


def test_renders_carry_what_an_agent_needs():
    kb, sid = submitted("bo", "cy")
    assert "empty" in kb.render_pending()
    assert "ada" in kb.render_pending()
    assert kb.render() == "knowledge base: empty"

    kb.verdict("bo", sid, TRUE, "checked")
    kb.verdict("cy", sid, TRUE, "checked")

    assert "src/parse.py:31" in kb.render()
    assert kb.render_pending() == "jury queue: empty"


def test_a_restored_knowledge_base_keeps_its_bar_and_its_id_sequence():
    kb, first = submitted()
    second = kb.submit("bo", "the suite needs PANOPTICON_HOME", "else it writes to the real home")
    kb.join("ada", second.id)

    revived = Knowledge()
    revived.restore(json.loads(json.dumps(kb.snapshot())))

    assert list(revived.pending) == [first, second.id]
    with pytest.raises(ValueError, match="your own statement"):
        revived.join("bo", second.id)
    assert revived.submit("cy", "third", "proof").id == "s3"
