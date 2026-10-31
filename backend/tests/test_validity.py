"""Tests for the three-valued validity core.

This is the research contribution. If these fail, the paper has no claim.
Prioritised first in the test strategy (plan §13).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.validity import (
    Event,
    EventType,
    SourceType,
    SupersessionGraph,
    Validity,
    content_hash,
)

BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


def dt(days: float) -> datetime:
    return BASE + timedelta(days=days)


def lab(eid: str, value: str, day: float, **kw) -> Event:
    return Event(
        event_id=eid,
        patient_id=kw.pop("patient_id", "p1"),
        event_type=EventType.LAB,
        valid_time=dt(day),
        payload={"test": "potassium", "value": value, "unit": "mEq/L"},
        **kw,
    )


# ── construction invariants ──────────────────────────────────────────


def test_correction_requires_target():
    with pytest.raises(ValueError, match="must declare supersedes_id"):
        Event(
            event_id="c1",
            patient_id="p1",
            event_type=EventType.LAB,
            valid_time=dt(3),
            payload={},
            source_type=SourceType.CORRECTION,
        )


def test_non_correction_cannot_supersede():
    with pytest.raises(ValueError, match="cannot declare supersedes_id"):
        lab("e1", "2.1", 0, supersedes_id="e0")


def test_confidence_bounds_enforced():
    for bad in (-0.1, 1.1, 2.0):
        with pytest.raises(ValueError, match="outside"):
            lab("e1", "2.1", 0, source_type=SourceType.OCR, confidence=bad)


def test_duplicate_event_id_rejected():
    with pytest.raises(ValueError, match="duplicate event_id"):
        SupersessionGraph([lab("e1", "2.1", 0), lab("e1", "4.8", 1)])


def test_supersedes_unknown_event_rejected():
    orphan = Event(
        event_id="c1",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=dt(3),
        payload={},
        source_type=SourceType.CORRECTION,
        supersedes_id="does-not-exist",
    )
    with pytest.raises(ValueError, match="supersedes unknown"):
        SupersessionGraph([orphan])


def test_cross_patient_supersession_rejected():
    """A correction may never cross a patient boundary."""
    target = lab("e1", "2.1", 0)
    correction = Event(
        event_id="c1",
        patient_id="p2",
        event_type=EventType.LAB,
        valid_time=dt(2),
        payload={},
        source_type=SourceType.CORRECTION,
        supersedes_id="e1",
    )
    with pytest.raises(ValueError, match="cross-patient"):
        SupersessionGraph([target, correction])


# ── the core scenario from the plan ──────────────────────────────────


@pytest.fixture
def amended_potassium() -> SupersessionGraph:
    """The potassium scenario: 2.1 on 12 Feb, amended to 4.8 on 14 Feb."""
    return SupersessionGraph(
        [
            lab("e4471", "2.1", 11),
            Event(
                event_id="e4489",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(13),
                payload={"test": "potassium", "value": "4.8", "unit": "mEq/L"},
                source_type=SourceType.CORRECTION,
                supersedes_id="e4471",
            ),
        ]
    )


def test_original_is_superseded(amended_potassium):
    assert amended_potassium.state(amended_potassium._by_id["e4471"]) is Validity.SUPERSEDED
    assert amended_potassium.state(amended_potassium._by_id["e4489"]) is Validity.CURRENT


def test_correction_is_current(amended_potassium):
    assert amended_potassium.state(amended_potassium._by_id["e4489"]) is Validity.CURRENT


def test_current_view_excludes_superseded(amended_potassium):
    ids = {e.event_id for e in amended_potassium.current_view("p1")}
    assert ids == {"e4489"}, "a stale value must never enter a present-tense answer"


def test_current_view_orders_newest_first(amended_potassium):
    view = amended_potassium.current_view("p1")
    assert [e.event_id for e in view] == ["e4489"]


# ── the property that distinguishes this work from decay baselines ───


def test_as_of_before_correction_keeps_original(amended_potassium):
    """On 13 Feb the 2.1 was still the truth. It must be retrievable."""
    original = amended_potassium._by_id["e4471"]
    assert amended_potassium.state_as_of(original, dt(12)) is Validity.CURRENT
    ids = {e.event_id for e in amended_potassium.as_of_view("p1", dt(12))}
    assert "e4471" in ids
    assert "e4489" not in ids, "the correction did not exist yet"


def test_as_of_after_correction_excludes_original(amended_potassium):
    original = amended_potassium._by_id["e4471"]
    assert amended_potassium.state_as_of(original, dt(14)) is Validity.SUPERSEDED
    ids = {e.event_id for e in amended_potassium.as_of_view("p1", dt(14))}
    assert "e4489" in ids
    assert "e4471" not in ids


def test_as_of_boundary_is_inclusive_of_correction(amended_potassium):
    """The correction is valid from its own valid_time onward."""
    original = amended_potassium._by_id["e4471"]
    assert amended_potassium.state_as_of(original, dt(13)) is Validity.SUPERSEDED


def test_historical_and_present_answers_differ(amended_potassium):
    """The tension the paper measures: one record, two correct answers."""
    historical = amended_potassium.as_of_view("p1", dt(12))
    present = amended_potassium.current_view("p1")
    assert historical[0].payload["value"] == "2.1"
    assert present[0].payload["value"] == "4.8"


# ── transitive chains ────────────────────────────────────────────────


def test_chain_corrected_twice():
    """2.1 -> 4.8 -> 5.1. Validity must be transitive."""
    graph = SupersessionGraph(
        [
            lab("a", "2.1", 0),
            Event(
                event_id="b",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(2),
                payload={"value": "4.8"},
                source_type=SourceType.CORRECTION,
                supersedes_id="a",
            ),
            Event(
                event_id="c",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(4),
                payload={"value": "5.1"},
                source_type=SourceType.CORRECTION,
                supersedes_id="b",
            ),
        ]
    )
    assert graph.state(graph._by_id["a"]) is Validity.SUPERSEDED
    assert graph.state(graph._by_id["b"]) is Validity.SUPERSEDED
    assert graph.state(graph._by_id["c"]) is Validity.CURRENT
    assert [e.event_id for e in graph.chain("a")] == ["a", "b", "c"]


def test_effective_version_walks_chain_to_instant():
    graph = SupersessionGraph(
        [
            lab("a", "2.1", 0),
            Event(
                event_id="b",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(2),
                payload={"value": "4.8"},
                source_type=SourceType.CORRECTION,
                supersedes_id="a",
            ),
            Event(
                event_id="c",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(4),
                payload={"value": "5.1"},
                source_type=SourceType.CORRECTION,
                supersedes_id="b",
            ),
        ]
    )
    assert graph.effective_version(graph._by_id["a"], dt(1)).event_id == "a"
    assert graph.effective_version(graph._by_id["a"], dt(3)).event_id == "b"
    assert graph.effective_version(graph._by_id["a"], dt(5)).event_id == "c"


def test_effective_version_none_before_first_record():
    graph = SupersessionGraph([lab("a", "2.1", 10)])
    assert graph.effective_version(graph._by_id["a"], dt(5)) is None


def test_effective_version_terminates_on_cycle():
    """Defensive: a malformed cyclic graph must not hang the request."""
    a = Event(
        event_id="a",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=dt(0),
        payload={},
        source_type=SourceType.CORRECTION,
        supersedes_id="b",
    )
    b = Event(
        event_id="b",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=dt(1),
        payload={},
        source_type=SourceType.CORRECTION,
        supersedes_id="a",
    )
    graph = SupersessionGraph([a, b])
    assert graph.chain("a") is not None  # terminates, does not spin


# ── patient isolation ────────────────────────────────────────────────


def test_current_view_never_leaks_across_patients():
    graph = SupersessionGraph(
        [lab("a", "2.1", 0, patient_id="p1"), lab("b", "7.4", 1, patient_id="p2")]
    )
    assert [e.event_id for e in graph.current_view("p1")] == ["a"]
    assert [e.event_id for e in graph.current_view("p2")] == ["b"]


def test_as_of_view_never_leaks_across_patients():
    graph = SupersessionGraph(
        [
            lab("a", "2.1", 0, patient_id="p1"),
            lab("b", "7.4", 1, patient_id="p2"),
        ]
    )
    ids = {e.event_id for e in graph.as_of_view("p1", dt(5))}
    assert ids == {"a"}


def test_superseded_pairs_scoped_to_patient():
    graph = SupersessionGraph(
        [
            lab("a", "2.1", 0, patient_id="p1"),
            Event(
                event_id="ca",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(2),
                payload={},
                source_type=SourceType.CORRECTION,
                supersedes_id="a",
            ),
            lab("b", "7.4", 0, patient_id="p2"),
        ]
    )
    pairs = graph.superseded_pairs("p1")
    assert len(pairs) == 1
    assert pairs[0][0].event_id == "a"
    assert pairs[0][1].event_id == "ca"
    assert graph.superseded_pairs("p2") == []


# ── misc ─────────────────────────────────────────────────────────────


def test_content_hash_is_deterministic():
    a = lab("e1", "2.1", 0)
    b = lab("e1", "2.1", 0)
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_with_payload():
    assert content_hash(lab("e1", "2.1", 0)) != content_hash(lab("e1", "4.8", 0))


def test_history_includes_both_states():
    graph = SupersessionGraph(
        [
            lab("a", "2.1", 0),
            Event(
                event_id="b",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(2),
                payload={},
                source_type=SourceType.CORRECTION,
                supersedes_id="a",
            ),
        ]
    )
    assert [e.event_id for e in graph.history("p1")] == ["a", "b"]
    assert graph.stats("p1") == {
        "total": 2,
        "current": 1,
        "superseded": 1,
        "corrections": 1,
    }


def test_naive_datetime_is_normalised():
    """Naive datetimes must not raise on comparison."""
    naive = Event(
        event_id="a",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=datetime(2026, 2, 1),  # no tzinfo
        payload={},
    )
    graph = SupersessionGraph([naive])
    assert graph.current_view("p1")[0].event_id == "a"
