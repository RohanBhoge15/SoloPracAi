"""Tests for entity collapse.

A longitudinal record holds three Metformin rows because the dose changed
twice. The answer must show one line per drug, with a note where the
record shows a change.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.validity import Event, EventType, Validity
from app.services.collapse import collapse, _identity

BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


def dt(days: float) -> datetime:
    return BASE + timedelta(days=days)


def med(eid: str, drug: str, dose: str, day: float, **kw) -> Event:
    status = kw.pop("status", "active")
    return Event(
        event_id=eid,
        patient_id="p1",
        event_type=EventType.MEDICATION,
        valid_time=dt(day),
        payload={
            "drug": drug,
            "dose": dose,
            "frequency": "twice daily",
            "status": status,
        },
        **kw,
    )


def test_drug_versions_collapse_to_one_row():
    rows = [
        (med("a", "Metformin", "500 mg", 0), Validity.CURRENT),
        (med("b", "Metformin", "1000 mg", 10), Validity.CURRENT),
    ]
    kept, _ = collapse(rows)
    assert len(kept) == 1
    assert kept[0][0].payload["dose"] == "1000 mg", "most recent version wins"


def test_collapse_produces_a_change_note():
    rows = [
        (med("a", "Metformin", "500 mg", 0), Validity.CURRENT),
        (med("b", "Metformin", "1000 mg", 10), Validity.CURRENT),
    ]
    _, notes = collapse(rows)
    assert notes
    assert "Metformin" in notes[0]


def test_single_version_produces_no_note():
    rows = [(med("a", "Metformin", "500 mg", 0), Validity.CURRENT)]
    _, notes = collapse(rows)
    assert notes == []


def test_stopped_drug_is_not_reported_as_active():
    """A discontinued drug must never be shown as one the patient takes.

    The newest row is "stopped", an older row is "active". Status has to
    outrank recency, or the answer tells a physician to continue a drug
    that was discontinued for a clinical reason.
    """
    rows = [
        (med("a", "Metformin", "1000 mg", 0, status="active"), Validity.CURRENT),
        (med("b", "Metformin", "1000 mg", 20, status="stopped"), Validity.CURRENT),
    ]
    kept, _ = collapse(rows)
    assert kept[0][0].payload["status"] == "stopped"


def test_active_row_wins_over_older_stopped():
    """Re-starting a drug after a stop must show as active again."""
    rows = [
        (med("a", "Metformin", "500 mg", 0, status="stopped"), Validity.CURRENT),
        (med("b", "Metformin", "500 mg", 30, status="active"), Validity.CURRENT),
    ]
    kept, _ = collapse(rows)
    assert kept[0][0].payload["status"] == "active"


def test_stopped_only_drug_is_still_shown():
    """If a drug was only ever stopped, it should appear marked stopped."""
    rows = [(med("a", "Aspirin", "75 mg", 5, status="stopped"), Validity.CURRENT)]
    kept, _ = collapse(rows)
    assert len(kept) == 1
    assert kept[0][0].payload["status"] == "stopped"


def test_superseded_version_is_not_promoted():
    """A superseded row must never become the shown value."""
    rows = [
        (med("a", "Metformin", "500 mg", 0), Validity.SUPERSEDED),
        (med("b", "Metformin", "1000 mg", 10), Validity.CURRENT),
    ]
    kept, _ = collapse(rows)
    assert kept[0][0].payload["dose"] == "1000 mg"


def test_different_drugs_do_not_collapse():
    rows = [
        (med("a", "Metformin", "500 mg", 0), Validity.CURRENT),
        (med("b", "Telmisartan", "40 mg", 1), Validity.CURRENT),
    ]
    kept, _ = collapse(rows)
    assert len(kept) == 2


def test_case_insensitive_drug_matching():
    rows = [
        (med("a", "Metformin", "500 mg", 0), Validity.CURRENT),
        (med("b", "metformin", "1000 mg", 10), Validity.CURRENT),
    ]
    kept, _ = collapse(rows)
    assert len(kept) == 1


def test_labs_collapse_by_test_name():
    from app.core.validity import EventType as ET

    def lab(eid, value, day):
        return Event(
            event_id=eid,
            patient_id="p1",
            event_type=ET.LAB,
            valid_time=dt(day),
            payload={"test": "Potassium", "value": value},
        )

    kept, _ = collapse(
        [(lab("a", "2.1", 0), Validity.SUPERSEDED), (lab("b", "4.8", 3), Validity.CURRENT)]
    )
    assert len(kept) == 1
    assert kept[0][0].payload["value"] == "4.8"


def test_unidentifiable_records_pass_through():
    note = Event(
        event_id="n1",
        patient_id="p1",
        event_type=EventType.NOTE,
        valid_time=dt(0),
        payload={"note": "follow up"},
    )
    kept, _ = collapse([(note, Validity.CURRENT)])
    assert len(kept) == 1


def test_identity_is_none_for_unknown_shape():
    note = Event(
        event_id="n1",
        patient_id="p1",
        event_type=EventType.NOTE,
        valid_time=dt(0),
        payload={},
    )
    assert _identity(note) is None


def test_output_sorted_newest_first():
    rows = [
        (med("a", "Metformin", "500 mg", 0), Validity.CURRENT),
        (med("b", "Telmisartan", "40 mg", 5), Validity.CURRENT),
    ]
    kept, _ = collapse(rows)
    times = [k[0].valid_time for k in kept]
    assert times == sorted(times, reverse=True)


def _note(eid: str, text: str, day: float) -> Event:
    return Event(
        event_id=eid,
        patient_id="p1",
        event_type=EventType.NOTE,
        valid_time=dt(day),
        payload={"note": text, "author": "Dr Rohan Bhoge"},
    )


def test_identical_notes_collapse_to_one_row():
    """The reported bug: one standing note re-saved at six visits rendered
    as six identical rows with six citations."""
    rows = [(_note(f"n{i}", "evening headache since starting Metformin", i), Validity.CURRENT) for i in range(6)]
    kept, _ = collapse(rows)
    assert len(kept) == 1
    assert kept[0][0].event_id == "n5", "newest version wins"


def test_different_notes_do_not_collapse():
    rows = [
        (_note("n1", "evening headache", 0), Validity.CURRENT),
        (_note("n2", "morning nausea", 5), Validity.CURRENT),
    ]
    kept, _ = collapse(rows)
    assert len(kept) == 2


def test_imaging_collapses_by_study():
    def scan(eid: str, day: float) -> Event:
        return Event(
            event_id=eid,
            patient_id="p1",
            event_type=EventType.IMAGING,
            valid_time=dt(day),
            payload={"study": "chest x-ray"},
        )

    kept, _ = collapse(
        [(scan("s1", 0), Validity.CURRENT), (scan("s2", 9), Validity.CURRENT)]
    )
    assert len(kept) == 1
    assert kept[0][0].event_id == "s2"
