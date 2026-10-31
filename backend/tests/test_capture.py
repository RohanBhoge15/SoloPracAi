"""Tests for the capture abstention logic.

The gate is a safety mechanism: a wrong number written into a permanent
record is worse than an unread one. These tests assert the refusal path
works and that the parser does not invent structure.
"""

from __future__ import annotations

import pytest

from app.api.routes.capture import parse_dates, parse_labs


SAMPLE = """
City Hospital Laboratory
Report Date: 12/02/2026
Patient: Rakesh Mehta

Test              Result     Unit      Reference
Potassium         2.1        mEq/L     3.5 - 5.1
Sodium            140        mEq/L     135 - 145
Creatinine        1.4        mg/dL     0.6 - 1.3
HbA1c             7.2        %         4.0 - 5.6
"""


def test_extracts_lab_rows():
    rows, confidence = parse_labs(SAMPLE)
    assert len(rows) == 4
    names = {r["test"] for r in rows}
    assert "potassium" in names
    assert "hba1c" in names


def test_known_tests_get_high_confidence():
    rows, confidence = parse_labs(SAMPLE)
    assert confidence > 0.7, "a clean lab report should pass the gate"


def test_known_unit_is_canonicalised():
    rows, _ = parse_labs(SAMPLE)
    potassium = next(r for r in rows if r["test"] == "potassium")
    assert potassium["unit"] == "mEq/L"


def test_reference_range_is_parsed():
    rows, _ = parse_labs(SAMPLE)
    potassium = next(r for r in rows if r["test"] == "potassium")
    assert potassium["ref_low"] == 3.5
    assert potassium["ref_high"] == 5.1
    assert potassium["flag"] == "low", "2.1 is below the reference range"


def test_normal_value_flagged_normal():
    rows, _ = parse_labs(SAMPLE)
    sodium = next(r for r in rows if r["test"] == "sodium")
    assert sodium["flag"] == "normal"


def test_high_value_detected():
    rows, _ = parse_labs(SAMPLE)
    creatinine = next(r for r in rows if r["test"] == "creatinine")
    assert creatinine["flag"] == "high"


def test_unknown_test_lowers_confidence():
    """An unrecognised test name must not be committed at high confidence."""
    _, confident = parse_labs("Zorblax levels 4.2 mEq/L 3.0 - 5.0")
    _, known = parse_labs("Potassium 4.2 mEq/L 3.5 - 5.1")
    assert confident < known


def test_missing_unit_lowers_confidence():
    _, with_unit = parse_labs("Potassium 4.2 mEq/L")
    _, without = parse_labs("Potassium 4.2")
    assert without < with_unit


def test_empty_text_yields_no_rows():
    rows, confidence = parse_labs("")
    assert rows == []
    assert confidence == 0.0


def test_prose_yields_no_rows():
    rows, _ = parse_labs("The patient reports mild dizziness since morning.")
    assert rows == []


def test_garbage_does_not_crash():
    rows, confidence = parse_labs("!!! ))) 12345 --- 0.0.0.0")
    assert isinstance(rows, list)
    assert 0.0 <= confidence <= 1.0


def test_below_threshold_would_refuse():
    """The gate itself: low confidence must not commit."""
    from app.core.config import get_settings

    threshold = get_settings().ocr_confidence_threshold
    _, confidence = parse_labs("Zorblax 4.2")
    assert confidence < threshold, "an unknown test must fall below threshold"


def test_date_parsing():
    assert parse_dates("Report Date: 12/02/2026") == "12/02/2026"
    assert parse_dates("2026-02-12") == "2026-02-12"
    assert parse_dates("no date here") is None


def test_document_serialiser_handles_a_row():
    """Regression: DocumentOut once validated straight off the ORM row.

    `reference` is derived from `content` rather than being a column, so
    from_attributes validation failed with a bare 500. This pins the
    explicit serialiser to the documented shape.
    """
    from types import SimpleNamespace

    from app.api.routes.documents import _out

    row = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        patient_id="22222222-2222-2222-2222-222222222222",
        doctor_id=None,
        kind="prescription",
        title="Prescription for Rakesh",
        description=None,
        filename=None,
        content_type="application/pdf",
        size_bytes=12345,
        content={"reference": "PRE-20260930-ABC123", "items": []},
        extracted={},
        ocr_confidence=None,
        committed=False,
        issued_to_patient=True,
        issued_at=None,
        read_at=None,
        version=1,
        superseded_by_id=None,
        created_at=None,
    )
    # `is_current` is a model property, not a column, so the stub has to
    # provide it the way the real object does.
    row.is_current = row.superseded_by_id is None

    out = _out(row)
    assert out["reference"] == "PRE-20260930-ABC123"
    assert out["is_current"] is True
    assert out["size_bytes"] == 12345
    assert out["patient_id"].startswith("2222")
