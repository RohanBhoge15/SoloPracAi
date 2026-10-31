"""Tests for the de-identification boundary.

A leak here means real patient data reached a third-party endpoint. This
is a security boundary, so the tests are adversarial: they try hard to get
identifiers past the scrubber and assert that the request is blocked.
"""

from __future__ import annotations

import pytest

from app.services.pii import (
    DeidentificationError,
    Deidentifier,
    assert_payload_safe,
    rehydrate,
)


@pytest.fixture
def deid() -> Deidentifier:
    return Deidentifier(
        patient_id="p-4471",
        patient_name="Rakesh Mehta",
        age=54,
        phone="9876543210",
        email="rakesh.mehta@example.com",
        address="42 MG Road, Pune",
        facility="Sai Clinic",
    )


# ── scrubbing ────────────────────────────────────────────────────────


def test_name_scrubbed(deid):
    out = deid.scrub("Patient Rakesh Mehta presented today")
    assert "Rakesh" not in out
    assert "Mehta" not in out
    assert "[PT_01]" in out


def test_honorific_name_scrubbed(deid):
    out = deid.scrub("Seen by Mr Rajesh Kumar")
    assert "Rajesh" not in out
    assert "Kumar" not in out


def test_hindi_name_scrubbed(deid):
    out = deid.scrub("नाम: रमेश कुमार")
    assert "रमेश" not in out
    assert "कुमार" not in out


def test_phone_scrubbed(deid):
    assert "9876543210" not in deid.scrub("Call 9876543210")


def test_phone_with_country_code_scrubbed(deid):
    assert "9876543210" not in deid.scrub("Call +91 98765-43210")


def test_email_scrubbed(deid):
    assert "rakesh.mehta" not in deid.scrub("Mail rakesh.mehta@example.com")


def test_address_scrubbed(deid):
    assert "MG Road" not in deid.scrub("Lives at 42 MG Road, Pune")


def test_facility_scrubbed(deid):
    assert "Sai Clinic" not in deid.scrub("Referred from Sai Clinic")


def test_age_scrubbed(deid):
    out = deid.scrub("A 54-year-old male")
    assert "54-year-old" not in out
    assert "[AGE_54]" in out


def test_calendar_date_scrubbed():
    """A calendar date is an identifier: it can re-identify a visit."""
    d = Deidentifier(patient_id="p1")
    out = d.scrub("Recorded on 2026-02-14")
    assert "2026-02-14" not in out


def test_date_becomes_relative_offset():
    """The anchor is removed but the ordering survives.

    Replacing a date with a flat token would leave the model unable to
    reason about recency, which is most of what it needs to do.
    """
    from datetime import date

    d = Deidentifier(patient_id="p1", today=date(2026, 3, 1))
    out = d.scrub("Recorded on 2026-02-15")
    assert "2026-02-15" not in out
    assert "[T_MINUS_14D]" in out


def test_today_placeholder():
    from datetime import date

    d = Deidentifier(patient_id="p1", today=date(2026, 3, 1))
    assert "[T_TODAY]" in d.scrub("Recorded on 2026-03-01")


def test_yesterday_placeholder():
    from datetime import date

    d = Deidentifier(patient_id="p1", today=date(2026, 3, 1))
    assert "[T_YESTERDAY]" in d.scrub("Recorded on 2026-02-28")


def test_relative_placeholders_do_not_trip_the_detector():
    """The detector must not flag its own output."""
    from datetime import date

    d = Deidentifier(patient_id="p1", today=date(2026, 3, 1))
    scrubbed = d.scrub("Recorded on 2026-02-15 by lab technician 4412")
    d.assert_safe(scrubbed)


def test_unparseable_date_still_masked():
    from datetime import date

    d = Deidentifier(patient_id="p1", today=date(2026, 3, 1))
    out = d.scrub("Recorded on 2026-02-15 and also 2026-03-02")
    assert "2026-" not in out


def test_long_digit_run_scrubbed():
    d = Deidentifier(patient_id="p1")
    assert "12345678" not in d.scrub("Ref 12345678")


def test_mrn_scrubbed():
    d = Deidentifier(patient_id="p1")
    out = d.scrub("MRN: MH-99281")
    assert "MH-99281" not in out


# ── the clinically important negative ─────────────────────────────────


def test_clinical_values_survive(deid):
    """A lab value is not an identifier. Losing it would break the model."""
    text = "Potassium 4.8 mEq/L, HbA1c 7.2 %, BP 128/82 mmHg"
    out = deid.scrub(text)
    assert "4.8" in out
    assert "7.2" in out
    assert "128/82" in out
    assert "Potassium" in out


def test_payload_scrubbed_recursively(deid):
    payload = {
        "note": "Rakesh Mehta reports dizziness",
        "contact": {"phone": "9876543210", "email": "rakesh.mehta@example.com"},
        "items": ["Sai Clinic referral", "follow-up"],
    }
    out = deid.scrub_payload(payload)
    assert "Rakesh" not in str(out)
    assert "9876543210" not in str(out)
    assert "Sai Clinic" not in str(out)
    assert "follow-up" in out["items"]


# ── the hard gate ────────────────────────────────────────────────────


def test_assert_safe_accepts_clean_payload(deid):
    deid.assert_safe("PT_01 had potassium [VAL_01] mEq/L on [T_MINUS_14D]")


def test_assert_safe_blocks_phone(deid):
    with pytest.raises(DeidentificationError, match="phone"):
        deid.assert_safe("PT_01 phone 9876543210")


def test_assert_safe_blocks_email(deid):
    with pytest.raises(DeidentificationError, match="email"):
        deid.assert_safe("contact rakesh@example.com")


def test_assert_safe_blocks_calendar_date():
    d = Deidentifier(patient_id="p1")
    with pytest.raises(DeidentificationError, match="calendar_date"):
        d.assert_safe("visit on 2026-02-14")


def test_assert_safe_blocks_aadhaar():
    d = Deidentifier(patient_id="p1")
    with pytest.raises(DeidentificationError, match="aadhaar"):
        d.assert_safe("aadhaar 1234 5678 9012")


def test_module_gate_blocks_leak():
    with pytest.raises(DeidentificationError):
        assert_payload_safe("email me at rakesh@example.com")


def test_module_gate_accepts_dict():
    assert_payload_safe({"note": "potassium 4.8 mEq/L", "n": 3})


def test_scrubbed_payload_passes_gate(deid):
    """Scrub then gate must be clean — the happy path, end to end."""
    raw = (
        "Rakesh Mehta, 54-year-old male from Sai Clinic, "
        "phone 9876543210, seen 2026-02-14. "
        "Potassium 4.8 mEq/L, corrected from 2.1."
    )
    assert_payload_safe(deid.scrub(raw))


def test_empty_input_is_safe():
    Deidentifier(patient_id="p1").assert_safe("")
    assert Deidentifier.find_leaks("") == []


# ── rehydration ──────────────────────────────────────────────────────


def test_rehydrate_restores_values():
    mapping = {"[VAL_K_01]": "4.8 mEq/L", "[PT_01]": "Rakesh Mehta"}
    out = rehydrate("[PT_01] potassium [VAL_K_01]", mapping)
    assert "4.8 mEq/L" in out
    assert "Rakesh Mehta" in out


def test_rehydrate_leaves_unmatched_visible():
    out = rehydrate("value [VAL_UNKNOWN] here", {"[PT_01]": "x"})
    assert "[VAL_UNKNOWN]" in out, "a partial round trip must degrade visibly"


def test_rehydrate_handles_empty():
    assert rehydrate("", {"a": "b"}) == ""
