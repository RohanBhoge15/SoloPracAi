"""Tests for provenance locking and the two-axis verifier.

The headline test is `test_stale_value_rejected_in_present_tense`: a
perfectly-cited, perfectly-entailed claim about a superseded record is
rejected because it asserts a present-tense fact. This is the bug the
whole project exists to prevent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.validity import Event, EventType, SourceType, SupersessionGraph
from app.services.provenance import (
    Claim,
    ProvenanceLinker,
    assemble,
    filter_accepted,
)

BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


def dt(days: float) -> datetime:
    return BASE + timedelta(days=days)


def potassium(eid: str, value: str, day: float) -> Event:
    return Event(
        event_id=eid,
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=dt(day),
        payload={"test": "potassium", "value": value, "unit": "mEq/L"},
    )


@pytest.fixture
def graph() -> SupersessionGraph:
    return SupersessionGraph(
        [
            potassium("e4471", "2.1", 11),
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


@pytest.fixture
def linker(graph) -> ProvenanceLinker:
    return ProvenanceLinker(graph, "p1")


# ── the headline failure mode ────────────────────────────────────────


def test_stale_value_rejected_in_present_tense(linker):
    """Faithful, correctly cited, and clinically wrong. Must be rejected."""
    claim = Claim(
        text="The patient's potassium is 2.1 mEq/L.",
        event_ids=("e4471",),
    )
    verdict = linker.verify([claim], {"e4471", "e4489"})[0]
    assert verdict.faithfulness, "the citation is genuine — Axis A must pass"
    assert not verdict.temporal_validity, "Axis B is what must reject it"
    assert not verdict.accepted


def test_corrected_value_accepted_in_present_tense(linker):
    claim = Claim(
        text="The patient's potassium is 4.8 mEq/L.",
        event_ids=("e4489",),
    )
    verdict = linker.verify([claim], {"e4471", "e4489"})[0]
    assert verdict.accepted


def test_stale_value_accepted_in_past_tense(linker):
    """A historical question about a superseded record is legitimate."""
    claim = Claim(
        text="The potassium was 2.1 mEq/L before the correction.",
        event_ids=("e4471",),
    )
    verdict = linker.verify([claim], {"e4471", "e4489"})[0]
    assert verdict.faithfulness
    assert verdict.temporal_validity, "past-tense claims may cite superseded events"
    assert verdict.accepted


def test_tense_detection():
    assert Claim("potassium is 4.8", ()).tense == "present"
    assert Claim("potassium was 2.1", ()).tense == "past"
    assert Claim("metformin dose", ()).tense == "neutral"


# ── faithfulness rejections ──────────────────────────────────────────


def test_unknown_citation_rejected(linker):
    claim = Claim("Potassium is 4.8 mEq/L", ("does-not-exist",))
    verdict = linker.verify([claim], {"e4489"})[0]
    assert not verdict.faithfulness
    assert any("unknown citation" in r for r in verdict.reasons)


def test_citation_outside_retrieved_set_rejected(linker):
    """Citing a real event that retrieval never returned is still a failure."""
    claim = Claim("Potassium is 4.8 mEq/L", ("e4489",))
    verdict = linker.verify([claim], {"e4471"})[0]
    assert not verdict.faithfulness
    assert any("not in retrieved set" in r for r in verdict.reasons)


def test_invented_number_rejected(linker):
    claim = Claim("Potassium is 9.9 mEq/L", ("e4489",))
    verdict = linker.verify([claim], {"e4489"})[0]
    assert not verdict.faithfulness
    assert any("not present in any citation" in r for r in verdict.reasons)


def test_numeric_claim_without_citation_rejected(linker):
    claim = Claim("Potassium is 4.8 mEq/L", ())
    verdict = linker.verify([claim], {"e4489"})[0]
    assert not verdict.faithfulness


def test_qualitative_claim_needs_no_number(linker):
    claim = Claim("The patient reports dizziness.", ("e4489",))
    verdict = linker.verify([claim], {"e4489"})[0]
    assert verdict.faithfulness, "a claim with no numbers is judged on citation only"


def test_cross_patient_citation_rejected(graph):
    foreign = Event(
        event_id="x1",
        patient_id="p2",
        event_type=EventType.LAB,
        valid_time=dt(12),
        payload={"value": "9.9"},
    )
    g = SupersessionGraph(graph.all_events + [foreign])
    linker = ProvenanceLinker(g, "p1")
    claim = Claim("Potassium is 9.9 mEq/L", ("x1",))
    verdict = linker.verify([claim], {"x1"})[0]
    assert not verdict.faithfulness
    assert any("cross-patient" in r for r in verdict.reasons)


def test_integer_and_float_forms_match(graph):
    """'5' and '5.0' are the same value.

    Without normalisation the linker would report a false mismatch and drop
    a perfectly correct claim. Over-strict numeric checking is a real
    failure mode: it silently discards good answers.
    """
    whole = Event(
        event_id="n1",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=dt(5),
        payload={"test": "sodium", "value": 5, "unit": "mmol/L"},
    )
    g = SupersessionGraph(graph.all_events + [whole])
    linker = ProvenanceLinker(g, "p1")
    claim = Claim("Sodium is 5.0 mmol/L", ("n1",))
    verdict = linker.verify([claim], {"n1"})[0]
    assert verdict.faithfulness, verdict.reasons


def test_genuinely_different_number_still_rejected(linker):
    """Normalisation must not make 9.9 look like 4.8."""
    claim = Claim("Potassium is 9.9 mEq/L", ("e4489",))
    verdict = linker.verify([claim], {"e4489"})[0]
    assert not verdict.faithfulness


# ── assembly ─────────────────────────────────────────────────────────


def test_accepted_claims_survive(linker):
    verdicts = linker.verify(
        [Claim("Potassium is 4.8 mEq/L", ("e4489",))], {"e4471", "e4489"}
    )
    kept = filter_accepted(verdicts)
    assert len(kept) == 1
    assert "4.8" in kept[0].text


def test_stale_claim_is_suppressed_not_silently_dropped(linker):
    """A doctor should be able to see that a value was corrected."""
    verdicts = linker.verify(
        [Claim("Potassium is 2.1 mEq/L", ("e4471",))], {"e4471", "e4489"}
    )
    answer = assemble(verdicts)
    assert answer.text == ""
    assert answer.suppressed, "the correction must be reported, not hidden"
    assert answer.has_stale_assertion


def test_fabrication_is_dropped(linker):
    verdicts = linker.verify(
        [Claim("Potassium is 9.9 mEq/L", ("e4489",))], {"e4489"}
    )
    answer = assemble(verdicts)
    assert answer.text == ""
    assert answer.dropped
    assert not answer.suppressed


def test_mixed_claims_keep_only_verified(linker):
    verdicts = linker.verify(
        [
            Claim("Potassium is 4.8 mEq/L", ("e4489",)),
            Claim("Potassium is 2.1 mEq/L", ("e4471",)),
        ],
        {"e4471", "e4489"},
    )
    answer = assemble(verdicts)
    assert "4.8" in answer.text
    assert "2.1" not in answer.text
    assert answer.suppressed


def test_citations_deduplicated(linker):
    verdicts = linker.verify(
        [
            Claim("Potassium is 4.8 mEq/L", ("e4489",)),
            Claim("Units are mEq/L", ("e4489",)),
        ],
        {"e4489"},
    )
    answer = assemble(verdicts)
    assert len(answer.citations) == 1


def test_empty_input_produces_empty_answer(linker):
    assert assemble(linker.verify([], set())).text == ""


# ── temporal offsets are not clinical values ────────────────────────
#
# "HbA1c was 8.9 % as of 60 days ago" is fully grounded. The "60" is the
# model quoting the question back, and treating it as a measurement rejects
# a correct historical answer - the one this system exists to give. These
# tests pin that behaviour AND the adversarial case, so relaxing the check
# cannot quietly become a way past the verifier.


def test_relative_offset_is_not_treated_as_a_clinical_value():
    from app.services.provenance import _clinical_numbers

    numbers = _clinical_numbers("HbA1c was 8.9 % as of 60 days ago")
    assert "8.9" in numbers
    assert "60" not in numbers, "the offset leaked into the clinical check"


def test_placeholder_offsets_are_ignored():
    from app.services.provenance import _clinical_numbers

    numbers = _clinical_numbers("HbA1c was 8.9 % as of T_MINUS_90D")
    assert "8.9" in numbers
    assert "90" not in numbers


def test_explicit_dates_are_ignored():
    from app.services.provenance import _clinical_numbers

    for text in (
        "HbA1c was 8.9 % on 2026-02-12",
        "HbA1c was 8.9 % as of 12 Feb 2026",
    ):
        numbers = _clinical_numbers(text)
        assert "8.9" in numbers, text
        assert "2026" not in numbers and "12" not in numbers, text


def test_a_fabricated_value_cannot_hide_inside_a_temporal_phrase():
    """The relaxation must not become a loophole.

    The offset pattern requires a unit word after the number, so a real
    measurement written next to a date is still checked.
    """
    from app.services.provenance import _clinical_numbers

    numbers = _clinical_numbers(
        "HbA1c was 8.9 % as of 60 days ago, and potassium was 99 mEq/L"
    )
    assert "8.9" in numbers
    assert "99" in numbers, "a fabricated value was allowed through"
    assert "60" not in numbers


def test_offset_removal_requires_a_unit_word():
    """A bare number beside a date is not an offset and must be checked."""
    from app.services.provenance import _clinical_numbers

    numbers = _clinical_numbers("potassium 4.8 on 2026-02-12 measured 2")
    assert "4.8" in numbers
    assert "2" in numbers
    assert "2026" not in numbers and "12" not in numbers


def test_grounder_accepts_a_historical_claim_with_an_offset():
    from app.core.validity import (
        Event, EventType, SourceType, SupersessionGraph,
    )
    from app.services.provenance import Claim, ProvenanceLinker
    from datetime import datetime, timezone

    event = Event(
        event_id="e1",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        payload={"test": "HbA1c", "value": 8.9, "unit": "%"},
        source_type=SourceType.MANUAL,
    )
    claim = Claim(
        text="HbA1c was 8.9 % as of 60 days ago", event_ids=("e1",)
    )
    linker = ProvenanceLinker(SupersessionGraph([event]), "p1")
    verdict = linker.verify([claim], retrieved_ids={"e1"})
    assert verdict[0].accepted, verdict[0].reasons


def test_grounder_still_rejects_an_invented_value():
    """The original guarantee must survive the relaxation."""
    from app.core.validity import (
        Event, EventType, SourceType, SupersessionGraph,
    )
    from app.services.provenance import Claim, ProvenanceLinker
    from datetime import datetime, timezone

    event = Event(
        event_id="e1",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        payload={"test": "HbA1c", "value": 8.9, "unit": "%"},
        source_type=SourceType.MANUAL,
    )
    claim = Claim(text="HbA1c was 12.4 %", event_ids=("e1",))
    linker = ProvenanceLinker(SupersessionGraph([event]), "p1")
    verdict = linker.verify([claim], retrieved_ids={"e1"})
    assert not verdict[0].accepted
    assert any("not present" in r for r in verdict[0].reasons)
