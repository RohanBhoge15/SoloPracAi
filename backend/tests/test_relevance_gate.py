"""Tests for the relevance gate in the extractive answer.

A potassium question must not answer with the patient's blood pressure.
Retrieval ranks correctly; the presentation layer must respect that.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.validity import Event, EventType, SourceType
from app.services.pipeline import AnswerPipeline, build_context
from app.services.retrieval import lexical_similarity

BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def dt(days: float) -> datetime:
    return BASE + timedelta(days=days)


def lab(eid, test, value, day, **kw):
    return Event(
        event_id=eid,
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=dt(day),
        payload={"test": test, "value": value, "unit": "mEq/L"},
        **kw,
    )


def med(eid, drug, dose, day):
    return Event(
        event_id=eid,
        patient_id="p1",
        event_type=EventType.MEDICATION,
        valid_time=dt(day),
        payload={"drug": drug, "dose": dose, "frequency": "twice daily",
                 "status": "active"},
    )


@pytest.fixture
def pipeline():
    return AnswerPipeline(llm=None, semantic_fn=lexical_similarity, k=8)


@pytest.fixture
def mixed_context():
    events = [
        lab("k1", "Potassium", 2.1, 11),
        lab("k2", "Potassium", 4.8, 13, source_type=SourceType.CORRECTION,
            supersedes_id="k1"),
        lab("s1", "Sodium", 140, 20),
        med("m1", "Metformin", "500 mg", 25),
        med("m2", "Telmisartan", "40 mg", 26),
    ]
    return build_context("p1", events)


async def test_potassium_question_excludes_other_tests(pipeline, mixed_context):
    result = await pipeline.answer(mixed_context, "current potassium level?", now=NOW)
    lowered = result.answer.text.lower()
    assert "potassium" in lowered
    assert "sodium" not in lowered, "zero-similarity records are noise, not context"
    assert "metformin" not in lowered


async def test_medication_question_includes_drugs(pipeline, mixed_context):
    result = await pipeline.answer(mixed_context, "current medications?", now=NOW)
    lowered = result.answer.text.lower()
    assert "metformin" in lowered and "telmisartan" in lowered


async def test_medication_question_excludes_labs(pipeline, mixed_context):
    result = await pipeline.answer(mixed_context, "current medications?", now=NOW)
    assert "sodium" not in result.answer.text.lower()


async def test_citations_match_shown_content(pipeline, mixed_context):
    result = await pipeline.answer(mixed_context, "current potassium level?", now=NOW)
    assert result.answer.citations
    assert all("event_id" in c for c in result.answer.citations)


async def test_broad_question_still_shows_context(pipeline, mixed_context):
    """A vague question legitimately returns more."""
    result = await pipeline.answer(mixed_context, "summarise the record", now=NOW)
    assert len(result.retrieved) > 1
