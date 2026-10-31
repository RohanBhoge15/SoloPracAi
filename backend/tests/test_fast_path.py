"""Tests for the model-free fast path.

A single unambiguous record needs no LLM. Free tiers take seconds and
rate-limit, and a physician asking a simple question should not wait.
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


class ExplodingLLM:
    """Any outbound call is a failure for these tests."""

    async def complete(self, *args, **kwargs):
        raise AssertionError("LLM must not be called on the direct path")

    text = ""
    model = "never"
    degraded = False


@pytest.fixture
def pipeline():
    return AnswerPipeline(llm=ExplodingLLM(), semantic_fn=lexical_similarity, k=8)


async def test_single_clear_match_avoids_the_llm(pipeline):
    context = build_context("p1", [lab("k", "Potassium", 4.8, 20)])
    result = await pipeline.answer(context, "current potassium level?", now=NOW)
    assert result.llm is None, "should have answered without the model"
    assert "4.8" in result.answer.text


async def test_ambiguous_query_still_uses_the_llm(pipeline):
    """Two drugs match equally well; the model is needed to disambiguate."""
    meds = [
        Event(
            event_id="m1",
            patient_id="p1",
            event_type=EventType.MEDICATION,
            valid_time=dt(20),
            payload={"drug": "Metformin", "dose": "500 mg",
                     "frequency": "twice daily", "status": "active"},
        ),
        Event(
            event_id="m2",
            patient_id="p1",
            event_type=EventType.MEDICATION,
            valid_time=dt(21),
            payload={"drug": "Telmisartan", "dose": "40 mg",
                     "frequency": "once daily", "status": "active"},
        ),
    ]
    context = build_context("p1", meds)
    # The exploding LLM raises, so the pipeline must fall back to extract.
    result = await pipeline.answer(context, "current medications?", now=NOW)
    assert result.llm is None or result.llm.model == "never"
    assert "Metformin" in result.answer.text
    assert "Telmisartan" in result.answer.text


async def test_superseded_only_match_does_not_take_the_fast_path(pipeline):
    """A present-tense answer must never come from a superseded record."""
    events = [
        lab("a", "Potassium", 2.1, 10),
        lab("b", "Potassium", 4.8, 12,
            source_type=SourceType.CORRECTION, supersedes_id="a"),
    ]
    context = build_context("p1", events)
    result = await pipeline.answer(context, "current potassium level?", now=NOW)
    assert "4.8" in result.answer.text
    assert "2.1" not in result.answer.text


async def test_fast_path_is_fast(pipeline):
    """The point of the fast path is that it does not wait."""
    import time

    context = build_context("p1", [lab("k", "Potassium", 4.8, 20)])
    started = time.perf_counter()
    await pipeline.answer(context, "current potassium level?", now=NOW)
    assert (time.perf_counter() - started) < 0.05
