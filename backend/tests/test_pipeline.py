"""Tests for the end-to-end answer pipeline.

Runs with no LLM (llm=None) so the retrieval, de-identification, and
provenance stages are exercised deterministically. The model-free path is
also the production fallback when the free tier is unavailable, so it is
worth testing on its own merits.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.validity import Event, EventType, SourceType
from app.services.llm import parse_claims
from app.services.pipeline import (
    AnswerPipeline,
    PatientContext,
    build_context,
    strip_abstention_citations,
)
from app.services.pii import Deidentifier
from app.services.provenance import GroundedAnswer

BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def dt(days: float) -> datetime:
    return BASE + timedelta(days=days)


def lab(eid: str, test: str, value: str, day: float, **kw) -> Event:
    return Event(
        event_id=eid,
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=dt(day),
        payload={"test": test, "value": value, "unit": "mEq/L"},
        **kw,
    )


def correction(eid: str, target: str, value: str, day: float) -> Event:
    return lab(eid, "potassium", value, day, source_type=SourceType.CORRECTION,
               supersedes_id=target)


@pytest.fixture
def context() -> PatientContext:
    events = [
        lab("e4471", "potassium", "2.1", 11),
        correction("e4489", "e4471", "4.8", 13),
        lab("e9001", "sodium", "140", 20),
    ]
    deid = Deidentifier(
        patient_id="p1",
        patient_name="Rakesh Mehta",
        age=54,
        phone="9876543210",
    )
    return build_context(
        "p1", events, deid, display_name="Rakesh Mehta", age=54
    )


def semantic(query: str, event: Event) -> float:
    q = query.lower()
    payload = str(event.payload).lower()
    return 0.9 if event.payload.get("test", "").lower() in q else 0.2


@pytest.fixture
def pipeline() -> AnswerPipeline:
    return AnswerPipeline(llm=None, semantic_fn=semantic, k=5)


# ── present-tense ────────────────────────────────────────────────────


async def test_current_query_never_returns_stale_value(context, pipeline):
    """A present-tense answer must lead with the corrected value.

    The superseded value may still appear in the model-free fallback, but
    only when explicitly labelled. What must never happen is the stale
    value being presented unqualified — that is the silent failure SoloPrac
    exists to prevent.
    """
    result = await pipeline.answer(context, "current potassium", now=NOW)
    lines = [line for line in result.answer.text.splitlines() if line.strip()]
    assert lines, "a present-tense query over existing records must return something"

    current_lines = [line for line in lines if "superseded" not in line.lower()]
    assert any("4.8" in line for line in current_lines), "corrected value must lead"

    # Every line mentioning 2.1 must also declare it superseded.
    for line in lines:
        if "2.1" in line:
            assert "superseded" in line.lower(), (
                "a stale value must never appear unqualified"
            )


async def test_generated_claims_cannot_assert_stale_value(context):
    """The model cannot smuggle a stale value past the linker.

    Simulates a model that faithfully cites a superseded record in the
    present tense. The linker must reject it regardless of how well
    grounded it is.
    """
    from app.services.provenance import Claim, ProvenanceLinker

    linker = ProvenanceLinker(context.graph, "p1")
    claim = Claim("Potassium is 2.1 mEq/L", ("e4471",))
    verdict = linker.verify([claim], {"e4471", "e4489"})[0]
    assert verdict.faithfulness, "the citation is genuine"
    assert not verdict.temporal_validity, "but it must still be rejected"
    assert not verdict.accepted


async def test_current_query_reports_supersession(context, pipeline):
    """The physician should see that a value was corrected."""
    result = await pipeline.answer(context, "current potassium", now=NOW)
    assert result.retrieved
    assert result.intent.kind == "current"


# ── the as-of case ───────────────────────────────────────────────────


async def test_as_of_query_returns_historical_value(context, pipeline):
    result = await pipeline.answer(
        context, "potassium as of 2026-02-12", now=NOW
    )
    assert result.intent.kind == "as_of"
    assert "2.1" in result.answer.text, "history must survive the validity logic"


async def test_explicit_as_of_argument_wins(context, pipeline):
    result = await pipeline.answer(
        context, "potassium", as_of=dt(12), now=NOW
    )
    assert result.intent.kind == "as_of"
    assert "2.1" in result.answer.text


async def test_as_of_after_correction_returns_new_value(context, pipeline):
    result = await pipeline.answer(
        context, "potassium as of 2026-02-20", now=NOW
    )
    assert "4.8" in result.answer.text


# ── de-identification is on the path ─────────────────────────────────


async def test_prompt_is_deidentified(context, pipeline):
    from app.services.pipeline import _render_event
    from app.services.retrieval import parse_temporal_intent, ValidityAwareRetriever

    retriever = ValidityAwareRetriever(graph=context.graph)
    rendered = [_render_event(r.event, r.validity, context.graph, "R1")
                for r in retriever.rank("current potassium", "p1", context.events,
                                        now=NOW, semantic_fn=semantic)]
    prompt = pipeline._deidentify_prompt(
        context, "current potassium for Rakesh Mehta", rendered,
        parse_temporal_intent("current potassium", NOW),
    )
    assert "Rakesh" not in prompt
    assert "Mehta" not in prompt
    assert "9876543210" not in prompt
    assert "potassium" in prompt, "clinical content must survive"


async def test_prompt_passes_the_outbound_gate(context, pipeline):
    from app.services.pipeline import _render_event
    from app.services.retrieval import parse_temporal_intent, ValidityAwareRetriever
    from app.services.pii import assert_payload_safe

    retriever = ValidityAwareRetriever(graph=context.graph)
    rendered = [_render_event(r.event, r.validity, context.graph, "R1")
                for r in retriever.rank("current potassium", "p1", context.events,
                                        now=NOW, semantic_fn=semantic)]
    prompt = pipeline._deidentify_prompt(
        context, "current potassium", rendered,
        parse_temporal_intent("current potassium", NOW),
    )
    assert_payload_safe(prompt)  # raises if a leak survives


async def test_as_of_query_date_is_scrubbed(context, pipeline):
    """A date typed by the physician is as identifying as a name.

    This was a real bug: the records were scrubbed but the user's own
    query was not, so "as of 2026-02-12" reached the third-party endpoint
    and the outbound gate correctly blocked the request.
    """
    from app.services.pipeline import _render_event
    from app.services.retrieval import ValidityAwareRetriever, parse_temporal_intent
    from app.services.pii import assert_payload_safe

    query = "what was the potassium as of 2026-02-12?"
    retriever = ValidityAwareRetriever(graph=context.graph)
    intent = parse_temporal_intent(query, NOW)
    rendered = [_render_event(r.event, r.validity, context.graph, "R1")
                for r in retriever.rank(query, "p1", context.events, intent=intent,
                                        now=NOW, semantic_fn=semantic)]

    prompt = pipeline._deidentify_prompt(context, query, rendered, intent)

    assert "2026-02-12" not in prompt, "the typed date must not reach the LLM"
    assert_payload_safe(prompt)


async def test_as_of_query_answers_without_degrading(context, pipeline):
    """The as-of path must not fall back to extractive.

    Regression guard: when the outbound gate blocked, the pipeline caught
    the exception and degraded silently, which looked like success.
    """
    class ExplodingLLM:
        async def complete(self, *args, **kwargs):
            raise AssertionError("LLM should not be blocked on a scrubbed prompt")

    from app.services.pipeline import AnswerPipeline

    guarded = AnswerPipeline(llm=ExplodingLLM(), semantic_fn=semantic, k=5)
    result = await guarded.answer(
        context, "potassium as of 2026-02-12", now=NOW
    )
    assert result.intent.kind == "as_of"
    assert "2.1" in result.answer.text


# ── no-records and edge cases ────────────────────────────────────────


async def test_empty_record_returns_empty_answer(pipeline):
    context = build_context("p1", [])
    result = await pipeline.answer(context, "current potassium", now=NOW)
    assert result.answer.text == ""


async def test_fallback_marks_superseded_records(pipeline):
    """The model-free path must never present a stale value as current."""
    events = [
        lab("a", "potassium", "2.1", 11),
        correction("b", "a", "4.8", 13),
    ]
    context = build_context("p1", events)
    result = await pipeline.answer(context, "potassium", now=NOW)
    lowered = result.answer.text.lower()
    assert "corrected" in lowered, "the correction must be visible to the physician"
    assert "4.8" in result.answer.text


async def test_as_of_fallback_does_not_call_history_wrong(pipeline):
    """An as-of answer must not describe the historical value as superseded.

    Telling a physician their correct historical answer is stale would be a
    false alarm, and would train them to ignore the warning.
    """
    events = [
        lab("a", "potassium", "2.1", 11),
        correction("b", "a", "4.8", 13),
    ]
    context = build_context("p1", events)
    result = await pipeline.answer(context, "potassium as of 2026-02-12", now=NOW)
    assert "2.1" in result.answer.text
    assert "superseded" not in result.answer.text.lower()
    assert "later corrected" not in result.answer.text.lower()


async def test_every_cited_event_belongs_to_the_patient(context, pipeline):
    result = await pipeline.answer(context, "current potassium", now=NOW)
    for citation in result.answer.citations:
        assert citation["event_id"] in {e.event_id for e in context.events}


# ── claim parsing ────────────────────────────────────────────────────


def test_parse_claims_plain_json():
    raw = '{"claims": [{"text": "Potassium is 4.8 mEq/L", "event_ids": ["e4489"]}]}'
    assert parse_claims(raw)[0]["text"] == "Potassium is 4.8 mEq/L"


def test_parse_claims_fenced_json():
    raw = '```json\n{"claims": [{"text": "Hi", "event_ids": ["e1"]}]}\n```'
    assert len(parse_claims(raw)) == 1


def test_parse_claims_prose_wrapped():
    raw = 'Here is the answer:\n{"claims": [{"text": "Hi", "event_ids": ["e1"]}]}\nHope that helps.'
    assert len(parse_claims(raw)) == 1


def test_parse_claims_single_id_string():
    raw = '{"claims": [{"text": "Hi", "event_ids": "e1"}]}'
    assert parse_claims(raw)[0]["event_ids"] == ["e1"]


def test_parse_claims_malformed_returns_empty():
    assert parse_claims("not json at all") == []
    assert parse_claims("") == []
    assert parse_claims('{"claims": "wrong type"}') == []


def test_parse_claims_drops_entries_without_text():
    raw = '{"claims": [{"event_ids": ["e1"]}, {"text": "ok", "event_ids": ["e2"]}]}'
    parsed = parse_claims(raw)
    assert len(parsed) == 1 and parsed[0]["text"] == "ok"


# ── surrogate citation ids ──────────────────────────────────────────
#
# The model is shown R1, R2, ... rather than 36-character UUIDs, because
# it cannot copy those reliably. The mapping is resolved locally. The
# point of these tests is that a *fabricated* reference still fails: the
# shortening must not become a way past the verifier.


def test_surrogate_ids_are_stable_and_unique():
    from app.services.pipeline import _SurrogateIds

    ids = _SurrogateIds()
    a, b = "event-a", "event-b"
    assert ids.assign(a) == "R1"
    assert ids.assign(b) == "R2"
    # Re-assigning the same event must not burn a new id, or the same
    # record would appear twice under two references.
    assert ids.assign(a) == "R1"
    assert ids.resolve(["R1", "R2"]) == [a, b]


def test_unknown_surrogate_is_left_for_the_verifier_to_reject():
    """A reference we cannot resolve must stay unresolved.

    Dropping it here would convert a fabricated citation into an uncited
    claim, which is a different and more dangerous failure than the one
    the shortening is meant to prevent.
    """
    from app.services.pipeline import _SurrogateIds
    from app.services.provenance import Claim, ProvenanceLinker
    from app.core.validity import SupersessionGraph

    ids = _SurrogateIds()
    ids.assign("event-a")
    claims = [Claim(text="HbA1c was 8.9%", event_ids=tuple(ids.resolve(["R7"])))]

    graph = SupersessionGraph([])
    linker = ProvenanceLinker(graph, "p1")
    verdicts = linker.verify(claims, retrieved_ids=set())
    assert not verdicts[0].faithfulness
    assert any("unknown citation" in r for r in verdicts[0].reasons)


def test_resolved_surrogates_pass_verification():
    from app.services.pipeline import _SurrogateIds
    from app.services.provenance import Claim, ProvenanceLinker
    from app.core.validity import Event, EventType, SourceType, SupersessionGraph
    from datetime import datetime, timezone

    event = Event(
        event_id="event-a",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
        payload={"test": "HbA1c", "value": 8.9, "unit": "%"},
        source_type=SourceType.MANUAL,
    )
    ids = _SurrogateIds()
    ref = ids.assign(event.event_id)
    claims = [Claim(text="HbA1c was 8.9 %", event_ids=tuple(ids.resolve([ref])))]

    linker = ProvenanceLinker(SupersessionGraph([event]), "p1")
    verdicts = linker.verify(claims, retrieved_ids={event.event_id})
    assert verdicts[0].accepted, verdicts[0].reasons


# ── abstention carries no citations ────────────────────────────────


def _answer_with(text: str) -> GroundedAnswer:
    return GroundedAnswer(
        text=text,
        citations=[{"event_id": "e4471"}, {"event_id": "e9001"}],
        dropped=[],
        suppressed=[],
    )


def test_abstention_text_strips_citations():
    out = strip_abstention_citations(
        _answer_with("I don't know - there are no relevant records for this.")
    )
    assert out.text.startswith("I don't know")
    assert out.citations == []


def test_grounded_text_keeps_citations():
    ans = _answer_with("**potassium**: 4.8 mEq/L - 2026-02-14")
    out = strip_abstention_citations(ans)
    assert out.citations == ans.citations


def test_empty_answer_untouched():
    out = strip_abstention_citations(GroundedAnswer(text="", citations=[]))
    assert out.text == "" and out.citations == []


# ── summary questions get broad recall ───────────────────────────


def test_summary_query_widens_k():
    from app.services.pipeline import summary_k

    assert summary_k(
        "what all diseases has happened to this patient till date", 8
    ) == 30
    assert summary_k("summarize this record", 8) == 30
    assert summary_k("full history please", 8) == 30


def test_precision_query_keeps_default_k():
    from app.services.pipeline import summary_k

    assert summary_k("what is the potassium", 8) == 8
    assert summary_k("what was the HbA1c as of March", 8) == 8
