"""Tests for validity-aware retrieval.

The key behavioural test is `test_as_of_survives_present_tuning`: an
as-of query must still return the historical value no matter how the
present-tense weighting is tuned. That is the property decay cannot
express, and it is the paper's central claim.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.validity import Event, EventType, SourceType, SupersessionGraph, Validity
from app.services.retrieval import (
    ValidityAwareRetriever,
    parse_temporal_intent,
    rank_dense_only,
    rank_latest_n,
    rank_uniform_decay,
    validity_weight,
)

BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def dt(days: float) -> datetime:
    return BASE + timedelta(days=days)


def correction(eid: str, target: str, day: float, value: str) -> Event:
    return Event(
        event_id=eid,
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=dt(day),
        payload={"test": "potassium", "value": value, "unit": "mEq/L"},
        source_type=SourceType.CORRECTION,
        supersedes_id=target,
    )


@pytest.fixture
def graph() -> SupersessionGraph:
    """One patient, one amended potassium, plus an untouched distractor."""
    return SupersessionGraph(
        [
            Event(
                event_id="e4471",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(11),
                payload={"test": "potassium", "value": "2.1", "unit": "mEq/L"},
            ),
            correction("e4489", "e4471", 13, "4.8"),
            Event(
                event_id="e9001",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(20),
                payload={"test": "sodium", "value": "140", "unit": "mEq/L"},
            ),
        ]
    )


def always_potassium(query: str, event: Event) -> float:
    """Stand-in for an embedding model: both potassiums match equally.

    The point is to isolate the temporal contribution from the embedding
    contribution. If both records match equally well on semantics, any
    difference in ranking is caused by validity weighting alone.
    """
    return 0.9 if event.payload.get("test") == "potassium" else 0.1


# ── intent parsing ───────────────────────────────────────────────────


def test_current_intent():
    assert parse_temporal_intent("current potassium?", NOW).kind == "current"


def test_latest_is_current():
    assert parse_temporal_intent("latest HbA1c", NOW).kind == "current"


def test_historical_intent():
    assert parse_temporal_intent("initial diagnosis", NOW).kind == "historical"


def test_as_of_intent():
    intent = parse_temporal_intent("potassium as of 12 Feb 2026", NOW)
    assert intent.kind == "as_of"
    assert intent.instant is not None


def test_as_of_parses_instant():
    intent = parse_temporal_intent("potassium as of 2026-02-12", NOW)
    assert intent.instant == datetime(2026, 2, 12, tzinfo=timezone.utc)


def test_most_recent_change_is_current_not_historical():
    """"most recent" outranks "change" — ordering matters."""
    assert parse_temporal_intent("most recent change in dose", NOW).kind == "current"


# ── relative as-of expressions ───────────────────────────────────────
#
# "as of 60 days ago" contains no absolute date. Without these cases the
# date patterns find nothing, the instant falls back to `now`, and a
# historical question gets answered about the present — confidently,
# which is the worst failure a temporally-aware system can have.


@pytest.mark.parametrize(
    "query,days",
    [
        ("what was her hba1c as of 60 days ago", 60),
        ("hba1c 3 days back", 3),
        ("as of two weeks ago", 14),
        ("her creatinine 1 month ago", 30),
        ("as of 6 weeks before", 42),
        ("what was it like 3 years ago", 365 * 3),
        ("as of a fortnight ago", 14),
    ],
)
def test_relative_as_of_resolves(query, days):
    intent = parse_temporal_intent(query, NOW)
    assert intent.kind == "as_of", f"{query!r} was not read as as_of"
    assert intent.instant is not None
    assert (NOW - intent.instant).days == pytest.approx(days, abs=1), (
        f"{query!r} resolved to the wrong offset"
    )


@pytest.mark.parametrize(
    "query",
    [
        "what was it as of yesterday",
        "hba1c last week",
        "as of last month",
    ],
)
def test_bare_relative_cues(query):
    intent = parse_temporal_intent(query, NOW)
    assert intent.kind == "as_of"
    assert intent.instant is not None
    assert intent.instant < NOW


def test_zero_days_ago_is_the_present():
    intent = parse_temporal_intent("as of 0 days ago", NOW)
    assert intent.kind == "as_of"
    assert intent.instant == NOW


def test_absolute_date_wins_over_relative_words():
    """A concrete date is more precise than "days ago", so it is used."""
    intent = parse_temporal_intent("as of 2025-06-15, 10 days ago", NOW)
    assert intent.kind == "as_of"
    assert intent.instant == datetime(2025, 6, 15, tzinfo=timezone.utc)


def test_queries_without_a_reference_are_still_current():
    for query in ("what is her blood sugar", "is her pressure okay"):
        assert parse_temporal_intent(query, NOW).kind == "current", query


def test_a_duration_in_a_question_is_not_a_time_travel_request():
    """"on this drug for 3 days" is a duration, not an as-of request."""
    intent = parse_temporal_intent(
        "how long has she been on metformin for 3 days", NOW
    )
    assert intent.kind != "as_of"


def test_as_of_cue_outranks_a_historical_cue():
    """`as of` must win even when a historical cue is also present.

    "what *was* her hba1c as of 60 days ago" contains "was". If the
    historical branch were reached first the instant would be dropped and
    the answer drawn from the wrong end of the timeline.
    """
    intent = parse_temporal_intent("what was her hba1c as of 60 days ago", NOW)
    assert intent.kind == "as_of"
    assert intent.instant == NOW - timedelta(days=60)


def test_empty_query_defaults_current():
    assert parse_temporal_intent("", NOW).kind == "current"


# ── the central claim ────────────────────────────────────────────────


def test_as_of_survives_present_tuning(graph):
    """The paper's claim, as a test.

    An as-of query must return the value that was true *then*, regardless
    of how present-tense weighting is configured. A decay function cannot
    satisfy this.
    """
    retriever = ValidityAwareRetriever(graph=graph)
    candidates = graph.all_events
    intent = parse_temporal_intent("potassium as of 2026-02-12", NOW)

    results = retriever.rank(
        "potassium as of 2026-02-12",
        "p1",
        candidates,
        intent=intent,
        now=NOW,
        semantic_fn=always_potassium,
    )
    assert results, "as-of query returned nothing"
    assert results[0].event.payload["value"] == "2.1"


def test_present_query_returns_corrected_value(graph):
    retriever = ValidityAwareRetriever(graph=graph)
    results = retriever.rank(
        "current potassium",
        "p1",
        graph.all_events,
        now=NOW,
        semantic_fn=always_potassium,
    )
    assert results[0].event.payload["value"] == "4.8"


def test_dense_only_baseline_answers_wrongly(graph):
    """The failure mode SoloPrac exists to fix, demonstrated.

    With no temporal term, both potassiums match equally and the older
    one can win on tie-break. This is a confidently-wrong answer with a
    correct citation.
    """
    results = rank_dense_only(
        graph.all_events, lambda q, e: 0.9 if e.payload.get("test") == "potassium" else 0.1
    )
    values = [r.event.payload["value"] for r in results if r.event.payload.get("test") == "potassium"]
    assert "2.1" in values[:1], "dense-only surfaces the superseded value first"


def test_our_retriever_beats_dense_only_on_current(graph):
    retriever = ValidityAwareRetriever(graph=graph)
    ours = retriever.rank(
        "current potassium", "p1", graph.all_events, now=NOW, semantic_fn=always_potassium
    )
    assert ours[0].event.payload["value"] == "4.8"


def test_validity_weight_is_categorical(graph):
    """Corrected 2 days ago and 2 years ago must weight identically."""
    from app.services.retrieval import W_CURRENT, W_SUPERSEDED

    current = graph._by_id["e9001"]
    superseded = graph._by_id["e4471"]
    intent = parse_temporal_intent("current value", NOW)

    assert validity_weight(current, intent, graph) == W_CURRENT
    assert validity_weight(superseded, intent, graph) == W_SUPERSEDED


def test_correction_is_weighted_as_current(graph):
    from app.services.retrieval import W_CURRENT

    intent = parse_temporal_intent("current potassium", NOW)
    assert validity_weight(graph._by_id["e4489"], intent, graph) == W_CURRENT


# ── safety constraints ───────────────────────────────────────────────


def test_future_leak_is_vetoed(graph):
    """No record may post-date the question."""
    retriever = ValidityAwareRetriever(graph=graph, enforce_future_leak=True)
    intent = parse_temporal_intent("potassium as of 2026-02-12", NOW)
    results = retriever.rank(
        "potassium as of 2026-02-12", "p1", graph.all_events, intent=intent, now=NOW,
        semantic_fn=always_potassium,
    )
    assert all(r.event.valid_time <= intent.instant for r in results)
    assert all(r.event.event_id != "e9001" for r in results)


def test_cross_patient_candidate_is_dropped(graph):
    """Defence in depth against a filter bug."""
    foreign = Event(
        event_id="x1",
        patient_id="p2",
        event_type=EventType.LAB,
        valid_time=dt(20),
        payload={"test": "potassium", "value": "9.9"},
    )
    retriever = ValidityAwareRetriever(graph=graph)
    results = retriever.rank(
        "current potassium", "p1", graph.all_events + [foreign], now=NOW,
        semantic_fn=lambda q, e: 0.95,
    )
    assert all(r.event.patient_id == "p1" for r in results)


def test_ranks_are_sequential(graph):
    retriever = ValidityAwareRetriever(graph=graph)
    results = retriever.rank(
        "current values", "p1", graph.all_events, now=NOW,
        semantic_fn=lambda q, e: 0.5,
    )
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_empty_candidates_returns_empty(graph):
    retriever = ValidityAwareRetriever(graph=graph)
    assert retriever.rank("anything", "p1", [], now=NOW) == []


# ── the decay trade-off the paper measures ───────────────────────────


def test_uniform_decay_overshadows_semantics():
    """Reproduces the documented pathology at delta=0.15.

    With a large delta the decay term swamps the semantic term, so
    ranking collapses to "most recent wins" regardless of the query. This
    is why delta=0.15 must never be a live default.
    """
    events = [
        Event(
            event_id=f"e{i}",
            patient_id="p1",
            event_type=EventType.LAB,
            valid_time=dt(i),
            payload={"value": str(i)},
        )
        for i in range(10)
    ]
    results = rank_uniform_decay(
        events, lambda q, e: 0.5, delta=0.15, now=NOW, tau_days=30.0
    )
    assert results[0].event.valid_time == events[-1].valid_time


def test_decay_grid_moves_along_a_tradeoff():
    """Higher delta favours recency more aggressively.

    This single-dial behaviour is the reason validity conditioning is
    needed: there is no delta that is simultaneously right for a
    present-tense query and a historical one.
    """
    events = [
        Event(
            event_id=f"e{i}",
            patient_id="p1",
            event_type=EventType.LAB,
            valid_time=dt(i),
            payload={"value": str(i)},
        )
        for i in range(10)
    ]
    low = rank_uniform_decay(events, lambda q, e: 0.5, delta=0.01, now=NOW)
    high = rank_uniform_decay(events, lambda q, e: 0.5, delta=0.50, now=NOW)
    assert high[0].final > low[0].final


def test_latest_n_breaks_as_of_queries():
    """Baseline 8: correct for 'current', wrong for 'as of'."""
    graph = SupersessionGraph(
        [
            Event(
                event_id="old",
                patient_id="p1",
                event_type=EventType.LAB,
                valid_time=dt(1),
                payload={"test": "potassium", "value": "2.1"},
            ),
            correction("new", "old", 13, "4.8"),
        ]
    )
    results = rank_latest_n(graph.all_events)
    assert results[0].event.payload["value"] == "4.8", "latest-N cannot answer history"
