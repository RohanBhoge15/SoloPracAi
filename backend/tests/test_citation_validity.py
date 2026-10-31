"""Citation badges must agree with the ranking that produced the answer.

Regression tests for a defect found in review: the citation badge computed
validity with `graph.state(event)` â€” always the present tense â€” while the
retriever one layer down resolved it with `graph.state_as_of(event, instant)`
against the instant that was asked about.

The consequence was specific and bad. Ask "what was her potassium as of
<before the correction>" and the pipeline correctly answered with the
*original* value, because that record was current at the asked instant. The
citation chip then rendered amber and labelled `superseded` â€” telling the
doctor that the record they had just been shown was retracted, when it was
in fact the right answer to their question.

That is the last mile of the failure this project exists to prevent: the
retrieval is instant-aware and the presentation was not.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.api.routes.chat import _citation_validity
from app.core.validity import Event, EventType, SourceType, Validity
from app.services.pipeline import build_context
from app.services.pii import Deidentifier
from app.services.pipeline import AnswerPipeline
from app.services.retrieval import TemporalIntent

BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


def dt(days: float) -> datetime:
    return BASE + timedelta(days=days)


# The original is drawn on day 11. The correction supersedes it on day 13.
ORIGINAL_DAY = 11.0
CORRECTION_DAY = 13.0

BEFORE = dt(ORIGINAL_DAY + 0.5)   # after the original, before the correction
AFTER = dt(CORRECTION_DAY + 0.5)   # after the correction


@pytest.fixture
def context():
    events = [
        Event(
            event_id="e-orig",
            patient_id="p1",
            event_type=EventType.LAB,
            valid_time=dt(ORIGINAL_DAY),
            payload={"test": "potassium", "value": 2.1, "unit": "mEq/L"},
        ),
        Event(
            event_id="e-corr",
            patient_id="p1",
            event_type=EventType.LAB,
            valid_time=dt(CORRECTION_DAY),
            payload={"test": "potassium", "value": 4.8, "unit": "mEq/L"},
            source_type=SourceType.CORRECTION,
            supersedes_id="e-orig",
        ),
    ]
    deid = Deidentifier(
        patient_id="p1", patient_name="Rakesh Mehta", age=54, phone="9876543210"
    )
    return build_context("p1", events, deid, display_name="Rakesh Mehta", age=54)


def intent(kind: str, instant: datetime | None) -> TemporalIntent:
    return TemporalIntent(kind=kind, instant=instant, confidence=1.0, cue="test")


def event(context, eid: str) -> Event:
    return next(e for e in context.events if e.event_id == eid)


# â”€â”€ the defect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_original_cited_before_the_correction_is_current(context):
    """The record that was current when the question was asked is `current`.

    This is the assertion that failed before the fix. The original is
    superseded *now*, and the badge said so.
    """
    original = event(context, "e-orig")
    got = _citation_validity(original, context, intent("as_of", BEFORE))
    assert got == "current", (
        "a record that was current at the queried instant must not be "
        "badged superseded"
    )


def test_original_cited_after_the_correction_is_superseded(context):
    """Asking after the correction, the same record is superseded.

    Same event, different instant. If this fails, the badge is ignoring the
    instant entirely and the first test passes only by accident.
    """
    original = event(context, "e-orig")
    got = _citation_validity(original, context, intent("as_of", AFTER))
    assert got == "superseded"


def test_correction_is_current_at_both_instants(context):
    correction = event(context, "e-corr")
    assert _citation_validity(correction, context, intent("as_of", AFTER)) == "current"
    # Before it existed, a valid-time filter would have removed it rather
    # than superseded it; the badge reports validity, not visibility.
    assert _citation_validity(correction, context, intent("as_of", BEFORE)) == "current"


# â”€â”€ present tense must be unchanged â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_present_tense_reports_the_present(context):
    """No instant, or a present-tense intent: fall back to `state`.

    A doctor's untensed question gets the present-tense reading, which is
    the common clinical case.
    """
    original = event(context, "e-orig")
    correction = event(context, "e-corr")
    assert _citation_validity(original, context, intent("current", None)) == "superseded"
    assert _citation_validity(correction, context, intent("current", None)) == "current"


def test_none_instant_is_treated_as_present_tense(context):
    """`instant=None` with an as_of kind must not blow up or invert."""
    original = event(context, "e-orig")
    assert _citation_validity(original, context, intent("as_of", None)) == "superseded"


def test_historical_intent_also_resolves_as_of(context):
    """`historical` carries an instant too and uses the same rule."""
    original = event(context, "e-orig")
    assert _citation_validity(original, context, intent("historical", BEFORE)) == "current"
    assert _citation_validity(original, context, intent("historical", AFTER)) == "superseded"


# â”€â”€ the shared resolver, and every label derived from it â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_resolve_validity_matches_the_reported_label(context):
    """`resolve_validity` is the one definition of "valid" for a request.

    The weight and the label used to be computed by different calls, so an
    as-of query ranked the correct record and then reported it as superseded.
    Anything that labels a result must go through this.
    """
    from app.services.retrieval import resolve_validity

    original = event(context, "e-orig")
    for i in (intent("as_of", BEFORE), intent("as_of", AFTER), intent("current", None)):
        state = resolve_validity(original, i, context.graph)
        badge = _citation_validity(original, context, i)
        assert (state is Validity.SUPERSEDED) == (badge == "superseded"), (
            f"resolver and label disagree for {i.kind}"
        )


async def test_retriever_reports_as_of_validity_not_present_validity(context):
    """`ScoredEvent.validity` must be the resolved state, not `state()`.

    This is the defect the semantic-search E2E caught: the *ranking* was
    correct and the *label* on the same record was not, so an as-of query
    returned the right record marked "superseded".
    """
    from app.services.retrieval import resolve_validity

    pipeline = AnswerPipeline(llm=None, semantic_fn=lambda q, e: 0.9, k=5)
    result = await pipeline.answer(
        context, "what was her potassium as of february", as_of=BEFORE
    )
    cited = {c["event_id"] for c in result.answer.citations}
    assert "e-orig" in cited, "the record current at the instant must be cited"

    # And the state the resolver reports for it must be CURRENT, even though
    # the present-tense state is SUPERSEDED.
    as_of_intent = TemporalIntent(
        kind="as_of", instant=BEFORE, confidence=1.0, cue="test"
    )
    original = event(context, "e-orig")
    assert context.graph.state(original) is Validity.SUPERSEDED
    assert resolve_validity(original, as_of_intent, context.graph) is Validity.CURRENT


# â”€â”€ agreement with the retriever â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_badge_agrees_with_the_retrievers_own_weight(context):
    """The badge must not contradict the weight used to rank the record.

    This is the invariant that actually matters: whatever weight
    `validity_weight` assigned under an intent, the badge must describe the
    same record the same way. If they diverge, the UI is describing a
    ranking it did not produce.
    """
    from app.services.retrieval import validity_weight

    original = event(context, "e-orig")
    correction = event(context, "e-corr")

    for instant, kind in ((BEFORE, "as_of"), (AFTER, "as_of")):
        i = intent(kind, instant)
        for ev in (original, correction):
            weight = validity_weight(ev, i, context.graph)
            badge = _citation_validity(ev, context, i)
            if weight >= 1.0:            # W_AS_OF â€” the best weight available
                assert badge == "current", (
                    f"{ev.event_id} ranked at full weight but badged {badge}"
                )
            elif weight == 0.15:        # W_SUPERSEDED
                assert badge == "superseded", (
                    f"{ev.event_id} demoted but badged {badge}"
                )


