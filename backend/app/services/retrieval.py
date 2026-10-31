"""Validity-aware retrieval — the method behind R2.

Scores candidates as `alpha * semantic + w_state`, where `w_state` is a
*categorical* function of validity state, not a continuous decay in delta-t.

Why categorical: a lab corrected 2 hours ago and one corrected 2 years ago
are equally invalid for a present-tense question. No continuous decay can
represent that. Any function of delta-t trades staleness against historical
recall along a single dial; this has no dial, because the signal is graph
reachability rather than time distance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.validity import Event, SupersessionGraph, Validity

# ── scoring weights ──────────────────────────────────────────────────

#: Weight for an event that is CURRENT when the query has no temporal anchor.
W_CURRENT = 0.85
#: Weight for a SUPERSEDED event when the query has no temporal anchor.
#: Demoted, not discarded: the generator still needs to know it was corrected.
W_SUPERSEDED = 0.15
#: Weight when the event is valid at the explicitly requested instant,
#: whether or not it is current now. This is what preserves historical recall.
W_AS_OF = 1.00
#: Weight when the event post-dates the requested instant.
W_FUTURE = -1.0  # hard filter, applied as a veto rather than a penalty

ALPHA_SEMANTIC = 0.70


@dataclass(frozen=True, slots=True)
class TemporalIntent:
    """Parsed intent: is the question about now, the past, or a point in time?"""

    kind: str  # "current" | "historical" | "as_of"
    instant: Optional[datetime] = None
    confidence: float = 0.0
    cue: str = ""


_AS_OF_PATTERNS = (
    r"\bas\s+of\b",
    r"\bon\s+\d{1,2}\s+\w+\s+\d{4}",
    r"\bin\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    r"\bback\s+then\b",
    r"\bthat\s+time\b",
    # Relative offsets. These need the trailing direction word, so a bare
    # duration ("on this drug for 3 days") cannot match and become a
    # time-travel request it was never asking for.
    r"\b(?:\d{1,4}|[a-z]+)\s+"
    r"(?:day|week|fortnight|month|quarter|year)s?\s+"
    r"(?:ago|back|earlier|before)\b",
    r"\byesterday\b",
    r"\blast\s+(?:week|month|year|night)\b",
)

_HISTORICAL_CUES = (
    "initial",
    "first",
    "original",
    "baseline",
    "previously",
    "before",
    "earlier",
    "formerly",
    "prior to",
    "at the time",
    "used to",
    "history of",
    "when first",
    "when was",
    "when did",
    "ever started",
    "ever prescribed",
    "started on",
    "began",
    "initiated",
    "on file",
    "originally",
)

_CURRENT_CUES = (
    "current",
    "currently",
    "latest",
    "most recent",
    "newest",
    "now",
    "present",
    "today",
    "right now",
    "at present",
    "these days",
    "ongoing",
    "still",
)

_TREND_CUES = (
    "trend",
    "over time",
    "progression",
    "course",
    "changed",
    "change",
    "history",
    "trajectory",
    "evolved",
    "compared to",
    "versus",
    "improved",
    "worsened",
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        ],
        start=1,
    )
}


def parse_temporal_intent(query: str, now: Optional[datetime] = None) -> TemporalIntent:
    """Classify a query's temporal intent.

    Deterministic and dependency-free on the live path. An LLM classifier is
    an optional upgrade, not a requirement — the previous prototype made the
    whole retriever depend on a classifier trained on 100 hand-labelled
    queries, which is not a defensible foundation.
    """
    now = now or datetime.now(timezone.utc)
    q = (query or "").lower().strip()

    if not q:
        return TemporalIntent(kind="current", instant=now, confidence=0.0)

    for pattern in _AS_OF_PATTERNS:
        match = re.search(pattern, q)
        if match:
            instant = _resolve_instant(q, now)
            return TemporalIntent(
                kind="as_of",
                instant=instant or now,
                confidence=0.9,
                cue=match.group(0),
            )

    # Order matters: "most recent change" is current, not historical.
    for cue in _CURRENT_CUES:
        if cue in q:
            return TemporalIntent(kind="current", instant=now, confidence=0.8, cue=cue)

    for cue in _HISTORICAL_CUES:
        if cue in q:
            return TemporalIntent(kind="historical", instant=None, confidence=0.8, cue=cue)

    for cue in _TREND_CUES:
        if cue in q:
            return TemporalIntent(
                kind="historical", instant=now, confidence=0.6, cue=cue
            )

    # No temporal cue: default to current, which is the common clinical case.
    return TemporalIntent(kind="current", instant=now, confidence=0.5, cue="")


#: Relative offsets, in days. Words a clinician actually uses when they
#: ask about a past state.
_RELATIVE_UNITS = {
    "day": 1,
    "week": 7,
    "fortnight": 14,
    "month": 30,
    "quarter": 91,
    "year": 365,
}

#: "two days ago", "a week ago", "3 weeks back".
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
    "couple": 2, "few": 3, "several": 4,
}

#: Bare time units with no number, e.g. "as of yesterday".
_BARE_OFFSETS = {
    "yesterday": 1,
    "today": 0,
    "last week": 7,
    "last month": 30,
    "last year": 365,
}


def _resolve_relative(query: str, now: datetime) -> Optional[datetime]:
    """Resolve "as of 60 days ago" and friends.

    Without this, a relative as-of question is answered as of *now* — and
    it is answered confidently, which is the worst possible failure for a
    system whose whole claim is temporal awareness. "What was her HbA1c
    as of 60 days ago" has no absolute date in it, so the date patterns
    find nothing and the query silently resolves to the present.

    Returns None when there is no relative expression, so the caller can
    fall through to the absolute patterns.
    """
    q = query.lower()

    for phrase, days in _BARE_OFFSETS.items():
        if phrase in q:
            return now - timedelta(days=days)

    # "60 days ago", "3 weeks back", "two months before"
    match = re.search(
        r"\b(\d{1,4}|[a-z]+)\s+"
        r"(day|week|fortnight|month|quarter|year)s?\s+"
        r"(ago|back|before|earlier)\b",
        q,
    )
    if not match:
        return None

    raw = match.group(1)
    unit = _RELATIVE_UNITS.get(match.group(2))
    if unit is None:
        return None

    if raw.isdigit():
        count = int(raw)
    else:
        count = _WORD_NUMBERS.get(raw)
        if count is None:
            return None

    # A count of zero is not a time travel request, and treating "0 days
    # ago" as a past instant would be odd.
    if count <= 0:
        return now
    return now - timedelta(days=count * unit)


def _resolve_instant(query: str, now: datetime) -> Optional[datetime]:
    """Best-effort parse of the instant an as-of question refers to.

    An explicit calendar date is always preferred over a relative
    expression: "as of 2025-06-15, 10 days ago" is pinned to 15 June, and
    resolving the relative half instead would silently answer about a
    different day. Relative is only consulted when no date is present,
    which is the case that matters for "as of 60 days ago".
    """
    absolute = _resolve_absolute(query, now)
    if absolute is not None:
        return absolute
    return _resolve_relative(query, now)


def _resolve_absolute(query: str, now: datetime) -> Optional[datetime]:
    """ISO, then day-month-year, then month-day-year."""
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", query)
    if iso:
        try:
            return datetime(
                int(iso.group(1)), int(iso.group(2)), int(iso.group(3)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            return now

    for pattern in (r"\b(\d{1,2})\s+(\w{3,9})\s+(\d{4})", r"\b(\w{3,9})\s+(\d{1,2}),?\s*(\d{4})"):
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            continue
        groups = [g.lower() for g in match.groups()]
        month = next((_MONTHS[g[:3]] for g in groups if g[:3] in _MONTHS), None)
        if month is None:
            continue
        day = next((int(g) for g in groups if g.isdigit() and int(g) <= 31), 1)
        year = next((int(g) for g in groups if g.isdigit() and len(g) == 4), now.year)
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ── default semantic scorer ──────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9.]+")

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "is", "was", "are", "were", "to", "in", "on",
        "at", "for", "and", "or", "what", "which", "patient", "s", "this",
        "that", "with", "by", "from", "as", "it", "be", "has", "have", "had",
    }
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS}


def lexical_similarity(query: str, event: Event) -> float:
    """Token-overlap similarity, with a clinical-term boost.

    The default when no embedding model is loaded. It keeps the system
    fully functional out of the box and — more importantly — lets the
    TAP-QA harness isolate the *temporal* contribution from the
    *embedding* contribution, since the two cannot interact when the
    semantic function is fixed and trivial.

    `query` is empty for the baseline helpers, which pass through a
    pre-computed score instead.
    """
    q = _tokens(query)
    if not q:
        return 0.0
    payload = " ".join(
        f"{k} {v}" for k, v in (event.payload or {}).items()
    )
    d = _tokens(f"{payload} {event.event_type}")
    if not d:
        return 0.0
    overlap = len(q & d)
    if not overlap:
        return 0.0
    # Normalised overlap, with a floor so an exact test-name match ranks
    # clearly above incidental word overlap.
    return min(1.0, 0.35 + 0.65 * (overlap / len(q)))


def resolve_validity(
    event: Event, intent: TemporalIntent, graph: SupersessionGraph
) -> Validity:
    """Validity of `event` **as of the instant the query asked about**.

    This is the single place that decides what "valid" means for a request,
    and both the ranking weight and the reported label come from it. They
    used to be computed separately - the weight via `state_as_of`, the label
    via `state` - so an as-of query ranked the right record and then reported
    it as superseded. One function, so they cannot drift.
    """
    if intent.kind == "as_of" and intent.instant is not None:
        return graph.state_as_of(event, intent.instant)
    # A `historical` query with no explicit instant asks for the oldest match,
    # where present-tense validity is not the deciding factor.
    return graph.state(event)


def validity_weight(
    event: Event, intent: TemporalIntent, graph: SupersessionGraph
) -> float:
    """Categorical validity weight for one event under one intent."""
    if intent.kind in ("as_of", "historical"):
        # W_AS_OF, not W_CURRENT: a record that is superseded now but was
        # current then is the correct answer, and must outrank everything.
        return W_AS_OF if resolve_validity(event, intent, graph) is Validity.CURRENT else W_SUPERSEDED

    return W_CURRENT if resolve_validity(event, intent, graph) is Validity.CURRENT else W_SUPERSEDED


def age_years(valid_time: datetime, now: datetime) -> float:
    return max(0.0, (now - valid_time).total_seconds() / 31_536_000.0)


@dataclass(slots=True)
class ScoredEvent:
    event: Event
    semantic: float
    validity: Validity
    weight: float
    final: float
    excluded_reason: Optional[str] = None
    rank: int = 0

    @property
    def event_id(self) -> str:
        return self.event.event_id

    @property
    def is_citable(self) -> bool:
        return self.excluded_reason is None


@dataclass
class ValidityAwareRetriever:
    """Scores events by semantic similarity modulated by validity state.

    `semantic_fn` is injected so the retriever can be evaluated without
    an embedding model — which is exactly what the TAP-QA harness needs to
    isolate the *temporal* contribution from the *embedding* contribution.
    """

    graph: SupersessionGraph
    alpha: float = ALPHA_SEMANTIC
    enforce_future_leak: bool = True

    def rank(
        self,
        query: str,
        patient_id: str,
        candidates: Sequence[Event],
        intent: Optional[TemporalIntent] = None,
        now: Optional[datetime] = None,
        semantic_fn=None,
        k: int = 8,
    ) -> List[ScoredEvent]:
        now = now or datetime.now(timezone.utc)
        intent = intent or parse_temporal_intent(query, now)
        graph = self.graph

        scored: List[ScoredEvent] = []
        for event in candidates:
            if event.patient_id != patient_id:
                # Defence in depth. Retrieval should already be filtered; if
                # a candidate from another patient appears, it is dropped
                # rather than scored.
                continue

            # Hard veto: a present-tense or as-of query must never surface a
            # record that post-dates the question. This is the future-leak
            # constraint, and it is a filter, not a penalty.
            if self.enforce_future_leak and intent.instant is not None:
                if event.valid_time > intent.instant:
                    continue

            semantic = (
                semantic_fn(query, event) if semantic_fn else 0.0
            )
            weight = validity_weight(event, intent, graph)
            final = self.alpha * semantic + (1.0 - self.alpha) * weight

            scored.append(
                ScoredEvent(
                    event=event,
                    semantic=semantic,
                    # Same resolution the weight came from, so the label a
                    # client reads cannot contradict the ranking it caused.
                    validity=resolve_validity(event, intent, graph),
                    weight=weight,
                    final=final,
                )
            )

        # Tiebreak matters. For an as-of query every valid record scores the
        # same validity weight, so a plain recency tiebreak returns the most
        # recent record rather than the one that was current *then*. Proximity
        # to the requested instant is the correct tiebreak.
        if intent.kind == "as_of" and intent.instant is not None:
            target = intent.instant.timestamp()
            scored.sort(
                key=lambda s: (
                    -s.final,
                    abs(s.event.valid_time.timestamp() - target),
                )
            )
        else:
            scored.sort(key=lambda s: (-s.final, -s.event.valid_time.timestamp()))

        for position, item in enumerate(scored, start=1):
            item.rank = position
        return scored[:k]


# ── baselines, kept beside the method so comparison is auditable ──────


def rank_dense_only(
    candidates: Sequence[Event], semantic_fn, k: int = 8
) -> List[ScoredEvent]:
    """Baseline 2: pure semantic similarity, no temporal term at all.

    This is the system SoloPrac is arguing with - the naive approach that
    quotes a retracted value with perfect citations.
    """
    scored = [
        ScoredEvent(
            event=e,
            semantic=semantic_fn("", e),
            validity=Validity.CURRENT,
            weight=1.0,
            final=semantic_fn("", e),
        )
        for e in candidates
    ]
    scored.sort(key=lambda s: -s.final)
    for position, item in enumerate(scored, start=1):
        item.rank = position
    return scored[:k]


def rank_uniform_decay(
    candidates: Sequence[Event],
    semantic_fn,
    delta: float,
    now: Optional[datetime] = None,
    tau_days: float = 30.0,
    k: int = 8,
) -> List[ScoredEvent]:
    """Baseline 5: continuous recency decay.

    Retained deliberately, including its known pathology. The previous
    prototype used delta=0.15 as a live default while its own paper proved
    that value collapses R@1 to 14%. It is here as a *measured baseline*,
    swept over a grid — never as a production default.
    """
    now = now or datetime.now(timezone.utc)
    scored: List[ScoredEvent] = []
    for event in candidates:
        semantic = semantic_fn("", event)
        days = max(0.0, (now - event.valid_time).total_seconds() / 86_400.0)
        decay = math.exp(-days / tau_days)
        scored.append(
            ScoredEvent(
                event=event,
                semantic=semantic,
                validity=Validity.CURRENT,
                weight=1.0,
                final=semantic + delta * decay,
            )
        )
    scored.sort(key=lambda s: -s.final)
    for position, item in enumerate(scored, start=1):
        item.rank = position
    return scored[:k]


def rank_latest_n(candidates: Sequence[Event], k: int = 8) -> List[ScoredEvent]:
    """Baseline 8: most recent first. Correct for 'current', destroys 'as of'."""
    scored = [
        ScoredEvent(
            event=e,
            semantic=0.0,
            validity=Validity.CURRENT,
            weight=1.0,
            final=e.valid_time.timestamp(),
        )
        for e in candidates
    ]
    scored.sort(key=lambda s: -s.final)
    for position, item in enumerate(scored, start=1):
        item.rank = position
    return scored[:k]
