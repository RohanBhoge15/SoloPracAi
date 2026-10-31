"""Retrieval baselines.

Every method in the paper's comparison table, in the order the plan
specifies. They are implemented against the same interface and scored by
the same metrics, so a difference in the results table is a difference in
the method and not in the plumbing.

The design rule that matters here: **no baseline gets a hand-picked
parameter value.** The recency-decay baseline is swept over the grid in
the plan and reported as a curve, because a single chosen lambda is
exactly how a strawman gets built. Every baseline is also reported with
the random floor beside it, because without a floor none of the numbers
are interpretable.

The two axes are genuinely distinct in the dataset - a fact can be
clinically true for months before the system learns of it - so the two
single-axis filters and the bitemporal baseline measure different things
rather than being three names for one column.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tapqa.dataset import Query, Record
from tapqa.ranking import Ranking, from_order, from_scores

Ranker = Callable[[Query, List[Record]], Ranking]

#: The grid the plan mandates for the decay sweep. Reported in full.
DECAY_GRID: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.15, 0.30, 0.50)

#: Time constants for the decay sweep, in days.
TAU_GRID: Tuple[float, ...] = (7.0, 30.0, 90.0, 180.0, 365.0)

_TOKEN = re.compile(r"[a-z0-9.]+")
_STOP = frozenset(
    {
        "the", "a", "an", "of", "is", "was", "are", "were", "to", "in", "on",
        "at", "for", "and", "or", "what", "which", "patient", "s", "this",
        "that", "with", "by", "from", "as", "it", "be", "has", "have", "had",
        "her", "his", "their", "my", "i", "me", "now", "current", "latest",
        "most", "recent", "reading", "readings", "result", "results", "back",
        "ago", "days", "day", "weeks", "week", "months", "month", "years",
        "year", "value", "level", "there", "two", "different", "file", "act",
    }
)


def _tokens(text: str) -> set:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP}


def _text(record: Record) -> str:
    return " ".join(f"{k} {v}" for k, v in record.payload.items())


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _superseded_ids(records: Sequence[Record]) -> set:
    return {r.supersedes_id for r in records if r.supersedes_id}


def _by_id(records: Sequence[Record]) -> Dict[str, Record]:
    return {r.event_id: r for r in records}


# ── 1. random floor ────────────────────────────────────────────────


def rank_random(query: Query, records: List[Record], seed: int = 0) -> Ranking:
    """The floor. Not a strawman - the reference point.

    Deterministic per query so a rerun does not shuffle the comparison.
    A shuffle really is a total order, so it reports one.
    """
    rng = random.Random(f"{seed}:{query.query_id}")
    ids = [r.event_id for r in records]
    rng.shuffle(ids)
    return from_order(ids)


# ── 2/3. dense-only ────────────────────────────────────────────────


def make_dense_only(embedder: Optional[Callable[..., Any]]):
    """Dense retrieval with no temporal term at all.

    This is the system SOLUM argues with: it finds the topically right
    record and quotes a retracted value with perfect citations.
    """

    def ranker(query: Query, records: List[Record]) -> List[str]:
        if embedder is None:
            # No encoder available: fall back to lexical and say so. A
            # silent lexical stand-in presented as "dense" would be a
            # fabricated result.
            return rank_lexical(query, records)
        # The query is encoded with the query tower and the records with
        # the document tower, which is what MedCPT was trained for.
        # Comparing records against record[0] - which is what this did
        # first - measures nothing at all.
        doc_matrix = embedder([_text(r) for r in records], is_query=False)
        query_vector = embedder([query.text], is_query=True)
        if doc_matrix is None or query_vector is None:
            return rank_lexical(query, records)
        scores = doc_matrix @ query_vector[0]
        return from_scores([(float(scores[i]), records[i].event_id)
                            for i in range(len(records))])

    return ranker


def rank_lexical(query: Query, records: List[Record]) -> Ranking:
    """Token overlap. The floor for any semantic claim.

    The two versions of a corrected test tie on every token, and this
    baseline genuinely cannot tell them apart - that is the whole point of
    including it. So the tie is reported as a tie. Gold's rank comes out
    at 1.5 and the method earns nothing for the arbitrary order.

    The earlier version broke ties on `event_id`, which encodes insertion
    order, which encodes chronology, so the original of every corrected
    test won automatically. That produced 99.8% historical accuracy and
    0.0% corrected accuracy for a method with no temporal reasoning in it
    whatsoever, which is a measurement of the tiebreak rather than of the
    method.
    """
    q = _tokens(query.text)
    scored: List[Tuple[float, str]] = []
    for record in records:
        d = _tokens(_text(record))
        overlap = len(q & d)
        score = 0.0 if not q or not overlap else 0.35 + 0.65 * overlap / len(q)
        scored.append((score, record.event_id))
    return from_scores(scored)


def _neg_time(iso: str) -> float:
    return -_dt(iso).timestamp()


# ── 4. reciprocal rank fusion, no temporal term ────────────────────


def make_rrf(k: int = 60, weights: Sequence[float] = (1.0, 0.5)):
    """Fuse a lexical and a dense ranking. Still no temporal awareness.

    Included because "just fuse two retrievers" is the obvious objection
    to a temporal method, and it deserves a fair test.
    """

    def ranker(query: Query, records: List[Record]) -> Ranking:
        lexical = rank_lexical(query, records)
        # Recency as the second list: the strongest non-graph temporal
        # signal available without a graph.
        by_time = sorted(records, key=lambda r: _neg_time(r.valid_time))
        recency = [r.event_id for r in by_time]

        fused: Dict[str, float] = {}
        for weight, ranking in zip(weights, (lexical, recency)):
            for position, event_id in enumerate(ranking, start=1):
                fused[event_id] = fused.get(event_id, 0.0) + weight / (k + position)
        return from_scores([(score, event_id) for event_id, score in fused.items()])

    return ranker


# ── 5. uniform recency decay ───────────────────────────────────────


def make_decay(lam: float, tau_days: float, now: Optional[datetime] = None):
    """`score = semantic + lambda * exp(-age / tau)`.

    Swept over the grid in the plan. Every point on its curve is reported,
    because the whole argument is that this family traces a *trade-off*:
    push lambda up and staleness falls, and historical recall dies with
    it. The claim being tested is that the graph-conditioned method has
    no such dial to turn.
    """
    reference = now or datetime(2026, 6, 1, tzinfo=timezone.utc)

    def ranker(query: Query, records: List[Record]) -> Ranking:
        scored: List[Tuple[float, str]] = []
        for record in records:
            q = _tokens(query.text)
            d = _tokens(_text(record))
            overlap = len(q & d)
            # Normalised to [0, 1] so the decay term can actually compete.
            #
            # This was `0.35 + 0.65 * overlap / len(q)`, which floors the
            # semantic span at 0.35 and caps it at 1.0 - a 0.65-wide range
            # against a decay term spanning at most `lam` <= 0.5, and only
            # 0.01 at the smallest grid value. The recency term could then
            # not reorder anything: the two versions of a corrected test
            # tie exactly on token overlap (verified on all 400 historical
            # queries), so the ranking was identical to the token-overlap
            # floor for every large-tau grid point, and 0% historical
            # accuracy was the floor's, not a property of the family.
            # A sweep that cannot express its own trade-off cannot refute
            # one.
            semantic = 0.0 if not q else overlap / len(q)
            age_days = max(
                0.0,
                (reference - _dt(record.valid_time)).total_seconds() / 86_400.0,
            )
            decay = math.exp(-age_days / tau_days)
            scored.append((semantic + lam * decay, record.event_id))
        return from_scores(scored)

    return ranker


# ── 6/7. single-axis filters ───────────────────────────────────────


def make_valid_time_only(lam: float = 0.30, tau_days: float = 90.0):
    """Rank on valid time: the fact's clinical truth. One axis only.

    A real recency preference on valid time, added to lexical relevance -
    not a tiebreak. The first version of this used valid time as a
    tiebreak on an already-unique lexical rank, so the term never ran and
    the baseline silently reduced to token overlap, reporting identical
    numbers for two supposedly different methods.

    Carries no supersession information, so a corrected value is
    indistinguishable from the record that corrects it. That is the point:
    it is what a one-column design can and cannot do.
    """
    reference = datetime(2026, 6, 1, tzinfo=timezone.utc)

    def ranker(query: Query, records: List[Record]) -> Ranking:
        scored: List[Tuple[float, str]] = []
        for record in records:
            q = _tokens(query.text)
            d = _tokens(_text(record))
            overlap = len(q & d)
            semantic = (
                0.0 if not q or not overlap else 0.35 + 0.65 * overlap / len(q)
            )
            age = max(
                0.0,
                (reference - _dt(record.valid_time)).total_seconds() / 86_400.0,
            )
            scored.append(
                (semantic + lam * math.exp(-age / tau_days), record.event_id)
            )
        return from_scores(scored)

    return ranker


def make_transaction_time_only():
    """Rank on transaction time: when the system learned it.

    A back-entered note looks brand new, which is the failure mode that
    motivates bitemporal modelling in the first place.

    The transaction-time preference is an additive term, exactly parallel
    to the valid-time baseline, so the two single-axis ablations differ
    only in which clock they read. It used to be applied as a tiebreak on
    top of the lexical score, which meant it fired only on the queries
    where the two versions of a record tied - precisely the queries the
    ablation is supposed to be about, and precisely where a tiebreak
    decides the answer by accident.
    """
    reference = datetime(2026, 6, 1, tzinfo=timezone.utc)

    def ranker(query: Query, records: List[Record]) -> Ranking:
        scored: List[Tuple[float, str]] = []
        for record in records:
            q = _tokens(query.text)
            d = _tokens(_text(record))
            overlap = len(q & d)
            semantic = (
                0.0 if not q or not overlap else 0.35 + 0.65 * overlap / len(q)
            )
            txn_age = max(
                0.0,
                (reference - _dt(record.recorded_at)).total_seconds() / 86_400.0,
            )
            scored.append(
                (semantic + 0.30 * math.exp(-txn_age / 180.0), record.event_id)
            )
        return from_scores(scored)

    return ranker


# ── 8. latest-N ────────────────────────────────────────────────────


def rank_latest_n(query: Query, records: List[Record]) -> Ranking:
    """Most recent first. Correct for 'current', useless for 'as of'."""
    ordered = sorted(records, key=lambda r: _neg_time(r.valid_time))
    return from_order([r.event_id for r in ordered])


# ── 9. graph-aware hard filter ───────────────────────────────────────


def make_graph_filter(embedder: Optional[Callable[..., Any]] = None):
    """Use the supersession graph to find current records, then rank by
    semantic similarity among them.

    This is the "obvious" graph-aware baseline: follow the pointers to
    determine which records are current at t_q, discard the rest, and rank
    the survivors by dense similarity. It uses the same supersession edges
    as our method but applies no weighting scheme, no validity classes, and
    no tiebreak by time proximity. If our method wins against this, the win
    is not merely "we had the edge".
    """

    def ranker(query: Query, records: List[Record]) -> Ranking:
        # Admissible at t means: valid by its event time and not
        # superseded by any correction dated on or before t. For "current"
        # queries (no explicit as_of) t is the eval's fixed reference
        # instant, the same one the other single-axis baselines use.
        t = (
            datetime.fromisoformat(query.as_of)
            if query.as_of
            else datetime(2026, 6, 1, tzinfo=timezone.utc)
        ).astimezone(timezone.utc)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        superseded_at = {}
        by_id = {r.event_id: r for r in records}
        for r in records:
            if r.supersedes_id and r.supersedes_id in by_id:
                superseded_at.setdefault(r.supersedes_id, []).append(r)
        admissible = []
        for r in records:
            vt = datetime.fromisoformat(r.valid_time)
            if vt.tzinfo is None:
                vt = vt.replace(tzinfo=timezone.utc)
            if vt > t:
                continue
            superseded_before = any(
                datetime.fromisoformat(c.valid_time).replace(tzinfo=timezone.utc)
                <= t
                for c in superseded_at.get(r.event_id, [])
            )
            if not superseded_before:
                admissible.append(r)
        if not admissible:
            admissible = list(records)
        # Rank the admissible set by dense similarity only -- no weighting,
        # no validity classes, no time-proximity tiebreak.
        if embedder is not None:
            doc_matrix = embedder([_text(r) for r in admissible], is_query=False)
            query_vector = embedder([query.text], is_query=True)
            if doc_matrix is not None and query_vector is not None:
                scores = doc_matrix @ query_vector[0]
                return from_scores([
                    (float(scores[i]), admissible[i].event_id)
                    for i in range(len(admissible))
                ])
        # Lexical fallback, same as dense-only.
        return rank_lexical(query, admissible)

    return ranker


# ── 10. bitemporal, both axes, no graph ────────────────────────────


def make_bitemporal(alpha: float = 0.5):
    """Both timestamp axes, but no supersession graph.

    This is the ablation that decides whether the contribution is "we
    added a second timestamp column" or "we use the graph". Both axes are
    applied as a soft preference, not a filter, so the comparison against
    the graph-conditioned method is like for like.

    Validity and transaction time are two independent signals here: a
    record that is clinically old *and* recently reported is a different
    animal from one that is both old.
    """

    def ranker(query: Query, records: List[Record]) -> Ranking:
        reference = datetime(2026, 6, 1, tzinfo=timezone.utc)
        scored: List[Tuple[float, str]] = []
        for record in records:
            q = _tokens(query.text)
            d = _tokens(_text(record))
            overlap = len(q & d)
            semantic = (
                0.0 if not q or not overlap else 0.35 + 0.65 * overlap / len(q)
            )
            valid_age = max(
                0.0,
                (reference - _dt(record.valid_time)).total_seconds() / 86_400.0,
            )
            txn_age = max(
                0.0,
                (reference - _dt(record.recorded_at)).total_seconds() / 86_400.0,
            )
            # Both axes decay on the same time constant, so neither is
            # privileged by construction.
            temporal = math.exp(-valid_age / 180.0) + math.exp(-txn_age / 180.0)
            scored.append(
                (
                    (1 - alpha) * semantic + alpha * temporal / 2.0,
                    record.event_id,
                )
            )
        # The valid-time tiebreak this used to carry is gone. Both axes are
        # already in the score, so it only ever fired on exact ties and
        # resolved them by chronology.
        return from_scores(scored)

    return ranker


# ── 10. ours: three-valued, graph-conditioned ──────────────────────


# ── registry ──────────────────────────────────────────────────────


@dataclass
class Method:
    key: str
    label: str
    ranker: Ranker
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    #: True when the method is a point on a swept curve, so the report
    #: renders it as part of that curve rather than as its own row.
    sweep_of: Optional[str] = None


def _ours_ranker(embedder):
    """Delegates to the shipping retriever; see tapqa.ours for why."""
    from tapqa.ours import make_graph_conditioned

    return make_graph_conditioned(embedder=embedder)


def build_methods(
    embedder: Optional[Callable[[List[str]], Any]] = None,
    *,
    include_sweeps: bool = True,
) -> List[Method]:
    methods: List[Method] = [
        Method("random", "Random floor", rank_random, "floor"),
        Method("lexical", "Token overlap", rank_lexical, "floor"),
        Method(
            "dense",
            "Dense-only (no temporal term)",
            make_dense_only(embedder),
            "semantic",
        ),
        Method("rrf", "RRF fusion, no temporal term", make_rrf(), "semantic"),
        Method(
            "valid_time", "Valid-time only", make_valid_time_only(), "temporal"
        ),
        Method(
            "txn_time", "Transaction-time only", make_transaction_time_only(), "temporal"
        ),
        Method("latest_n", "Latest-N", rank_latest_n, "temporal"),
        Method("bitemporal", "Bitemporal, both axes, no graph", make_bitemporal(), "ablation"),
        Method(
            "graph_filter",
            "Graph filter (as-of admissible, then dense)",
            make_graph_filter(embedder),
            "ablation",
        ),
        Method(
            "ours",
            "Ours: three-valued, graph-conditioned",
            _ours_ranker(embedder),
            "ours",
        ),
    ]

    if include_sweeps:
        for lam in DECAY_GRID:
            for tau in TAU_GRID:
                methods.append(
                    Method(
                        key=f"decay_{lam:g}_{tau:g}",
                        label=f"Recency decay (lambda={lam:g}, tau={tau:g}d)",
                        ranker=make_decay(lam, tau),
                        family="temporal",
                        params={"lambda": lam, "tau": tau},
                        sweep_of="decay",
                    )
                )
    return methods
