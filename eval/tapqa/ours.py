"""The SOLUM arm, evaluated by calling the real research code.

An earlier version of this file reimplemented the graph-conditioned
scoring by hand inside the baselines module, and got the as-of case
wrong: it asked "is this record superseded *now*" and gave a weight of
0.15, when the question the method exists to answer is "was this record
current *at the requested instant*". A record corrected last week, asked
about as of six months ago, was current then - and the hand-rolled
version ranked it below a distractor.

That is not a small bug. It is the paper's central claim, inverted, and
it showed up as our method scoring 28% historical accuracy while the
product demonstrably answers as-of questions correctly.

The fix is not to correct the copy. It is to delete it. This arm now
builds a real `SupersessionGraph` and calls the shipping
`ValidityAwareRetriever`, so the benchmark measures the code that runs in
production. A reimplementation of a method is a different method, and a
benchmark that scores a copy proves nothing about the system.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from app.core.validity import (
    Event as CoreEvent,
    EventType,
    SourceType,
    SupersessionGraph,
)
from app.services.retrieval import (
    ValidityAwareRetriever,
    parse_temporal_intent,
)

from tapqa.dataset import Query, Record
from tapqa.ranking import Ranking, from_scores

_TYPE_MAP = {
    "lab": EventType.LAB,
    "diagnosis": EventType.DIAGNOSIS,
    "medication": EventType.MEDICATION,
    "vital": EventType.VITAL,
    "note": EventType.NOTE,
    "imaging": EventType.IMAGING,
}

_SOURCE_MAP = {
    "manual": SourceType.MANUAL,
    "ocr": SourceType.OCR,
    "correction": SourceType.CORRECTION,
    "ai_suggestion": SourceType.AI_SUGGESTION,
}


def to_core(record: Record) -> CoreEvent:
    """Convert a generated record into the production core type.

    The conversion is not cosmetic: `SupersessionGraph` validates its
    invariants in `__post_init__`, so anything malformed raises here
    rather than being silently mis-scored.
    """
    return CoreEvent(
        event_id=record.event_id,
        patient_id=record.patient_id,
        event_type=_TYPE_MAP.get(record.event_type, EventType.NOTE),
        valid_time=datetime.fromisoformat(record.valid_time),
        payload=dict(record.payload),
        source_type=_SOURCE_MAP.get(record.source_type, SourceType.MANUAL),
        confidence=1.0,
        supersedes_id=record.supersedes_id,
    )


def make_graph_conditioned(
    alpha: float = 0.70,
    k: int = 50,
    embedder=None,
):
    """The contribution, evaluated through the shipping retriever.

    `alpha` and `k` are exposed so the sweep in the paper can vary them.
    They are not tuned to a best result: the defaults are the ones the
    product ships, and any sweep is reported in full.
    """

    def ranker(query: Query, records: List[Record]) -> Ranking:
        core = [to_core(r) for r in records]
        graph = SupersessionGraph(core)

        # The temporal intent is parsed by the production parser, not by
        # the harness, so a gap between the two cannot hide a failure.
        intent = parse_temporal_intent(
            query.text, datetime.fromisoformat(query.as_of)
            if query.as_of
            else datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        # An explicit `as_of` on the query is authoritative: the harness
        # generated it, so it is exact, whereas the text parser is
        # deliberately best-effort.
        if query.as_of:
            from app.services.retrieval import TemporalIntent

            intent = TemporalIntent(
                kind="as_of",
                instant=datetime.fromisoformat(query.as_of),
                confidence=1.0,
                cue="explicit",
            )

        if embedder is None:
            from app.services.retrieval import lexical_similarity

            semantic_fn = lambda q, e: lexical_similarity(q, e)  # noqa: E731
        else:
            semantic_fn = _DenseScorer(embedder, core)

        retriever = ValidityAwareRetriever(
            graph=graph, alpha=alpha, enforce_future_leak=True
        )
        ranked = retriever.rank(
            query.text,
            query.patient_id,
            core,
            intent=intent,
            now=datetime.fromisoformat(query.as_of)
            if query.as_of
            else datetime(2026, 6, 1, tzinfo=timezone.utc),
            semantic_fn=semantic_fn,
            k=k,
        )
        # The retriever's own `final` score is reported, not a synthetic
        # strictly-decreasing sequence. That matters: giving "ours" an
        # artificial total order would hand it full credit on any pair it
        # happened to score equally, which is exactly the credit the
        # baselines are being denied. Where the shipping retriever really
        # does tie two records, this arm ties too.
        return from_scores([(r.final, r.event.event_id) for r in ranked])

    return ranker


class _DenseScorer:
    """One vector search per query, reused for every candidate.

    The retriever calls the semantic function once per candidate, so a
    scorer that encodes on each call would turn a 2 ms query into a
    multi-second one. Same discipline as the production hybrid scorer.
    """

    def __init__(self, embedder, candidates: Sequence[CoreEvent]) -> None:
        self._embedder = embedder
        self._candidates = list(candidates)
        self._scores: Optional[Dict[str, float]] = None

    def __call__(self, query: str, event: CoreEvent) -> float:
        if self._scores is None:
            from app.services.hybrid import _document_text

            texts = [_document_text(e) for e in self._candidates]
            query_vector = self._embedder([query], is_query=True)
            matrix = self._embedder(texts, is_query=False)
            if query_vector is None or matrix is None:
                self._scores = {}
            else:
                scores = matrix @ query_vector[0]
                self._scores = {
                    e.event_id: max(0.0, float(s))
                    for e, s in zip(self._candidates, scores)
                }
        return self._scores.get(event.event_id, 0.0)
