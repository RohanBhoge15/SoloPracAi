"""The semantic function the retriever actually uses.

`ValidityAwareRetriever` takes an injected `semantic_fn`, so the temporal
weighting is independent of how similarity is computed. That separation
is deliberate and is what lets TAP-QA measure the temporal contribution
without the embedding layer confounding it.

At request time this resolves, in order:

    1. Qdrant vector search, if the index has this patient's records
    2. Lexical token overlap, if the index is cold or empty
    3. Nothing, if the patient has no records

The index is a cache. If it is lost, the record is still complete in
Postgres, so retrieval degrades in quality rather than failing. That is
the difference between a cache and a source of truth, and it is why this
can be the default path without being a risk.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from app.core.validity import Event, EventType
from app.services.embeddings import get_embedder
from app.services.retrieval import lexical_similarity
from app.services.vectors import get_store

logger = logging.getLogger(__name__)


class HybridSemantic:
    """Callable semantic scorer backed by Qdrant, with a lexical fallback.

    Instances are cheap and hold per-call state, so one is created per
    request rather than shared. The `patient_id` is fixed for the
    lifetime of a scorer, which is what guarantees the vector filter
    cannot be forgotten.
    """

    def __init__(
        self,
        patient_id: str,
        *,
        as_of_epoch: Optional[int] = None,
        include_superseded: bool = True,
    ):
        self.patient_id = patient_id
        self.as_of_epoch = as_of_epoch
        self.include_superseded = include_superseded
        self.embedder = get_embedder()
        self.store = get_store()
        #: event_id -> similarity, plus the query they were computed for.
        self._scores: Optional[Dict[str, float]] = None
        self._scores_for: Optional[str] = None
        self.backend = "lexical"

    def _load_scores(self, query: str) -> bool:
        """One vector search, reused for every candidate.

        The cache is keyed on the query text, not merely "have I loaded
        once". The cache exists so the retriever, which calls this once per
        candidate, pays for a single encode and a single Qdrant round-trip -
        not so that a scorer can be reused across different questions.

        Keyed only on emptiness it was worse than a slow path: a scorer
        handed a second query returned the first query's rankings, and four
        unrelated questions came back byte-identical. The API builds a fresh
        scorer per request so it never showed, but that is a coincidence of
        the call site rather than a property of the class.
        """
        if self._scores is not None and self._scores_for == query:
            return self._backend_ready()
        try:
            vector = self.embedder.encode_query(query)
        except Exception:  # noqa: BLE001
            logger.debug("query encode failed", exc_info=True)
            vector = None

        if vector is None or not self.store.healthy():
            self._scores = {}
            self._scores_for = query
            self.backend = "lexical"
            return False

        try:
            hits = self.store.search(
                vector,
                self.patient_id,
                as_of_epoch=self.as_of_epoch,
                include_superseded=self.include_superseded,
                limit=50,
            )
        except Exception:  # noqa: BLE001
            self._scores = {}
            self._scores_for = query
            self.backend = "lexical"
            return False

        self._scores = {h["event_id"]: h["score"] for h in hits if h.get("event_id")}
        self._scores_for = query
        self.backend = "qdrant" if self._scores else "lexical"
        return bool(self._scores)

    def _backend_ready(self) -> bool:
        return self.backend == "qdrant" and bool(self._scores)

    def __call__(self, query: str, event: Event) -> float:
        if self._load_scores(query):
            # Cosine similarity from Qdrant is -1..1; the retriever expects
            # roughly 0..1 so a 0.85 relevance is not treated as negative.
            return max(0.0, self._scores.get(event.event_id, 0.0))
        return lexical_similarity(query, event)


def semantic_for(
    patient_id: str,
    *,
    as_of: Optional[int] = None,
    include_superseded: bool = True,
) -> HybridSemantic:
    return HybridSemantic(
        patient_id, as_of_epoch=as_of, include_superseded=include_superseded
    )


def index_events(events: List[Event], graph=None) -> int:
    """Embed and index a patient's events.

    Synchronous by design: this is CPU-bound work (tokenising, a forward
    pass, then a network write) and calling it directly from an async
    handler would block the event loop for the duration. Routes wrap it in
    `asyncio.to_thread`.

    Silently does nothing when no encoder is available, so a deployment
    without torch still runs. Returns the number of vectors written.
    """
    if not events:
        return 0

    embedder = get_embedder()
    store = get_store()

    if not store.healthy():
        logger.info("qdrant unreachable; skipping index")
        return 0

    if not embedder.indexer_available():
        logger.info("no article encoder available; skipping index")
        return 0

    texts = [_document_text(e) for e in events]
    vectors = embedder.encode_texts(texts, query=False)
    if vectors is not None:
        return store.upsert_text(events, vectors, graph)

    # Fall back to encoding row by row: one bad record should not cost the
    # whole index, since a partially indexed patient is still useful.
    rows = [embedder.encode_document(text) for text in texts]
    keep = [
        (event, row)
        # strict: a length mismatch here would silently drop records from
        # the index, which looks identical to a healthy index.
        for event, row in zip(events, rows, strict=True)
        if row is not None
    ]
    if not keep:
        return 0
    return store.upsert_text(
        [event for event, _ in keep], np.vstack([row for _, row in keep]), graph
    )


def _document_text(event: Event) -> str:
    """Flatten an event into the text the article encoder sees."""
    payload = event.payload or {}
    label = {
        EventType.LAB: "Laboratory result",
        EventType.MEDICATION: "Medication",
        EventType.DIAGNOSIS: "Diagnosis",
        EventType.VITAL: "Vital sign",
        EventType.NOTE: "Clinical note",
        EventType.IMAGING: "Imaging report",
    }.get(event.event_type, "Record")

    parts = [label]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        parts.append(f"{str(key).replace('_', ' ')}: {value}")

    body = " ".join(parts)
    # MedCPT truncates at 512 tokens; a note can be long, so cap it here
    # rather than silently losing the head of the document.
    return body[:1800]
