"""Qdrant vector store.

One collection, named vectors. A single filter can then serve a text
query against document vectors and an image query against image vectors
without two collections that can drift out of sync.

Payload carries everything the validity-aware retriever needs to weight
its results:
    patient_id     hard filter, never optional
    validity_state "current" | "superseded", derived at index time
    valid_time_epoch  numeric, because Qdrant's Range filter is numeric
                      only - storing the ISO string alone makes the
                      as-of filter fail silently
    event_id, event_type, source_type

Indexed vectors are a *cache*. The event log is the source of truth: if
Qdrant is empty or stale, the record is still complete, and the
retriever falls back to scanning the record in memory. That is why a
lost index degrades retrieval quality rather than losing data.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

import numpy as np

from app.core.config import Settings, get_settings
from app.core.validity import Event, SupersessionGraph, Validity

logger = logging.getLogger(__name__)


def point_id(event_id: str) -> str:
    """Qdrant point ids must be UUIDs or unsigned ints.

    Deterministic so re-indexing the same event updates its vector rather
    than creating a duplicate.
    """
    return str(UUID(hashlib.sha1(event_id.encode()).hexdigest()[:32]))


def image_point_id(event_id: str) -> str:
    """A *separate* point for an event's image vector.

    This exists because of a bug that made image retrieval rot silently.

    `upsert` in Qdrant replaces a point's whole vector set rather than
    merging named vectors. `upsert_text` and `upsert_image` derived the same
    point id from the same event id, so whichever ran last won: an imaging
    event was written with its `biomedclip` vector, and then the background
    indexer wrote the event's `medcpt` text vector to the same point and
    erased the image vector.

    The symptom was the worst kind. Image search returned the study
    immediately after upload, passed every test, and then quietly stopped
    finding it a few seconds later -- by which point the vector was simply
    gone. The stored pixels were still on disk and the event was still in
    the record; only the index had lost it.

    `update_vectors` (PATCH) merges rather than replaces and would fix the
    collision, but it 404s on a point that does not exist yet, so it cannot
    create one. Choosing between upsert and patch per call means a
    read-then-write race with the indexer, and the failure mode of losing
    that race is the same silent loss.

    Separate ids remove the interaction entirely: neither write can disturb
    the other, in any order, with no extra round trip. Retrieval filters on
    `patient_id` and names the `biomedclip` vector, so it is unaffected.
    The cost is one extra point per imaging event, which nothing counts.
    """
    return str(UUID(hashlib.sha1(f"image:{event_id}".encode()).hexdigest()[:32]))


class VectorStore:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None
        self._ready = False

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=30,
            )
        return self._client

    @property
    def collection(self) -> str:
        return self.settings.qdrant_collection

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant unreachable: %s", exc)
            return False

    def ensure_collection(self, text_dim: int, image_dim: int) -> bool:
        """Create the collection if absent. Idempotent."""
        if self._ready:
            return True
        try:
            from qdrant_client import models

            existing = {
                c.name for c in self.client.get_collections().collections
            }
            if self.collection in existing:
                self._ready = True
                return True

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    self.settings.qdrant_text_vector: models.VectorParams(
                        size=text_dim, distance=models.Distance.COSINE
                    ),
                    self.settings.qdrant_image_vector: models.VectorParams(
                        size=image_dim, distance=models.Distance.COSINE
                    ),
                },
            )
            logger.info(
                "created collection %s (text %d-d, image %d-d)",
                self.collection, text_dim, image_dim,
            )
            self._ready = True
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not create collection: %s", exc)
            return False

    # â”€â”€ writing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def upsert_text(
        self,
        events: Sequence[Event],
        vectors: np.ndarray,
        graph: Optional[SupersessionGraph] = None,
    ) -> int:
        if not self.ensure_collection(
            self.settings.qdrant_text_dim, self.settings.qdrant_image_dim
        ):
            return 0

        from qdrant_client import models

        points = []
        # strict: a length mismatch would otherwise truncate the index
        # silently, and a quietly incomplete index is indistinguishable
        # from a working one until retrieval starts missing records.
        for event, vector in zip(events, vectors, strict=True):
            points.append(
                models.PointStruct(
                    id=point_id(event.event_id),
                    vector={
                        self.settings.qdrant_text_vector: (
                            np.asarray(vector).ravel().tolist()
                        )
                    },
                    payload={
                        "patient_id": event.patient_id,
                        "event_id": event.event_id,
                        "event_type": str(event.event_type),
                        "source_type": str(event.source_type),
                        "validity_state": (
                            "superseded"
                            if graph and graph.state(event) is Validity.SUPERSEDED
                            else "current"
                        ),
                        "valid_time": event.valid_time.isoformat(),
                        "valid_time_epoch": int(
                            event.valid_time.timestamp()
                        ),
                        "content_hash": (event.payload or {}).get(
                            "content_hash", ""
                        ),
                    },
                )
            )
        try:
            self.client.upsert(
                collection_name=self.collection, points=points, wait=False
            )
            return len(points)
        except Exception as exc:  # noqa: BLE001
            logger.warning("upsert failed: %s", exc)
            return 0

    def upsert_image(
        self,
        event_id: str,
        patient_id: str,
        vector: np.ndarray,
        caption: str = "",
        event_type: str = "imaging",
        valid_time: Optional[datetime] = None,
    ) -> bool:
        if not self.ensure_collection(
            self.settings.qdrant_text_dim, self.settings.qdrant_image_dim
        ):
            return False
        from qdrant_client import models

        when = valid_time or datetime.now(timezone.utc)
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=image_point_id(event_id),
                        vector={
                            self.settings.qdrant_image_vector: (
                                np.asarray(vector).ravel().tolist()
                            )
                        },
                        payload={
                            "patient_id": patient_id,
                            "event_id": event_id,
                            "event_type": event_type,
                            "source_type": "ocr",
                            "validity_state": "current",
                            "caption": caption[:500],
                            "valid_time": when.isoformat(),
                            "valid_time_epoch": int(when.timestamp()),
                            "content_hash": "",
                        },
                    )
                ],
                wait=False,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("image upsert failed: %s", exc)
            return False

    # â”€â”€ reading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def search(
        self,
        vector: np.ndarray,
        patient_id: str,
        *,
        as_of_epoch: Optional[int] = None,
        include_superseded: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Similarity search scoped to one patient.

        Two invariants, both enforced in the filter rather than in
        application code, because a forgotten `if` in Python is a
        cross-patient leak and a forgotten filter clause is the same
        thing with less visibility.
        """
        if not self._ready and not self.healthy():
            return []

        from qdrant_client import models

        must: List[Any] = [
            models.FieldCondition(
                key="patient_id",
                match=models.MatchValue(value=patient_id),
            )
        ]
        if not include_superseded:
            must.append(
                models.FieldCondition(
                    key="validity_state",
                    match=models.MatchValue(value="current"),
                )
            )
        # A future record must never answer an as-of query. Range is
        # numeric only, which is why valid_time_epoch exists.
        if as_of_epoch is not None:
            must.append(
                models.FieldCondition(
                    key="valid_time_epoch",
                    range=models.Range(lte=as_of_epoch),
                )
            )

        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=(
                    self.settings.qdrant_text_vector,
                    np.asarray(vector).ravel().tolist(),
                ),
                query_filter=models.Filter(must=must),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("search failed: %s", exc)
            return []

        out = []
        for hit in results:
            payload = hit.payload or {}
            out.append(
                {
                    "event_id": payload.get("event_id"),
                    "score": float(hit.score),
                    "validity": payload.get("validity_state", "current"),
                    "valid_time": payload.get("valid_time"),
                    "event_type": payload.get("event_type"),
                }
            )
        return out

    def search_by_image(
        self,
        vector: np.ndarray,
        patient_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self._ready and not self.healthy():
            return []
        from qdrant_client import models

        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=(
                    self.settings.qdrant_image_vector,
                    np.asarray(vector).ravel().tolist(),
                ),
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="patient_id",
                            match=models.MatchValue(value=patient_id),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
            )
        except Exception:  # noqa: BLE001
            return []

        return [
            {
                "event_id": (h.payload or {}).get("event_id"),
                "score": float(h.score),
                "caption": (h.payload or {}).get("caption", ""),
            }
            for h in results
        ]

    def count(self) -> int:
        try:
            result = self.client.count(
                collection_name=self.collection, exact=False
            )
            return int(result.count)
        except Exception:  # noqa: BLE001
            return 0

    def delete_patient(self, patient_id: str) -> bool:
        from qdrant_client import models

        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="patient_id",
                                match=models.MatchValue(value=patient_id),
                            )
                        ]
                    )
                ),
            )
            return True
        except Exception:  # noqa: BLE001
            return False


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

