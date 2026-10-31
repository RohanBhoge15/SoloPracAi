"""Text indexing must not destroy image vectors.

Found while making image retrieval work end to end. An imaging event was
written with its `biomedclip` vector, and then the background indexer wrote
the same event's `medcpt` text vector to the same Qdrant point -- and
Qdrant's `upsert` replaces a point's whole vector set rather than merging
named vectors. The image vector was gone.

The failure mode was silent and slow. Image search worked immediately after
upload, so every test that uploaded and then searched passed. It stopped
working a few seconds later, once the indexer caught up. The pixels were
still on disk, the event was still in the record, and only the index had
lost the study -- which is exactly the state a health check calls healthy.

These tests run against a real Qdrant rather than a fake, because the whole
question is what the server does to a point when it is written twice. A
mock would have encoded the assumption under test.

Run order is the point: image first, then text, because that is the order
the product produces. The reverse order would also have been broken, so it
is checked too.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.validity import Event as CoreEvent
from app.core.validity import EventType, SourceType
from app.services import vectors as vec


@pytest.fixture
def store():
    """A live VectorStore, skipped rather than failed when Qdrant is absent.

    The store points at a throwaway collection so the assertions cannot be
    confused by, or damage, the development index.
    """
    settings = vec.get_store().settings.model_copy(deep=True)
    suffix = uuid.uuid4().hex[:10]
    settings.qdrant_collection = f"test_imgvec_{suffix}"
    s = vec.VectorStore(settings)
    if not s.ensure_collection(settings.qdrant_text_dim, settings.qdrant_image_dim):
        pytest.skip("Qdrant is not reachable from this environment")
    yield s
    try:
        s.client.delete_collection(collection_name=s.collection)
    except Exception:  # noqa: BLE001
        pass


def _event(event_id: str, patient_id: str) -> CoreEvent:
    return CoreEvent(
        event_id=event_id,
        patient_id=patient_id,
        event_type=EventType.IMAGING,
        valid_time=datetime(2024, 3, 11, tzinfo=timezone.utc),
        payload={"study": "Chest X-Ray PA view"},
        source_type=SourceType.OCR,
        confidence=1.0,
    )


def _unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ── the id itself ───────────────────────────────────────────────────


def test_image_and_text_point_ids_differ():
    """The whole fix rests on these not colliding.

    Same event, two derivations. If they ever match again, the text indexer
    goes back to erasing image vectors and every test below still passes,
    because they each assert on `image_point_id` rather than on the
    separation.
    """
    eid = "11111111-2222-3333-4444-555555555555"
    assert vec.point_id(eid) != vec.image_point_id(eid)


def test_image_point_id_is_stable_and_a_uuid():
    """Re-uploading the same study must update its vector, not duplicate it."""
    from uuid import UUID

    eid = "11111111-2222-3333-4444-555555555555"
    assert vec.image_point_id(eid) == vec.image_point_id(eid)
    assert vec.image_point_id(eid) != vec.image_point_id("other")
    assert str(UUID(vec.image_point_id(eid))) == vec.image_point_id(eid)


# ── the behaviour that actually broke ───────────────────────────────


def test_text_indexing_does_not_erase_an_image_vector(store):
    """The regression, in the order the product produces it.

    Upload files the imaging event and its pixels; the indexer then writes
    the event's text vector. The study must still be findable by image
    afterwards.
    """
    pid = uuid.uuid4().hex
    eid = uuid.uuid4().hex
    ev = _event(eid, pid)
    dim = store.settings.qdrant_image_dim

    assert store.upsert_image(eid, pid, _unit(dim, 1), "Chest X-Ray PA view",
                              "imaging", ev.valid_time)
    assert len(store.search_by_image(_unit(dim, 1), pid, 5)) == 1, (
        "the image vector should be findable straight after upload"
    )

    # Now the background indexer writes the text vector for the same event.
    written = store.upsert_text([ev], np.vstack([_unit(store.settings.qdrant_text_dim, 2)]))
    assert written == 1

    hits = store.search_by_image(_unit(dim, 1), pid, 5)
    assert len(hits) == 1, (
        "text indexing erased the image vector: the study is stored and in "
        "the record but no longer findable by its pixels"
    )
    assert hits[0]["event_id"] == eid
    assert hits[0]["caption"] == "Chest X-Ray PA view"


def test_image_indexing_does_not_erase_a_text_vector(store):
    """The reverse order must be safe too.

    Re-uploading a study rewrites its image vector. If that dropped the text
    vector, the record would stop being findable by chat -- a study visible
    in one search and invisible in the other.
    """
    pid = uuid.uuid4().hex
    eid = uuid.uuid4().hex
    ev = _event(eid, pid)

    assert store.upsert_text([ev], np.vstack([_unit(store.settings.qdrant_text_dim, 3)]))
    assert store.upsert_image(eid, pid, _unit(store.settings.qdrant_image_dim, 4),
                              "Chest X-Ray PA view", "imaging", ev.valid_time)

    found = store.search(_unit(store.settings.qdrant_text_dim, 3), pid, limit=5)
    assert len(found) == 1, "image indexing erased the text vector"
    assert found[0]["event_id"] == eid


def test_both_vectors_coexist_after_a_reindex(store):
    """A reindex rebuilds text for every event, including imaging ones.

    This is the sequence that produced the original bug report, so it is
    asserted directly rather than inferred from the two ordering tests.
    """
    pid = uuid.uuid4().hex
    eid = uuid.uuid4().hex
    ev = _event(eid, pid)
    s = store.settings

    store.upsert_image(eid, pid, _unit(s.qdrant_image_dim, 5),
                       "Chest X-Ray PA view", "imaging", ev.valid_time)
    for _ in range(3):
        store.upsert_text([ev], np.vstack([_unit(s.qdrant_text_dim, 6)]))

    assert len(store.search_by_image(_unit(s.qdrant_image_dim, 5), pid, 5)) == 1
    assert len(store.search(_unit(s.qdrant_text_dim, 6), pid, limit=5)) == 1


def test_image_search_stays_scoped_to_the_patient(store):
    """The separate id must not weaken the ownership filter.

    A second patient's study must stay invisible, or the extra point becomes
    a cross-practice leak.
    """
    s = store.settings
    mine, theirs = uuid.uuid4().hex, uuid.uuid4().hex
    eid = uuid.uuid4().hex

    store.upsert_image(eid, theirs, _unit(s.qdrant_image_dim, 7),
                       "Chest X-Ray PA view", "imaging", _event(eid, theirs).valid_time)

    assert store.search_by_image(_unit(s.qdrant_image_dim, 7), mine, 5) == []
    assert len(store.search_by_image(_unit(s.qdrant_image_dim, 7), theirs, 5)) == 1


def test_delete_patient_removes_the_image_point_too(store):
    """Otherwise a deleted chart leaves its images retrievable."""
    pid = uuid.uuid4().hex
    eid = uuid.uuid4().hex
    s = store.settings

    store.upsert_image(eid, pid, _unit(s.qdrant_image_dim, 8),
                       "Chest X-Ray PA view", "imaging", _event(eid, pid).valid_time)
    assert len(store.search_by_image(_unit(s.qdrant_image_dim, 8), pid, 5)) == 1

    store.delete_patient(pid)
    assert store.search_by_image(_unit(s.qdrant_image_dim, 8), pid, 5) == []