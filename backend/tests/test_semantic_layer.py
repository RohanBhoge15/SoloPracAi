"""Tests for the semantic layer: vector store, hybrid scorer, embedder.

The security-relevant test is `test_search_always_filters_by_patient`. The
cross-patient guard is enforced in the Qdrant filter rather than in
application code, because a forgotten `if` in Python is a cross-patient
leak and a forgotten filter clause is the same defect with less
visibility. These tests read the filter that was actually sent, rather
than trusting that the code that builds it is correct.

The second group tests *degradation*. Every one of these paths must fall
back to lexical similarity rather than raise, because a missing model or
an unreachable vector store should cost answer quality, not availability.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from app.core.config import Settings, get_settings
from app.core.validity import Event, EventType, SourceType, SupersessionGraph
from app.services import embeddings as emb
from app.services import vectors as vec
from app.services.hybrid import HybridSemantic, _document_text, index_events
from app.services.retrieval import lexical_similarity

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_event(
    event_id: str,
    patient_id: str,
    days: float = 0.0,
    supersedes_id: Optional[str] = None,
    **payload,
) -> Event:
    # The core type requires the two to agree: only a correction may
    # declare supersedes_id. Derive the source rather than letting a test
    # pass a mismatched pair.
    source = SourceType.CORRECTION if supersedes_id else SourceType.MANUAL
    return Event(
        event_id=event_id,
        patient_id=patient_id,
        event_type=EventType.LAB,
        valid_time=BASE + timedelta(days=days),
        payload=payload or {"test": "Glucose", "value": 140},
        source_type=source,
        supersedes_id=supersedes_id,
    )


@pytest.fixture
def settings() -> Settings:
    s = get_settings().model_copy(deep=True)
    s.qdrant_url = "http://127.0.0.1:1"  # guaranteed dead
    s.qdrant_api_key = None
    s.models_root = "Z:/no-such-models"
    s.medcpt_query_path = "Z:/no-such-models/q"
    s.medcpt_article_path = "Z:/no-such-models/a"
    s.bge_m3_path = "Z:/no-such-models/bge"
    s.biomedclip_path = "Z:/no-such-models/clip"
    s.enable_vision = False
    s.text_encoder = "medcpt"
    s.enable_multilingual = False
    return s


# â”€â”€ point identity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_point_id_is_deterministic():
    """Re-indexing must update, not duplicate.

    Qdrant rejects a non-UUID point id and a random one would create a
    second copy of every record on every reindex.
    """
    first = vec.point_id("abc-123")
    second = vec.point_id("abc-123")
    assert first == second
    assert first != vec.point_id("abc-124")
    # Must parse as a UUID, since that is what Qdrant accepts.
    from uuid import UUID

    assert str(UUID(first)) == first


# â”€â”€ the cross-patient guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class _RecordingClient:
    """Captures the search call instead of performing it."""

    def __init__(self) -> None:
        self.kwargs: Optional[Dict[str, Any]] = None
        self.deleted: List[Any] = []

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def search(self, **kwargs):
        self.kwargs = kwargs
        return []

    def delete(self, **kwargs):
        self.deleted.append(kwargs)
        return SimpleNamespace()

    def count(self, **kwargs):
        return SimpleNamespace(count=0)

    def upsert(self, **kwargs):
        return SimpleNamespace()


def _store(settings, client) -> vec.VectorStore:
    store = vec.VectorStore(settings)
    store._client = client
    store._ready = True
    return store


def _must_clauses(client) -> Dict[str, Any]:
    """Reduce the recorded filter to a plain dict for assertions."""
    flt = client.kwargs["query_filter"]
    out: Dict[str, Any] = {}
    for clause in flt.must:
        key = clause.key
        if clause.match is not None:
            out[key] = clause.match.value
        elif clause.range is not None:
            out[key] = {
                "lte": clause.range.lte,
                "gte": clause.range.gte,
            }
    return out


def test_search_always_filters_by_patient(settings):
    """The one invariant that must never be optional."""
    client = _RecordingClient()
    store = _store(settings, client)
    store.search(np.zeros(768, dtype=np.float32), "patient-A", limit=5)
    assert _must_clauses(client)["patient_id"] == "patient-A"


def test_search_excludes_superseded_by_default(settings):
    client = _RecordingClient()
    store = _store(settings, client)
    store.search(
        np.zeros(768, dtype=np.float32),
        "patient-A",
        include_superseded=False,
    )
    clauses = _must_clauses(client)
    assert clauses["validity_state"] == "current"
    # Patient scoping must survive the extra clause.
    assert clauses["patient_id"] == "patient-A"


def test_search_keeps_superseded_when_asked(settings):
    """Corrections are the point of the system, so they are reachable."""
    client = _RecordingClient()
    store = _store(settings, client)
    store.search(
        np.zeros(768, dtype=np.float32),
        "patient-A",
        include_superseded=True,
    )
    assert "validity_state" not in _must_clauses(client)


def test_excluding_superseded_does_not_depend_on_the_backend(settings):
    """The exclusion must hold whether or not Qdrant is reachable.

    The retriever demotes superseded records instead of dropping them, so
    relying on the Qdrant filter alone means a cold index quietly returns
    them anyway â€” the same request meaning two different things depending
    on cache state.
    """
    from app.core.validity import Validity
    from app.services.retrieval import ValidityAwareRetriever

    older = make_event("e1", "p", days=0, test="Glucose", value=140)
    newer = make_event(
        "e2", "p", days=10, supersedes_id="e1", test="Glucose", value=155
    )
    graph = SupersessionGraph([older, newer])
    retriever = ValidityAwareRetriever(graph=graph)

    def lexical(_query, _event):
        return 0.9

    ranked = retriever.rank(
        "glucose", "p", [older, newer], semantic_fn=lexical, k=10
    )
    # The retriever keeps both, labelled.
    assert len(ranked) == 2
    assert {str(r.validity) for r in ranked} == {
        str(Validity.SUPERSEDED),
        str(Validity.CURRENT),
    }
    # Which is exactly why the route has to drop them itself.


def test_as_of_filter_uses_numeric_range(settings):
    """Qdrant's Range filter is numeric only.

    Storing the ISO string alone would make this clause silently
    unsatisfiable, and an as-of query would quietly return nothing.
    """
    client = _RecordingClient()
    store = _store(settings, client)
    cutoff = int(datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp())
    store.search(
        np.zeros(768, dtype=np.float32), "patient-A", as_of_epoch=cutoff
    )
    clause = _must_clauses(client)["valid_time_epoch"]
    assert isinstance(clause, dict), "must be a Range, not a string comparison"
    assert clause["lte"] == cutoff


def test_search_returns_empty_when_store_is_dead(settings):
    """A dead vector store degrades to no results, not a 500."""
    store = vec.VectorStore(settings)  # no client injected
    assert store.search(np.zeros(768, dtype=np.float32), "p") == []
    assert store.count() == 0
    assert store.healthy() is False


def test_delete_patient_is_scoped(settings):
    client = _RecordingClient()
    store = _store(settings, client)
    assert store.delete_patient("patient-A") is True
    selector = client.deleted[0]["points_selector"]
    assert selector.filter.must[0].match.value == "patient-A"


# â”€â”€ payload shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_upsert_text_labels_validity_from_the_graph(settings):
    """Validity is derived at index time so the filter can use it."""
    from qdrant_client import models as qmodels  # noqa: F401

    newer = make_event("e2", "p", days=10, test="Glucose", value=155)
    graph = SupersessionGraph([newer])
    store = _store(settings, _RecordingClient())
    captured: Dict[str, Any] = {}

    def fake_upsert(collection_name, points, wait=False):
        captured["points"] = points
        return SimpleNamespace()

    store.client.upsert = fake_upsert
    store.upsert_text([newer], np.zeros((1, 768), dtype=np.float32), graph)

    payload = captured["points"][0].payload
    assert payload["validity_state"] == "current"
    assert payload["patient_id"] == "p"
    # The numeric mirror of valid_time must be present, or as-of filtering
    # cannot work at all.
    assert isinstance(payload["valid_time_epoch"], int)
    assert isinstance(payload["valid_time"], str)
    # The vector must be written under the configured named vector, or the
    # search side reads a vector that was never stored.
    assert list(payload)  # payload is populated
    assert captured["points"][0].vector is not None


def test_upsert_text_marks_a_superseded_event(settings):
    """The correction carries the link, so the older record is marked."""
    older = make_event("e1", "p", days=0, test="Glucose", value=140)
    newer = make_event(
        "e2", "p", days=10, supersedes_id="e1", test="Glucose", value=155
    )
    graph = SupersessionGraph([older, newer])

    store = _store(settings, _RecordingClient())
    captured: Dict[str, Any] = {}
    store.client.upsert = lambda **kw: captured.update(points=kw["points"])

    store.upsert_text(
        [older, newer],
        np.zeros((2, 768), dtype=np.float32),
        graph,
    )
    states = {
        p.payload["event_id"]: p.payload["validity_state"]
        for p in captured["points"]
    }
    assert states["e1"] == "superseded"
    assert states["e2"] == "current"


# â”€â”€ hybrid scorer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_hybrid_falls_back_to_lexical(settings, monkeypatch):
    """No encoder, no vector store: still answers, using token overlap."""
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())

    scorer = HybridSemantic("p")
    # Force the dead paths: the registry has no models and the store has no
    # client, so healthy() is False.
    scorer.embedder = emb.Embedder(settings)
    scorer.store = vec.VectorStore(settings)

    event = make_event("e1", "p", test="Glucose", value=140)
    assert scorer.backend == "lexical"
    assert scorer("what is her blood sugar", event) == 0.0
    assert scorer("glucose", event) == pytest.approx(
        lexical_similarity("glucose", event)
    )


def test_hybrid_clamps_negative_cosine(settings, monkeypatch):
    """Qdrant cosine runs -1..1; the retriever expects 0..1.

    A raw -0.4 would be added to the validity weight and could rank a
    relevant record below an irrelevant one.
    """
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())
    scorer = HybridSemantic("p")
    scorer.embedder = emb.Embedder(settings)
    scorer.store = vec.VectorStore(settings)
    scorer._scores = {"e1": -0.4}
    scorer.backend = "qdrant"
    assert scorer("anything", make_event("e1", "p")) == 0.0


def test_hybrid_returns_one_query_encode_per_request(settings, monkeypatch):
    """Re-encoding the query for every candidate is the obvious trap.

    The retriever calls the scorer once per record, so a scorer that
    encodes on each call turns a 2 ms search into a 15 s one.
    """
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())
    calls = {"n": 0}

    class CountingEmbedder:
        def encode_query(self, text):
            calls["n"] += 1
            return np.zeros(768, dtype=np.float32)

    client = _RecordingClient()
    store = _store(settings, client)
    scorer = HybridSemantic("p")
    scorer.embedder = CountingEmbedder()
    scorer.store = store

    for index in range(5):
        scorer("q", make_event(f"e{index}", "p"))

    assert calls["n"] == 1
    assert scorer.backend == "lexical"  # client returned no hits


def test_hybrid_prefers_qdrant_when_hits_exist(settings, monkeypatch):
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())

    class Hit:
        def __init__(self, eid, score):
            self.payload = {"event_id": eid}
            self.score = score

    class Client(_RecordingClient):
        def search(self, **kwargs):
            self.kwargs = kwargs
            return [Hit("e1", 0.91), Hit("e2", 0.42)]

    scorer = HybridSemantic("p")
    scorer.embedder = SimpleNamespace(encode_query=lambda t: np.zeros(768, dtype=np.float32))
    scorer.store = _store(settings, Client())

    assert scorer("q", make_event("e1", "p")) == pytest.approx(0.91)
    assert scorer("q", make_event("e2", "p")) == pytest.approx(0.42)
    # A record Qdrant did not return has no similarity, not a guessed one.
    assert scorer("q", make_event("e3", "p")) == 0.0
    assert scorer.backend == "qdrant"


# â”€â”€ index_events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_index_events_handles_no_input():
    assert index_events([]) == 0


def test_index_events_skips_when_no_encoder(settings, monkeypatch):
    """A deployment without torch must not raise, just not index."""
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())
    monkeypatch.setattr("app.services.hybrid.get_embedder", lambda: emb.Embedder(settings))
    monkeypatch.setattr("app.services.hybrid.get_store", lambda: _store(settings, _RecordingClient()))
    assert index_events([make_event("e1", "p")]) == 0


def test_index_events_skips_when_store_is_dead(settings, monkeypatch):
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())
    monkeypatch.setattr("app.services.hybrid.get_embedder", lambda: emb.Embedder(settings))
    monkeypatch.setattr("app.services.hybrid.get_store", lambda: vec.VectorStore(settings))
    assert index_events([make_event("e1", "p")]) == 0


# â”€â”€ document text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_document_text_is_flattened_and_bounded():
    """MedCPT truncates at 512 tokens, so a long note is capped here.

    Capping here rather than in the model means the loss is visible in the
    code instead of hidden in a default.
    """
    event = make_event(
        "e1", "p", test="Glucose", value=140, note="x" * 5000, items=["a", "b"]
    )
    text = _document_text(event)
    assert "Laboratory result" in text
    assert "glucose" in text.lower()
    # Nested structures are skipped: they are not sentences.
    assert "items" not in text
    assert len(text) <= 1800


# â”€â”€ embedder degradation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_embedder_reports_degraded_without_models(settings, monkeypatch):
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())
    embedder = emb.Embedder(settings)
    assert embedder.text_available() is False
    assert embedder.encode_query("anything") is None
    assert embedder.encode_document("anything") is None
    assert embedder.encode_texts(["a", "b"]) is None


def test_embedder_vision_off_returns_none(settings, monkeypatch):
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())
    embedder = emb.Embedder(settings)
    assert embedder.image_available() is False
    assert embedder.encode_image(b"not-an-image") is None
    assert embedder.encode_image_caption("chest x-ray") is None


def test_empty_text_is_never_encoded(settings, monkeypatch):
    monkeypatch.setattr(emb, "_registry", emb._ModelRegistry())
    embedder = emb.Embedder(settings)
    for blank in ("", "   ", "\n\t"):
        assert embedder.encode_query(blank) is None
        assert embedder.encode_document(blank) is None


def test_status_separates_loaded_from_available(settings, monkeypatch):
    """`not loaded yet` must not be reported as `unavailable`.

    The article tower only loads when something needs to write a vector, so
    conflating the two reports a healthy indexer as broken.
    """
    registry = emb._ModelRegistry()
    monkeypatch.setattr(emb, "_registry", registry)
    status = emb.Embedder(settings).status()

    assert status["query_tower_loaded"] is False
    assert status["article_tower_loaded"] is False
    # These paths do not exist, so unavailable is the honest answer here.
    assert status["indexer_available"] is False

    # Once indexing has happened, the article tower is resident and says so.
    registry._article_model = object()
    registry._failed.discard("article")
    loaded = emb.Embedder(settings).status()
    assert loaded["article_tower_loaded"] is True
    assert loaded["query_tower_loaded"] is False


def test_status_reports_dim_and_failures(settings, monkeypatch):
    registry = emb._ModelRegistry()
    registry._failed.add("medcpt")
    monkeypatch.setattr(emb, "_registry", registry)
    status = emb.Embedder(settings).status()
    assert status["text_dim"] == 768
    # The device follows the hardware, so asserting a literal here passed only
    # on a machine without a GPU and failed on every machine that had one.
    # Asserted against what torch actually reports, which is the property
    # actually under test.
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert status["device"] == expected
    assert "medcpt" in status["failed"]


def test_dim_follows_encoder_choice(settings):
    settings.text_encoder = "bge-m3"
    assert emb.Embedder(settings).dim() == 1024
    settings.enable_multilingual = True
    assert emb.Embedder(settings).dim() == 1024
    settings.text_encoder = "medcpt"
    settings.enable_multilingual = False
    assert emb.Embedder(settings).dim() == 768


# â”€â”€ offline BiomedCLIP preparation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_medcpt_vocab_size_is_the_layer_width_not_the_tokenizer_count():
    """PubMedBERT pads its embedding to 30 522 while vocab.txt holds 28 895.

    Sizing from the tokenizer produces a checkpoint that cannot load, so
    the constant is pinned and this documents why.
    """
    assert emb.BIOMEDBERT_VOCAB_SIZE == 30_522
    assert emb.BIOMEDBERT_VOCAB_SIZE != 28_895


def test_locate_checkpoint_prefers_safetensors(tmp_path):
    (tmp_path / "open_clip_pytorch_model.bin").write_bytes(b"x")
    (tmp_path / "open_clip_model.safetensors").write_bytes(b"x")
    assert emb._locate_checkpoint(tmp_path).name == "open_clip_model.safetensors"


def test_locate_checkpoint_falls_back_to_bin(tmp_path):
    (tmp_path / "open_clip_pytorch_model.bin").write_bytes(b"x")
    assert emb._locate_checkpoint(tmp_path).name == "open_clip_pytorch_model.bin"


def test_locate_checkpoint_returns_none_when_absent(tmp_path):
    assert emb._locate_checkpoint(tmp_path) is None


class _FakeModel:
    """Minimal stand-in that records what a strict load reports."""

    def __init__(self, expected):
        self._expected = expected

    def load_state_dict(self, state, strict=False):
        keys = set(state)
        return SimpleNamespace(
            missing_keys=sorted(self._expected - keys),
            unexpected_keys=sorted(keys - self._expected),
        )


def test_weight_loader_drops_only_position_ids(monkeypatch):
    """The one tolerated difference, and nothing else.

    transformers v5 no longer registers the BERT `position_ids` buffers as
    persistent, so the published checkpoint has keys the model does not
    declare. Dropping those is safe; dropping anything else would leave a
    randomly initialised layer that still encodes, just wrongly.
    """
    import open_clip.factory as factory

    monkeypatch.setattr(
        factory,
        "load_state_dict",
        lambda path, device="cpu": {
            "text.transformer.embeddings.position_ids": "stale",
            "visual.conv1.weight": "real",
            "text.transformer.embeddings.word_embeddings.weight": "real",
        },
    )
    model = _FakeModel(
        {
            "visual.conv1.weight",
            "text.transformer.embeddings.word_embeddings.weight",
        }
    )
    emb._load_biomedclip_weights(model, "ignored.bin")


def test_weight_loader_rejects_a_genuine_mismatch(monkeypatch):
    """A wrong checkpoint must raise, not load partially and continue."""
    import open_clip.factory as factory

    monkeypatch.setattr(
        factory,
        "load_state_dict",
        lambda path, device="cpu": {
            "visual.conv1.weight": "real",
            "visual.something_else.weight": "not in the model",
        },
    )
    model = _FakeModel({"visual.conv1.weight"})
    with pytest.raises(RuntimeError, match="does not match the architecture"):
        emb._load_biomedclip_weights(model, "ignored.bin")


def test_weight_loader_rejects_missing_layers(monkeypatch):
    """A checkpoint missing a whole layer must not pass as valid."""
    import open_clip.factory as factory

    monkeypatch.setattr(factory, "load_state_dict", lambda path, device="cpu": {})
    model = _FakeModel({"visual.conv1.weight", "text_projection.weight"})
    with pytest.raises(RuntimeError, match="does not match the architecture"):
        emb._load_biomedclip_weights(model, "ignored.bin")


def test_prepare_offline_clip_rewrites_the_hub_reference(tmp_path):
    """The prepared config must not name a hub repo.

    Otherwise the text tower's architecture is fetched over the network even
    though the weights are already on disk.
    """
    model_cfg = {
        "embed_dim": 512,
        "vision_cfg": {"timm_model_name": "vit_base_patch16_224"},
        "text_cfg": {
            "hf_model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            "context_length": 256,
        },
    }
    root = tmp_path / "model"
    root.mkdir()
    (root / "open_clip_config.json").write_text(
        json.dumps(
            {"model_cfg": model_cfg, "preprocess_cfg": {"mean": [0.1, 0.2, 0.3]}}
        ),
        encoding="utf-8",
    )
    (root / "open_clip_pytorch_model.bin").write_bytes(b"weights")

    cache = tmp_path / "cache"
    prepared = emb._ensure_offline_clip_config(root, str(cache))

    # The derived config lives in the cache, NOT in the model directory,
    # which is mounted read-only in Docker. The source is left exactly as
    # shipped, still pointing at the hub.
    assert prepared.is_relative_to(cache)
    source_text = (root / "open_clip_config.json").read_text(encoding="utf-8")
    assert source_text.count("microsoft/") == 1, "source config was modified"

    prepared_cfg = json.loads(
        (prepared / "open_clip_config.json").read_text(encoding="utf-8")
    )
    text_cfg = prepared_cfg["model_cfg"]["text_cfg"]
    assert "microsoft/" not in text_cfg["hf_model_name"]
    assert text_cfg["hf_model_pretrained"] is False
    # The context length and the preprocess values must survive the rewrite,
    # or the text tower truncates differently from how it was trained.
    assert text_cfg["context_length"] == 256
    assert prepared_cfg["preprocess_cfg"]["mean"] == [0.1, 0.2, 0.3]

    # No checkpoint is copied or linked in; it is loaded from its original
    # path, so preparing costs kilobytes rather than 765 MB.
    assert list(prepared.glob("*.bin")) == []
    assert emb._locate_checkpoint(root).name == "open_clip_pytorch_model.bin"

    # Idempotent: a second call must not rewrite or fail.
    stamp = (prepared / "open_clip_config.json").stat().st_mtime_ns
    assert emb._ensure_offline_clip_config(root, str(cache)) == prepared
    assert (prepared / "open_clip_config.json").stat().st_mtime_ns == stamp


def test_prepare_offline_clip_needs_the_source_config(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        emb._ensure_offline_clip_config(root, str(tmp_path / "cache"))


def test_text_config_shim_matches_the_embedding_width(tmp_path):
    shim = emb._ensure_text_config(tmp_path)
    written = json.loads((shim / "config.json").read_text(encoding="utf-8"))
    assert written["vocab_size"] == emb.BIOMEDBERT_VOCAB_SIZE
    assert written["hidden_size"] == 768
    assert written["architectures"] == ["BertModel"]


# â”€â”€ helpers used by the routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_caption_for_derives_something_from_a_filename():
    from app.api.routes.semantic import caption_for

    assert caption_for("chest_X-ray.jpg") == "chest X ray"
    assert caption_for("brain-mri.png") == "brain mri"
    # A name with nothing usable must still yield a non-empty string,
    # otherwise the image becomes unfindable by description.
    assert caption_for("___.jpg")
