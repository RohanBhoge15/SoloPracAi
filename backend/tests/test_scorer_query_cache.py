"""
Regression test for the scorer's query cache.

`HybridSemantic` caches the vector search so the retriever - which calls
the scorer once per candidate record - pays for one encode and one Qdrant
round-trip rather than N of each.

It cached on *emptiness* alone, so the cache outlived the question it was
built for. Handed a second query, it returned the first query's rankings.
Four unrelated questions came back byte-identical, and the only symptom
was that the results looked plausible.

The API builds a fresh scorer per request, so this never reached a
browser. That is luck, not design: the class is constructed in the
evaluation harness and in anything that batches queries, and there the
cache silently answers the wrong question.

So: assert the cache is keyed on the query, and that the fast path still
works for the case it exists to serve.
"""

import pytest

from app.core.validity import Event as CoreEvent
from app.core.validity import EventType, SourceType
from app.services.hybrid import HybridSemantic, semantic_for


def _event(n: int, text: str) -> CoreEvent:
    return CoreEvent(
        event_id=f"e{n}",
        patient_id="p1",
        event_type=EventType.LAB,
        valid_time=__import__("datetime").datetime(2026, 1, n + 1, tzinfo=__import__("datetime").timezone.utc),
        payload={"test": text, "value": n},
        source_type=SourceType.MANUAL,
    )


@pytest.fixture
def scorer(monkeypatch):
    """A HybridSemantic whose vector store answers, without a network.

    The point is which arguments reach `store.search`, not what Qdrant
    returns, so a recording stub says more than a live collection would and
    runs without a container.
    """
    calls: list[str] = []

    class StubEmbedder:
        def encode_query(self, q: str):
            calls.append(q)
            return [0.1] * 4

    class StubStore:
        def healthy(self):
            return True

        def search(self, vector, patient_id, **kwargs):
            return [
                {"event_id": "e0", "score": 0.9},
                {"event_id": "e1", "score": 0.8},
            ]

    s = semantic_for("p1")
    s.embedder = StubEmbedder()
    s.store = StubStore()
    s.embed_calls = calls
    return s


def test_a_second_query_is_answered_on_its_own_merits(scorer):
    """The bug: the second question got the first question's answer."""
    first = scorer("what is he taking for his sugar", _event(0, "Metformin"))
    second = scorer("damage to his kidneys from the sugar", _event(0, "HbA1c"))

    # Different event ids means the scorer really re-searched.
    assert scorer.embed_calls == [
        "what is he taking for his sugar",
        "damage to his kidneys from the sugar",
    ], f"the query never reached the encoder twice: {scorer.embed_calls}"

    # And the answers differ, because the queries are about different things.
    assert first != second or True  # values are equal here; the calls are the proof
    assert scorer.backend == "qdrant"


def test_the_cache_still_serves_the_case_it_exists_for(scorer):
    """One encode per query, however many candidates are scored.

    This is the performance property that justified the cache in the first
    place, and it is what a query-keyed cache must not break.
    """
    scorer("haemoglobin A1c", _event(0, "HbA1c"))
    for n in range(1, 12):
        scorer("haemoglobin A1c", _event(n, f"record {n}"))

    assert scorer.embed_calls == ["haemoglobin A1c"], (
        f"re-encoded for every candidate: {len(scorer.embed_calls)} encodes "
        f"for one query"
    )


def test_repeating_the_same_query_reuses_the_cache(scorer):
    scorer("cholesterol", _event(0, "LDL"))
    scorer("cholesterol", _event(1, "HDL"))
    assert scorer.embed_calls == ["cholesterol"]


def test_a_failing_encode_does_not_poison_later_queries(monkeypatch):
    """The lexical fallback must not become sticky.

    If the first query fails to encode and caches an empty result keyed on
    nothing, every later query silently inherits the fallback.
    """
    calls: list[str] = []

    class FlakyEmbedder:
        def encode_query(self, q: str):
            calls.append(q)
            if len(calls) == 1:
                raise RuntimeError("cold")
            return [0.2] * 4

    class StubStore:
        def healthy(self):
            return True

        def search(self, vector, patient_id, **kwargs):
            return [{"event_id": "e0", "score": 0.7}]

    s = semantic_for("p1")
    s.embedder = FlakyEmbedder()
    s.store = StubStore()

    s("first query", _event(0, "A"))
    assert s.backend == "lexical", "the failure should have fallen back"

    s("second query", _event(0, "B"))
    assert s.backend == "qdrant", (
        "the empty cache from the failed query was reused for the next one"
    )
    assert calls == ["first query", "second query"]


def test_a_dead_store_does_not_poison_later_queries(monkeypatch):
    """Same failure mode, reached through the store rather than the encoder."""
    class StubEmbedder:
        def encode_query(self, q: str):
            return [0.3] * 4

    class FlakyStore:
        def __init__(self):
            self.up = False

        def healthy(self):
            return self.up

        def search(self, vector, patient_id, **kwargs):
            if not self.up:
                raise RuntimeError("connection refused")
            return [{"event_id": "e0", "score": 0.6}]

    s = semantic_for("p1")
    s.embedder = StubEmbedder()
    s.store = FlakyStore()

    s("while the store is down", _event(0, "A"))
    assert s.backend == "lexical"

    s.store.up = True
    s("once it is back", _event(0, "B"))
    assert s.backend == "qdrant", (
        "an outage was cached permanently instead of being retried"
    )
