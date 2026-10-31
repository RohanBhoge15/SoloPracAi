"""Embedding cache for the benchmark.

The dataset is fixed, so a record's vector is the same every time a query
is scored. The first version re-encoded every record on every query -
850 queries times about 16 records, so roughly 13,600 forward passes on
CPU - which took seven and a half minutes and produced the same numbers
every time.

This computes each unique record once, in batches, and reuses it. On the
same run that is roughly 4,500 record encodes plus 850 query encodes,
batched, which is the difference between eight minutes and seconds.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CachedEncoder:
    """Encodes each distinct string once and memoises the vector.

    Records and queries are cached in the same table, which is safe
    because the two towers are not interchangeable: `is_query` selects
    the tower *before* the cache is consulted, so a query string and a
    record string never share an entry even if the text is identical.
    """

    def __init__(self, embedder, batch_size: int = 16) -> None:
        self._embedder = embedder
        self._batch = batch_size
        # Keyed by (is_query, text), never by text alone. The two MedCPT
        # towers are separately trained, so a query and a record that
        # happen to share a string are different vectors - keying on the
        # text alone would hand a baseline whichever tower ran first.
        self._cache: Dict[Tuple[bool, str], np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self.peak_batch = 0
        self.encode_calls = 0

    def __call__(self, texts: Sequence[str], is_query: bool = False):
        rows: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_idx: List[int] = []
        missing_txt: List[str] = []

        for i, text in enumerate(texts):
            hit = self._cache.get((is_query, text))
            if hit is not None:
                rows[i] = hit
                self.hits += 1
            else:
                missing_idx.append(i)
                missing_txt.append(text)

        if missing_txt:
            self.misses += len(missing_txt)
            for start in range(0, len(missing_txt), self._batch):
                chunk = missing_txt[start : start + self._batch]
                matrix = self._embedder(chunk, is_query=is_query)
                if matrix is None:
                    return None
                self.encode_calls += 1
                self.peak_batch = max(self.peak_batch, len(chunk))
                for offset, vector in enumerate(matrix):
                    self._cache[(is_query, chunk[offset])] = vector
                    rows[missing_idx[start + offset]] = vector

        if not rows:
            return None
        return np.vstack([r for r in rows if r is not None])

    def warm(self, texts: Sequence[str], is_query: bool = False) -> int:
        """Pre-encode a set of strings. Returns how many were new."""
        before = len(self._cache)
        self(list(texts), is_query=is_query)
        return len(self._cache) - before

    def report(self) -> Dict[str, int]:
        total = self.hits + self.misses
        return {
            "cache_entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "forward_batches": self.encode_calls,
            "peak_batch": self.peak_batch,
        }
