"""Drug-name autocomplete over the FDA NDC directory.

The list is built once from `product.txt` inside `ndctext.zip`
(https://www.accessdata.fda.gov/cder/ndctext.zip - no login, US federal
public domain) and shipped as `app/data/drug_names.json`: distinct
proprietary names plus the individual ingredients of multi-ingredient
generics, deduplicated case-insensitively, ~52k entries in ~2.3 MB.

Lookup is an in-memory sorted list with binary search, so a keystroke
costs microseconds server-side and needs no database, no index build,
and no network call per key. Two tiers: names starting with the query
first, then names where any later word starts with it (so "clavulanate"
finds "Clavulanate Potassium" even though it sorts under C, not A).

A two-character minimum keeps single letters - which match tens of
thousands of rows - from ever leaving the server.
"""

from __future__ import annotations

import bisect
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent / "data" / "drug_names.json"

MIN_QUERY_CHARS = 2
DEFAULT_LIMIT = 8
MAX_LIMIT = 20


@lru_cache(maxsize=1)
def _load() -> Tuple[List[str], List[str]]:
    """Display names and parallel casefold keys, both in sort order."""
    try:
        names = json.loads(_DATA.read_text(encoding="utf-8"))["names"]
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("drug list unavailable (%s); suggestions disabled", exc)
        return [], []
    keys = [n.casefold() for n in names]
    order = sorted(range(len(names)), key=lambda i: keys[i])
    names = [names[i] for i in order]
    keys = [keys[i] for i in order]
    return names, keys


def count() -> int:
    """How many names are searchable. Reported, not assumed."""
    names, _keys = _load()
    return len(names)


def suggest(prefix: str, limit: int = DEFAULT_LIMIT) -> List[str]:
    """Up to `limit` drug names for a typed prefix. Never raises."""
    query = (prefix or "").strip().casefold()
    if len(query) < MIN_QUERY_CHARS:
        return []
    limit = max(1, min(int(limit), MAX_LIMIT))
    names, keys = _load()
    if not names:
        return []

    out: List[str] = []
    # Tier 1: the name itself starts with the query. Binary search lands
    # exactly on the first match, so this touches only the matches.
    i = bisect.bisect_left(keys, query)
    while i < len(names) and keys[i].startswith(query) and len(out) < limit:
        out.append(names[i])
        i += 1
    # Tier 2: a later word starts with it. Linear, but 52k short-string
    # comparisons are sub-millisecond and this only runs when tier 1
    # under-filled.
    if len(out) < limit:
        for name, key in zip(names, keys):
            if len(out) >= limit:
                break
            if key.startswith(query):
                continue
            if any(word.startswith(query) for word in key.split()):
                out.append(name)
    return out
