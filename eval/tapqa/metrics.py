"""Metrics for TAP-QA.

The two that carry the paper:

**Stale-Answer Rate (SAR)** - the fraction of `corrected` queries whose
top-ranked record is the superseded one. Lower is better. This is the
safety claim: the system must not lead with a value that has been
retracted.

**Historical Accuracy** - the fraction of `historical` queries whose
top-ranked record is the one that was valid at the requested instant.
Higher is better. This is the honesty claim: a system that answers
"current" to every question scores perfectly on SAR and zero here.

Reporting one without the other is how a strawman gets built. A method
that wins on SAR by suppressing superseded records entirely also scores
zero on historical accuracy, and that is visible only when both are on
the same axis.

Measured at the *retrieval* level rather than the generated-text level.
That is deliberate: the generator is a free-tier LLM whose output varies
between runs, so a text-level metric measured once is not reproducible.
The retrieval question - "does this method put the right version first?" -
is deterministic, and it is the question the method actually changes.
The text-level arm is measured separately on a stratified subsample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tapqa.dataset import Query


@dataclass
class QueryOutcome:
    query_id: str
    kind: str
    test: str
    #: 1-based rank of a gold event, or 0 if absent. Fractional when the
    #: method scored gold equal to a neighbour - see tapqa.ranking.gold_rank.
    gold_rank: float
    ranking_size: int
    #: Positions sharing a score with a neighbour. Reported so a headline
    #: number resting on many ties is visible as such.
    tie_positions: int = 0
    latency_ms: float = 0.0
    #: For contradiction probes: did the ranking surface both conflicting
    #: records? Picking a side is not an answer.
    surfaced_conflict: bool = False
    #: Number of superseded records ranked first. Only meaningful for
    #: present-tense queries.
    top_is_superseded: bool = False


@dataclass
class MethodResult:
    key: str
    label: str
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    sweep_of: Optional[str] = None
    outcomes: List[QueryOutcome] = field(default_factory=list)

    def by_kind(self, kind: str) -> List[QueryOutcome]:
        return [o for o in self.outcomes if o.kind == kind]

    def summary(self) -> Dict[str, Any]:
        def rate(items: Iterable[QueryOutcome], hit) -> Optional[float]:
            items = list(items)
            if not items:
                return None
            return sum(1 for o in items if hit(o)) / len(items)

        corrected = self.by_kind("corrected")
        historical = self.by_kind("historical")
        unchanged = self.by_kind("unchanged")
        contradiction = self.by_kind("contradiction")

        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "params": self.params,
            "sweep_of": self.sweep_of,
            "n_total": len(self.outcomes),
            # Headline pair.
            "stale_answer_rate": rate(corrected, lambda o: o.top_is_superseded),
            "historical_accuracy": rate(historical, lambda o: o.gold_rank == 1),
            # Controls.
            "corrected_accuracy": rate(corrected, lambda o: o.gold_rank == 1),
            "unchanged_accuracy": rate(unchanged, lambda o: o.gold_rank == 1),
            "contradiction_detection": rate(
                contradiction, lambda o: o.surfaced_conflict
            ),
            # Retrieval quality over every query.
            "r1": rate(self.outcomes, lambda o: o.gold_rank == 1),
            "r5": rate(self.outcomes, lambda o: 1 <= o.gold_rank <= 5),
            "mrr": _mrr(self.outcomes),
            "ndcg10": _ndcg(self.outcomes),
            # How much of this method's output is decided by its own scores
            # rather than by the order it happened to emit. A high value
            # means the method is abstaining between near-identical
            # records, which is the honest reading of a weak baseline and
            # the reason tied records earn it no credit.
            "tied_positions": sum(o.tie_positions for o in self.outcomes),
            "queries_with_ties": sum(
                1 for o in self.outcomes if o.tie_positions > 0
            ),
            # Deployability.
            "latency_p50_ms": _percentile(
                [o.latency_ms for o in self.outcomes], 50
            ),
            "latency_p95_ms": _percentile(
                [o.latency_ms for o in self.outcomes], 95
            ),
        }


def _mrr(outcomes: Sequence[QueryOutcome]) -> Optional[float]:
    values = [1.0 / o.gold_rank for o in outcomes if o.gold_rank > 0]
    if not values:
        return None
    return sum(values) / len(outcomes)


def _ndcg(outcomes: Sequence[QueryOutcome], k: int = 10) -> Optional[float]:
    """Single-relevant-item NDCG, which reduces to 1/log2(rank+1).

    Recorded as a per-query mean over queries where the gold item was
    ranked at all, and reported alongside a coverage figure so a method
    cannot improve NDCG by retrieving almost nothing.
    """
    gains = [
        1.0 / math.log2(o.gold_rank + 1)
        for o in outcomes
        if 1 <= o.gold_rank <= k
    ]
    if not gains:
        return None
    return sum(gains) / len(gains)


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def selective_coverage(
    outcomes: Sequence[QueryOutcome], confidences: Sequence[float], target: float = 0.95
) -> Dict[str, Any]:
    """Coverage at a target accuracy - the R5 selective-prediction curve.

    Answers the deployability question: if the system only answers when it
    is confident enough, what fraction of questions can it answer and what
    accuracy does it hold at that point? A system with perfect precision
    on 4% of queries is not useful, and NDCG alone would not show that.
    """
    if not outcomes or len(confidences) != len(outcomes):
        return {"coverage": None, "accuracy": None, "threshold": None}
    pairs = sorted(
        zip(confidences, [o.gold_rank == 1 for o in outcomes]),
        key=lambda t: -t[0],
    )
    total_correct = sum(1 for _, ok in pairs if ok)
    needed = math.ceil(target * total_correct)
    if needed == 0:
        return {"coverage": None, "accuracy": None, "threshold": None}
    correct_so_far = 0
    for index, (_, ok) in enumerate(pairs, start=1):
        if ok:
            correct_so_far += 1
        if correct_so_far >= needed:
            return {
                "coverage": index / len(pairs),
                "accuracy": correct_so_far / index,
                "threshold": pairs[index - 1][0],
            }
    return {
        "coverage": 1.0,
        "accuracy": total_correct / len(pairs),
        "threshold": pairs[-1][0],
    }
