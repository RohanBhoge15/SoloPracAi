"""Statistics for TAP-QA.

The plan's rule is that a 2-point gap on n=100 is noise and must not be
presented as a discovery. That is enforced here rather than left to
judgement:

- Every headline rate carries a **Wilson score interval**, which behaves
  correctly near 0 and 1 where the normal approximation does not. A
  stale-answer rate of 0% on n=200 is not "0%", it is 0-1.8%.
- Binary paired outcomes use **McNemar's exact test**, because the two
  methods see the same queries and are therefore correlated. An
  unpaired test would overstate the evidence.
- Retrieval ranks use a **paired bootstrap** over queries, since the
  per-query values are not independent across a shared patient.

Implemented from the standard definitions rather than pulled from SciPy,
which is not a dependency. Each function is small enough to verify by
eye, and the exact binomial is computed directly so there is no
`binom_test` call to trust.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Interval:
    point: float
    low: float
    high: float
    n: int

    def as_pct(self) -> str:
        return f"{self.point * 100:.1f}% [{self.low * 100:.1f}, {self.high * 100:.1f}]"


def wilson(successes: int, n: int, z: float = 1.959963985) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1]
    and stays sensible at 0 and n successes, which is exactly where a
    safety metric lives.
    """
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0, 0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return Interval(
        point=p,
        low=max(0.0, (centre - margin) / denominator),
        high=min(1.0, (centre + margin) / denominator),
        n=n,
    )


def _binomial_tail_at_least(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, no SciPy."""
    if n == 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return min(1.0, total)


def mcnemar_exact(
    outcomes_a: Sequence[bool], outcomes_b: Sequence[bool]
) -> Dict[str, float]:
    """Exact McNemar test on paired binary outcomes.

    Only the discordant pairs carry information: a query both methods got
    right, or both got wrong, says nothing about which is better. The
    binomial test is on b/(b+c) against 0.5.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("paired outcomes must be the same length")
    b = sum(1 for a, x in zip(outcomes_a, outcomes_b) if a and not x)
    c = sum(1 for a, x in zip(outcomes_a, outcomes_b) if x and not a)
    n = b + c
    if n == 0:
        # Perfect agreement: no evidence of a difference, and saying so is
        # the correct answer rather than p = 1.0 pretending otherwise.
        return {
            "b": b,
            "c": c,
            "p_value": 1.0,
            "significant": False,
            "note": "no discordant pairs",
        }
    # The two-sided p-value is 2 * P(X >= max(b, c)), X ~ Binom(n, 0.5).
    #
    # It is `max`, not `min`, and getting that backwards is silent: with
    # b=97 and c=3, P(X >= 3) is about 1.0, so the test reported p = 1.0
    # for a result that should be indistinguishable from zero. Every
    # comparison in the first full run came back as exactly 1.0000, which
    # is what gave it away - a genuine null does not land on the same
    # number eight times running. P(X >= min) is the mirror image of
    # P(X <= min) only under symmetry, and reading the tail as "at least
    # the smaller count" is the natural mistake and produces no error.
    extreme = max(b, c)
    p = 2 * _binomial_tail_at_least(extreme, n, 0.5)
    p = min(1.0, p)
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value": p,
        "significant": p < 0.05,
    }


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260601,
    alpha: float = 0.05,
    clusters: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """Paired bootstrap for the difference in means.

    `clusters` groups the paired samples, one id per sample. When it is
    supplied the resampling unit is the cluster, not the sample: a
    patient's several queries share a record set and are not independent,
    so resampling queries treats correlated observations as independent
    and understates the interval. With clusters omitted the behaviour is
    the previous query-level bootstrap, kept so the two can be compared.

    Both are reported in the paper. The clustered interval is the one we
    quote; the unclustered one is narrower and is shown only to make the
    size of the correction visible.
    """
    if len(values_a) != len(values_b):
        raise ValueError("paired samples must be the same length")
    n = len(values_a)
    if n == 0:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"),
                "p_value": float("nan"), "significant": False, "resamples": 0}

    deltas = [a - b for a, b in zip(values_a, values_b)]
    point = sum(deltas) / n
    rng = random.Random(seed)

    if clusters is None:
        units: List[List[float]] = [[d] for d in deltas]
    else:
        if len(clusters) != n:
            raise ValueError("clusters must align with the paired samples")
        grouped: Dict[Any, List[float]] = {}
        for delta, key in zip(deltas, clusters):
            grouped.setdefault(key, []).append(delta)
        units = list(grouped.values())

    k = len(units)
    means: List[float] = []
    for _ in range(resamples):
        picked = [units[rng.randrange(k)] for _ in range(k)]
        flat = [d for unit in picked for d in unit]
        means.append(sum(flat) / len(flat))
    means.sort()
    low = means[int((alpha / 2) * resamples)]
    high = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    # Two-sided bootstrap p-value: how often the resampled difference falls
    # on the other side of zero.
    crossing = sum(1 for m in means if (m <= 0 < point) or (m >= 0 > point))
    p = 2 * min(
        sum(1 for m in means if m <= 0) / resamples,
        sum(1 for m in means if m >= 0) / resamples,
    )
    return {
        "point": point,
        "low": low,
        "high": high,
        "p_value": min(1.0, max(p, crossing / resamples if crossing else 0.0)),
        "significant": (low > 0) or (high < 0),
        "resamples": resamples,
        "n": n,
    }


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Holm-Bonferroni step-down, for comparing many methods at once.

    With eleven methods and several metrics, unadjusted p-values would
    produce a significant result by chance roughly one time in twenty per
    comparison. Ordering the p-values and requiring each to clear a
    progressively stricter threshold controls that.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, Dict[str, float]] = {}
    for index, (key, p) in enumerate(ordered):
        threshold = alpha / (m - index)
        out[key] = {
            "p_raw": p,
            "threshold": threshold,
            "significant": p <= threshold,
            "rank": index + 1,
        }
    return out
