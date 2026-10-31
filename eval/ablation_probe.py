"""Reviewer-requested ablations. Answers reviewer Q2 and Q4.

  Q2  What happens to the valid-time-only baseline if a correction is dated
      to the observation it revises (v(c) == v(e)) rather than to its own
      effective time? This separates "a property of valid-time filtering"
      from "an artefact of our dating convention."

  Q4  How sensitive is the main result to alpha, and to the validity weights?

Run:  python -m ablation_probe
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tapqa.baselines import make_bitemporal, make_valid_time_only  # noqa: E402
from tapqa.dataset import Benchmark, build  # noqa: E402
from tapqa.harness import _superseded_set  # noqa: E402
from tapqa.ours import make_graph_conditioned  # noqa: E402
from tapqa.ranking import Ranking, gold_rank  # noqa: E402

SEED = 20260601


def measure(
    bench: Benchmark,
    ranker: Callable,
) -> Dict[str, Optional[float]]:
    """Independent scorer over the four reported quantities.

    Written against the Ranking rather than the report builder so an ablation
    cannot inherit a bug from the same code path it is meant to probe.
    """
    stale = safe = corr = corr_n = hist = hist_n = 0
    for q in bench.all_queries():
        recs = bench.records_for(q.patient_id)
        if not recs:
            continue
        rk: Ranking = ranker(q, recs)
        if rk.top is None:
            continue
        superseded = _superseded_set(recs)
        # Correctness is read from the rank of any acceptable answer, exactly
        # as the harness does. Comparing rk.top to gold alone would reject a
        # query whose acceptable set legitimately admits another record, and
        # would report this ablation as disagreeing with the main table for
        # that reason alone.
        acceptable = [q.gold_event_id] + [a for a in q.acceptable if a != q.gold_event_id]
        right = gold_rank(rk, acceptable) == 1
        if q.kind == "corrected":
            corr_n += 1
            is_stale = rk.top in superseded
            stale += is_stale
            corr += right
            safe += right and not is_stale
        elif q.kind == "historical":
            hist_n += 1
            hist += right
    return {
        "stale_answer_rate": stale / corr_n if corr_n else None,
        "safe_answer_rate": safe / corr_n if corr_n else None,
        "corrected_accuracy": corr / corr_n if corr_n else None,
        "historical_accuracy": hist / hist_n if hist_n else None,
    }


def line(tag: str, m: Dict[str, Optional[float]]) -> None:
    def f(key: str) -> str:
        v = m[key]
        return "    n/a" if v is None else "%6.1f%%" % (100.0 * v)

    print(
        "  %-38s SAR %s  Corr %s  Hist %s"
        % (tag, f("stale_answer_rate"), f("corrected_accuracy"), f("historical_accuracy"))
    )


def main() -> int:
    base = build(seed=SEED)
    counter = build(seed=SEED, same_valid_time=True)

    out: Dict[str, object] = {
        "seed": SEED,
        "alpha_sweep": {},
        "counterfactual_same_valid_time": {},
    }

    print("A. alpha sensitivity of the graph-conditioned ranker")
    print("   alpha weights the semantic term; 0.70 is the shipped default.")
    for alpha in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        m = measure(base, make_graph_conditioned(alpha=alpha))
        line("alpha=%.2f" % alpha, m)
        out["alpha_sweep"]["%.2f" % alpha] = m

    print()
    print("B. counterfactual: v(c) == v(e), corrections dated to the specimen")
    print("   The default generator makes a correction's valid time follow the")
    print("   observation it revises, so v(c) > v(e) by 1-14 days. This probes")
    print("   how much of each baseline's row is that convention.")
    rows = {
        "valid_time_only_default": measure(base, make_valid_time_only()),
        "valid_time_only_same_vt": measure(counter, make_valid_time_only()),
        "bitemporal_default": measure(base, make_bitemporal()),
        "bitemporal_same_vt": measure(counter, make_bitemporal()),
        "ours_default": measure(base, make_graph_conditioned()),
        "ours_same_vt": measure(counter, make_graph_conditioned()),
    }
    for tag, m in rows.items():
        line(tag, m)
    out["counterfactual_same_valid_time"] = rows

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ablation.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())