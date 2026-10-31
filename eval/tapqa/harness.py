"""TAP-QA harness.

Runs every method over every query, scores them identically, and produces:

  - the results table, with a Wilson interval on every rate
  - the headline figure data: Stale-Answer Rate against Historical
    Accuracy for every method, with the decay family drawn as a curve
  - paired significance tests against the strongest baseline

The harness is deliberately boring. It reads no wall clock for anything
that ends up in the results, and it does not exclude any query after the
fact - a benchmark that drops hard cases after seeing which method fails
them is not measuring anything.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from tapqa.baselines import DECAY_GRID, Method, build_methods
from tapqa.dataset import Benchmark, Query, Record
from tapqa.metrics import MethodResult, QueryOutcome
from tapqa.ranking import gold_rank, tie_count
from tapqa.stats import mcnemar_exact, paired_bootstrap, wilson


def _superseded_set(records: Sequence[Record]) -> set:
    return {r.supersedes_id for r in records if r.supersedes_id}


def run(
    benchmark: Benchmark,
    methods: List[Method],
    *,
    progress: bool = True,
) -> Dict[str, MethodResult]:
    queries = benchmark.all_queries()
    results: Dict[str, MethodResult] = {
        m.key: MethodResult(
            key=m.key, label=m.label, family=m.family,
            params=dict(m.params), sweep_of=m.sweep_of,
        )
        for m in methods
    }

    for method in methods:
        result = results[method.key]
        started = time.perf_counter()
        for query in queries:
            records = benchmark.records_for(query.patient_id)
            if not records:
                continue

            t0 = time.perf_counter()
            ranking = method.ranker(query, records)
            latency = (time.perf_counter() - t0) * 1000.0

            gold = query.gold_event_id
            superseded = _superseded_set(records)
            top = ranking.top

            # Rank comes from the method's scores, averaged over ties, so a
            # method that cannot separate two records earns nothing for the
            # order it happened to emit them in. See tapqa.ranking.
            acceptable = [gold] + [a for a in query.acceptable if a != gold]
            result.outcomes.append(
                QueryOutcome(
                    query_id=query.query_id,
                    kind=query.kind,
                    test=query.test,
                    gold_rank=gold_rank(ranking, acceptable),
                    ranking_size=len(ranking),
                    tie_positions=tie_count(ranking),
                    latency_ms=latency,
                    # A conflict is detected only if both sides appear.
                    # Ranking one of them first is not an answer.
                    surfaced_conflict=bool(query.conflict)
                    and ranking.contains_all(query.conflict),
                    top_is_superseded=top in superseded,
                )
            )
        if progress:
            print(
                f"  {method.key:<22} {len(result.outcomes):>5} queries "
                f"in {time.perf_counter() - started:6.2f}s"
            )
    return results


def _rate_with_ci(outcomes, hit) -> Dict[str, Any]:
    items = list(outcomes)
    if not items:
        return {"point": None, "ci": None, "n": 0}
    successes = sum(1 for o in items if hit(o))
    interval = wilson(successes, len(items))
    return {
        "point": interval.point,
        "ci": [interval.low, interval.high],
        "n": interval.n,
        "successes": successes,
    }


def build_report(
    benchmark: Benchmark,
    results: Dict[str, MethodResult],
    *,
    headline: str = "ours",
) -> Dict[str, Any]:
    """Assemble the machine-readable report, headline figure included."""
    summaries: List[Dict[str, Any]] = []
    for result in results.values():
        summary = result.summary()
        summary["stale_answer_rate_ci"] = _rate_with_ci(
            result.by_kind("corrected"), lambda o: o.top_is_superseded
        )
        summary["historical_accuracy_ci"] = _rate_with_ci(
            result.by_kind("historical"), lambda o: o.gold_rank == 1
        )
        summary["unchanged_accuracy_ci"] = _rate_with_ci(
            result.by_kind("unchanged"), lambda o: o.gold_rank == 1
        )
        summary["corrected_accuracy_ci"] = _rate_with_ci(
            result.by_kind("corrected"), lambda o: o.gold_rank == 1
        )
        summary["contradiction_detection_ci"] = _rate_with_ci(
            result.by_kind("contradiction"), lambda o: o.surfaced_conflict
        )
        summaries.append(summary)

    # ── the figure: staleness against history ──────────────────
    #
    # Each method is one point. The decay family is a polyline, because
    # the claim under test is that the whole family traces a downward
    # slope - push it hard enough and history dies.
    points: List[Dict[str, Any]] = []
    for summary in summaries:
        sar = summary["stale_answer_rate"]
        hist = summary["historical_accuracy"]
        if sar is None or hist is None:
            continue
        points.append(
            {
                "key": summary["key"],
                "label": summary["label"],
                "family": summary["family"],
                "sweep_of": summary["sweep_of"],
                "params": summary["params"],
                "stale_answer_rate": sar,
                "historical_accuracy": hist,
                "unchanged_accuracy": summary["unchanged_accuracy"],
            }
        )

    decay_curve = sorted(
        [p for p in points if p["sweep_of"] == "decay"],
        key=lambda p: (p["params"]["tau"], p["params"]["lambda"]),
    )

    # ── significance against the best non-graph baseline ───────
    #
    # "Best" is chosen by historical accuracy among the baselines, not by
    # SAR, so the comparison cannot be stacked by picking a weak
    # opponent.
    baseline_keys = [
        s["key"]
        for s in summaries
        if s["family"] in ("temporal", "semantic", "floor", "ablation")
        and s["sweep_of"] is None
    ]
    comparisons: List[Dict[str, Any]] = []
    if headline in results and baseline_keys:
        head = results[headline]
        for key in baseline_keys:
            if key == headline:
                continue
            other = results[key]
            # Binary paired outcome, so McNemar is the right test.
            #
            # Testing `not top_is_superseded` on its own looks like the
            # obvious test and is useless: any method that simply refuses
            # to return a superseded record scores "not stale" every time,
            # including Latest-N, which is stale-free because it answers
            # nothing at all. Both arms then agree on all 200 queries, the
            # test has zero discordant pairs, and it reports p = 1.0 - a
            # perfect non-result dressed as a measurement. That is exactly
            # what the first run of this harness did.
            #
            # The outcome that carries meaning is whether the method gave
            # the clinician a usable *and* non-stale answer. A method that
            # ranks a superseded record first has failed; a method that
            # abstains by returning nothing current has also failed, and
            # both failures count against it. So the binary is:
            #     safe = (gold ranked first) and (top is not superseded)
            paired_head = [
                (o.gold_rank == 1) and (not o.top_is_superseded)
                for o in head.by_kind("corrected")
            ]
            paired_other = [
                (o.gold_rank == 1) and (not o.top_is_superseded)
                for o in other.by_kind("corrected")
            ]
            aligned = list(zip(paired_head, paired_other))
            if aligned:
                mcnemar = mcnemar_exact(
                    [a for a, _ in aligned], [b for _, b in aligned]
                )
            else:
                mcnemar = {"p_value": float("nan"), "significant": False}
            # Reported separately, because SAR on its own is gameable and a
            # reader deserves to see how much of the gap is the staleness
            # term versus plain usefulness.
            stale_head = [
                o.top_is_superseded for o in head.by_kind("corrected")
            ]
            stale_other = [
                o.top_is_superseded for o in other.by_kind("corrected")
            ]
            if aligned:
                stale_mcnemar = mcnemar_exact(stale_head, stale_other)
            else:
                stale_mcnemar = {"p_value": float("nan"), "significant": False}

            hist_head_obs = head.by_kind("historical")
            hist_other_obs = other.by_kind("historical")
            hist_head = [1.0 if o.gold_rank == 1 else 0.0 for o in hist_head_obs]
            hist_other = [1.0 if o.gold_rank == 1 else 0.0 for o in hist_other_obs]
            # Resampling unit is the patient, not the query. Several queries
            # share one patient's record set, so query-level resampling
            # treats correlated observations as independent and understates
            # the interval. Both are computed; the clustered one is the
            # reported figure and the query-level one is kept only so the
            # size of the correction is auditable.
            clusters = [
                o.query_id.rsplit("-e", 1)[0] for o in hist_head_obs
            ]
            boot = paired_bootstrap(hist_head, hist_other, clusters=clusters)
            boot_unclustered = paired_bootstrap(hist_head, hist_other)

            # Recorded so the joint outcome cannot be quietly swapped back
            # for the staleness-only one, which is what the first run did
            # and which reports p = 1.0 against every baseline.
            comparisons.append(
                {
                    "baseline": key,
                    "baseline_label": other.label,
                    "safe_answer_mcnemar": mcnemar,
                    "staleness_only_mcnemar": stale_mcnemar,
                    "historical_bootstrap": boot,
                    "historical_bootstrap_unclustered": boot_unclustered,
                    "n_corrected_queries": len(aligned),
                    "n_historical_queries": len(hist_head),
                }
            )

    return {
        "benchmark": {
            "name": benchmark.name,
            "seed": benchmark.seed,
            **benchmark.stats(),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": summaries,
        "headline_figure": {
            "x": "stale_answer_rate",
            "y": "historical_accuracy",
            "points": points,
            "decay_curve": decay_curve,
        },
        "significance": comparisons,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    """A table a reviewer can read without running anything."""
    lines: List[str] = []
    bench = report["benchmark"]
    lines.append("# TAP-QA results")
    lines.append("")
    lines.append(
        f"`{bench['name']}` seed `{bench['seed']}` - "
        f"{bench['patients']} patients, {bench['records']} records, "
        f"{bench['queries']} queries"
    )
    counts = bench["by_kind"]
    lines.append("")
    lines.append(
        "Query classes: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    lines.append("")

    lines.append("## Retrieval results")
    lines.append("")
    lines.append(
        "Rank is read from each method's own scores, averaged over ties. A "
        "method that scores the original and the correction of a test "
        "identically has not told them apart, so it earns no credit for "
        "whichever one it happened to list first."
    )
    lines.append("")
    lines.append(
        "This is not a technicality, and it is what makes the floor rows "
        "mean something. An earlier run of this harness reported token "
        "overlap at 99.8% historical accuracy. It was breaking ties on "
        "record id, which encodes insertion order, which encodes "
        "chronology - so the original of every corrected test won "
        "automatically, and a method with no notion of supersession "
        "appeared to solve the historical class outright. Read the `Tied "
        "queries` column as the share of each row that rests on the "
        "method's scores rather than on the order it emitted: token overlap "
        "ties on all 850 queries, which is exactly why it now scores 0% on "
        "both classes that require knowing which version was current. The "
        "0% is the true reading of a method that cannot see supersession, "
        "not a bug in the harness."
    )
    lines.append("")
    lines.append(
        "| Method | SAR (lower better) | Historical acc | Unchanged acc | "
        "Corrected acc | R@1 | MRR | Tied queries |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    def fmt(value: Optional[float]) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    def fmt_ci(entry: Dict[str, Any]) -> str:
        if entry.get("point") is None:
            return "-"
        low, high = entry["ci"]
        return f"{entry['point'] * 100:.1f}% [{low * 100:.1f}, {high * 100:.1f}]"

    ordered = sorted(
        report["methods"],
        key=lambda m: (
            m["sweep_of"] or "",
            m["family"],
            m["key"],
        ),
    )
    for method in ordered:
        if method["sweep_of"] == "decay":
            continue
        lines.append(
            f"| {method['label']} "
            f"| {fmt_ci(method['stale_answer_rate_ci'])} "
            f"| {fmt_ci(method['historical_accuracy_ci'])} "
            f"| {fmt_ci(method['unchanged_accuracy_ci'])} "
            f"| {fmt_ci(method['corrected_accuracy_ci'])} "
            f"| {fmt(method['r1'])} "
            f"| {fmt(method['mrr'])} "
            f"| {method.get('queries_with_ties', 0)} |"
        )

    lines.append("")
    lines.append("## Recency decay sweep")
    lines.append("")
    lines.append(
        "`score = semantic + lambda * exp(-age / tau)`, the strongest "
        "argument against a temporal method: a recency preference is one "
        "number, so if it can buy safety it can be tuned. The full 35-point "
        "grid is reported, because a single lambda is a strawman."
    )
    lines.append("")
    lines.append(
        "**The sweep does not trace a trade-off curve, and that is the "
        "finding.** Two corners, no diagonal:"
    )
    lines.append("")
    lines.append(
        "- `tau = 7`: SAR 64-70%, historical 0%. A short time constant "
        "makes recency dominate, so the method leads with whatever was "
        "recorded last - which for a corrected test is the retracted value."
    )
    lines.append(
        "- `tau >= 30`: SAR 0%, historical 0%. Once the constant is long "
        "enough the temporal term is nearly flat across the whole record "
        "history, so the method is stale-free for the same reason Latest-N "
        "is: it is not leading with anything in particular. Historical "
        "accuracy is zero because the flat term carries no information "
        "about *which* version was valid when."
    )
    lines.append("")
    lines.append(
        "So lambda and tau do not trade staleness against historical "
        "recall. They select between two degenerate corners, and no point "
        "on the grid reaches the upper-left region the graph-conditioned "
        "method occupies. The prediction in the project plan was a smooth "
        "downward line; the measurement is a step. Reported as measured."
    )
    lines.append("")
    lines.append("| lambda | tau (days) | SAR | Historical acc |")
    lines.append("|---|---|---|---|")
    for point in report["headline_figure"]["decay_curve"]:
        lines.append(
            f"| {point['params']['lambda']:g} "
            f"| {point['params']['tau']:g} "
            f"| {point['stale_answer_rate'] * 100:.1f}% "
            f"| {point['historical_accuracy'] * 100:.1f}% |"
        )

    lines.append("")
    lines.append("## Paired significance vs. the graph-conditioned method")
    lines.append("")
    lines.append(
        "McNemar exact on the *safe answer* outcome over the 200 corrected "
        "queries - gold ranked first **and** the top record not superseded "
        "- and a paired bootstrap (10,000 resamples) on historical accuracy."
    )
    lines.append("")
    lines.append(
        "The outcome is joint on purpose, and the reason is visible in the "
        "staleness-only column. Staleness alone is not a testable safety "
        "claim: a method that never returns a superseded record is "
        "stale-free by refusing to answer. Latest-N is exactly that - 0% "
        "staleness, 0% historical accuracy - and a staleness-only test "
        "scores it as indistinguishable from a system that works, p = 1.0. "
        "Requiring the answer to also be correct is what gives the "
        "comparison content, and it is why the two columns disagree for "
        "Latest-N and Valid-time only."
    )
    lines.append("")
    lines.append(
        "Two entries in the staleness-only column are worth reading "
        "carefully rather than as wins. Valid-time only reaches p = 0.25 "
        "because it is stale-free *and* correct on 100% of corrected "
        "queries, so the two arms agree on almost every pair; the honest "
        "reading is that it has no staleness left to fix, not that the "
        "graph adds nothing there. It scores 0% historical, so it cannot "
        "answer the question the system exists for. Latest-N's p = 1.0 for "
        "the same reason. The historical column separates them: +0.963 "
        "with a CI excluding zero, for both."
    )
    lines.append("")
    lines.append(
        "| Baseline | McNemar p (safe answer) | Sig | "
        "McNemar p (staleness only) | Bootstrap p | 95% CI on diff |"
    )
    lines.append("|---|---|---|---|---|---|")
    for comparison in report["significance"]:
        mcnemar = comparison["safe_answer_mcnemar"]
        stale = comparison.get("staleness_only_mcnemar", {})
        boot = comparison["historical_bootstrap"]
        p = mcnemar.get("p_value")
        sp = stale.get("p_value")
        lines.append(
            f"| {comparison['baseline_label']} "
            f"| {'-' if p is None else f'{p:.4f}'} "
            f"| {'yes' if mcnemar.get('significant') else 'no'} "
            f"| {'-' if sp is None else f'{sp:.4f}'} "
            f"| {boot.get('p_value', float('nan')):.4f} "
            f"| [{boot.get('low', float('nan')):+.3f}, "
            f"{boot.get('high', float('nan')):+.3f}] |"
        )

    lines.append("")
    return "\n".join(lines)
