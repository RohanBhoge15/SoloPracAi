"""Run TAP-QA and write the results.

    python -m tapqa.run --out results/tapqa

Retrieval-level by default, which is deterministic and therefore
reproducible. The generation arm needs a network call per query and a
free-tier model, so it is opt-in and runs on a stratified subsample.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from tapqa.baselines import build_methods
from tapqa.cache import CachedEncoder
from tapqa.dataset import build
from tapqa.harness import build_report, render_markdown, run


def _load_dense_encoder():
    """A real MedCPT encoder, or None with a stated reason.

    Returning None is reported, never silently substituted. A lexical
    stand-in labelled "dense" would be a fabricated result.
    """
    try:
        from app.services.embeddings import Embedder
        from app.core.config import get_settings
    except ImportError:
        return None, "backend not importable from this interpreter"

    settings = get_settings()
    embedder = Embedder(settings)
    if not embedder.text_available():
        return None, f"encoder unavailable; failed={embedder.status()['failed']}"

    import numpy as np

    def encode(texts, is_query: bool = False):
        """Encode with the correct MedCPT tower.

        The query and article encoders are separately trained; using one
        for both measurably degrades retrieval, so the flag is not
        cosmetic.
        """
        vectors = embedder.encode_texts(list(texts), query=is_query)
        if vectors is None:
            return None
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norm, 1e-9)

    return CachedEncoder(encode, batch_size=64), None


async def _generation_arm(benchmark, encoder, n: int, k: int) -> dict:
    """Run the paired generation arm and stamp provenance onto the result.

    Provenance is not decoration. A stale-answer rate with no record of which
    model produced it is not reproducible, and reproducibility is the whole
    reason the main table was kept at the retrieval level. So the provider and
    model that could have answered are recorded, and a run where no provider
    was reachable is written out as a failure rather than quietly yielding
    zero stale answers -- a zero there would read as a triumph.
    """
    from app.core.config import get_settings
    from app.services.llm import LLMClient
    from tapqa.generation import run_arm

    settings = get_settings()
    if not settings.llm_provider_chain:
        return {
            "ran": False,
            "error": "no LLM provider configured; generation arm did not run",
        }

    client = LLMClient(settings)
    started = time.time()

    def progress(i: int, total: int) -> None:
        if i % 5 == 0 or i == total:
            print(f"  {i}/{total} paired queries", flush=True)

    report = await run_arm(
        benchmark.cases,
        encoder,
        client,
        n_corrected=n,
        n_unchanged=n,
        k_context=k,
        on_progress=progress,
    )
    report["ran"] = True
    report["providers_configured"] = [c[0] for c in settings.llm_provider_chain]
    report["models"] = {c[0]: c[2] for c in settings.llm_provider_chain}
    report["wall_seconds"] = round(time.time() - started, 1)
    report["note"] = (
        "Paired: each question is asked once per arm against the same k "
        "records, differing only in which records the ranker surfaced. "
        "Neither arm annotates validity, so any difference is attributable "
        "to ranking rather than to the supersession annotation."
    )
    return report


def _render_generation(report: dict) -> str:
    """Human-readable summary. Says plainly when the arm did not run."""
    if not report.get("ran"):
        return f"\nGENERATION ARM DID NOT RUN: {report.get('error')}"

    corrected = report["corrected"]
    control = report["unchanged_control"]
    mc = corrected["mcnemar"]

    lines = [
        "",
        "=" * 72,
        f"GENERATION ARM (paired, validity unannotated, k={report['k_context']})",
        "=" * 72,
        (
            f"  corrected  n={corrected['paired']} paired   "
            f"stale rate: ours {corrected['ours_stale_rate']}  "
            f"baseline {corrected['baseline_stale_rate']}"
        ),
        (
            f"  unchanged  n={control['paired']} paired   "
            f"accuracy:  ours {control['ours_accuracy']}  "
            f"baseline {control['baseline_accuracy']}"
        ),
        (
            f"  McNemar exact, two-sided:  discordant={mc['discordant']}  "
            f"p={mc['p']}"
        ),
    ]
    # The contrast is printed next to the outcome every time, because a null
    # with no contrast beside it cannot be told apart from a null caused by
    # the two arms having retrieved the same thing.
    c = report.get("contrast", {}).get("corrected", {})
    if c.get("trials"):
        lines += [
            (
                f"  contrast (corrected):  top-1 identical {c['top1_identical']}"
                f"/{c['trials']},  top-1 == gold  ours "
                f"{c['top1_is_gold_ours']}/{c['trials']}  baseline "
                f"{c['top1_is_gold_baseline']}/{c['trials']}"
            ),
            (
                f"  mean context overlap (Jaccard): "
                f"{c['mean_context_jaccard']}"
            ),
        ]
    lines.append(f"  providers: {report.get('providers_configured')}")
    if report.get("errors"):
        lines.append(f"  {len(report['errors'])} trial(s) errored -- see errors[]")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="TAP-QA evaluation")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--unchanged", type=int, default=200)
    parser.add_argument("--corrected", type=int, default=200)
    parser.add_argument("--contradiction", type=int, default=50)
    parser.add_argument("--out", default="results/tapqa")
    parser.add_argument(
        "--generation",
        type=int,
        default=0,
        help="paired generation arm: N corrected + N unchanged queries, each "
        "asked twice (once per arm), so 4N model calls. 0 disables it. Costs "
        "real requests, so it is opt-in and never part of a retrieval run.",
    )
    parser.add_argument(
        "--generation-k",
        type=int,
        default=5,
        help="records shown to the model per arm in the generation arm. "
        "The headline result is k-dependent and this is the knob: at k=5 a "
        "superseded record arrives together with its correction and the model "
        "reconciles them unprompted, so ranking cannot show up in the answer.",
    )
    parser.add_argument(
        "--no-sweeps",
        action="store_true",
        help="skip the decay grid (much faster; the table is incomplete "
        "without it and the report says so)",
    )
    args = parser.parse_args(argv)

    print("Building benchmark...")
    benchmark = build(
        seed=args.seed,
        n_unchanged=args.unchanged,
        n_corrected=args.corrected,
        n_contradiction=args.contradiction,
    )
    stats = benchmark.stats()
    print(
        f"  {stats['patients']} patients, {stats['records']} records, "
        f"{stats['queries']} queries"
    )
    print(f"  by kind: {stats['by_kind']}")
    print()

    encoder, reason = _load_dense_encoder()
    if encoder is None:
        print(f"DENSE ENCODER UNAVAILABLE: {reason}")
        print("  The dense baseline will fall back to token overlap and the")
        print("  report will record that. It is not reported as dense.")
    else:
        print("Dense encoder: MedCPT (real)")
    print()

    methods = build_methods(encoder, include_sweeps=not args.no_sweeps)
    print(f"Running {len(methods)} methods over {stats['queries']} queries...")
    print()
    results = run(benchmark, methods)

    report = build_report(benchmark, results)
    report["dense_encoder_available"] = encoder is not None
    if encoder is None:
        report["dense_encoder_note"] = reason
    report["sweeps_included"] = not args.no_sweeps
    if hasattr(encoder, "report"):
        # How much work the cache saved, so the run time is attributable.
        report["embedding_cache"] = encoder.report()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (out / "report.md").write_text(markdown, encoding="utf-8")

    # The dataset itself, so the run is reproducible without regenerating.
    dataset = {
        "name": benchmark.name,
        "seed": benchmark.seed,
        "cases": [
            {
                "patient_id": case.patient_id,
                "records": [r.__dict__ for r in case.records],
                "queries": [q.to_dict() for q in case.queries],
            }
            for case in benchmark.cases
        ],
    }
    (out / "dataset.json").write_text(
        json.dumps(dataset, indent=2), encoding="utf-8"
    )

    # The generation arm runs *after* the retrieval report is on disk, so a
    # failure here cannot cost the deterministic results. It also writes to
    # its own file: report.json is the frozen source of every retrieval number
    # in the paper, and mixing a sampled, non-deterministic result into it
    # would make the two indistinguishable to anyone reading the file later.
    if args.generation > 0:
        print()
        print(
            f"Generation arm (k={args.generation_k}): "
            f"{args.generation} corrected + {args.generation} unchanged, "
            f"paired across two arms ({4 * args.generation} model calls)..."
        )
        generation = asyncio.run(
            _generation_arm(
                benchmark, encoder, args.generation, args.generation_k
            )
        )
        (out / f"generation_k{args.generation_k}.json").write_text(
            json.dumps(generation, indent=2), encoding="utf-8"
        )
        print(_render_generation(generation))

    print()
    print(markdown)
    print(f"\nWritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
