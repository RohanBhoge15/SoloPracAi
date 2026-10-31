"""k=1 generation arm. Reviewer Q3: does first-place ranking matter when
the model sees a single record and cannot reconcile two versions?

Arms are identical to the k=5 and k=2 runs except k_context=1: our ranker
vs dense-only, no validity annotation in either arm, paired per query, same
model, unchanged control included so a regression is detectable.

Run:  python generation_k1.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "backend")))

from app.core.config import get_settings  # noqa: E402
from app.services.llm import LLMClient  # noqa: E402

from tapqa.dataset import build  # noqa: E402
from tapqa.generation import run_arm  # noqa: E402
from tapqa.run import _load_dense_encoder  # noqa: E402

OUT = os.path.join(HERE, "results", "generation_k1.json")


async def main() -> int:
    settings = get_settings()
    if not settings.llm_provider_chain:
        print("no LLM provider configured; refusing to fabricate a null")
        return 1

    bench = build(seed=20260601)
    encoder, err = _load_dense_encoder()
    print("encoder:", "loaded" if encoder else "UNAVAILABLE (%s)" % err)

    cases = bench.cases if hasattr(bench, "cases") else list(bench.all_cases())
    client = LLMClient(settings)

    def progress(done: int, total: int) -> None:
        print("  %d/%d" % (done, total), flush=True)

    report = await run_arm(
        cases,
        encoder,
        client,
        n_corrected=40,
        n_unchanged=40,
        k_context=1,
        on_progress=progress,
    )

    report["k_context"] = 1
    report["purpose"] = (
        "k=1 arm. At k=1 the model sees one record per arm and cannot "
        "reconcile two versions, so this is the direct test of whether "
        "first-place ranking changes the answer."
    )
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    corr = report.get("corrected", {})
    print("\n=== k=1 corrected arm ===")
    for key in (
        "trials",
        "paired",
        "ours_stale_rate",
        "baseline_stale_rate",
        "mcnemar_p",
        "discordant",
    ):
        if key in corr:
            print("  %-22s %s" % (key, corr[key]))
    print("  control:", json.dumps(report.get("unchanged_control", {}))[:200])
    print("  errors:", report.get("errors"))
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))