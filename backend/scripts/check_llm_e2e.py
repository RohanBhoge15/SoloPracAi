"""End-to-end LLM check against the real seeded patient.

Where `check_llm.py` asks the model a canned question, this runs the
actual production pipeline against Postgres, so it exercises retrieval,
de-identification, generation, the provenance linker, and re-hydration
together.

Usage:
    set OPENROUTER_API_KEY=sk-or-...
    python scripts/check_llm_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = [
    ("current potassium level?", "4.8", "2.1"),
    ("what was the potassium as of 2026-03-12?", "4.8", None),
    ("current medications?", None, None),
]


async def main() -> int:
    from app.core.config import get_settings

    settings = get_settings()
    chain = settings.llm_provider_chain
    if not chain:
        print("no LLM provider key set - need at least one of")
        print("  OPENCODE_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY in solum/.env")
        return 1

    from sqlalchemy import select

    from app.api.deps import load_core_events, make_deidentifier
    from app.db.models import Patient
    from app.db.session import get_engine, get_session_factory
    from app.services.llm import LLMClient
    from app.services.pipeline import AnswerPipeline, build_context
    from app.services.retrieval import lexical_similarity

    print("=" * 70)
    print("end-to-end LLM check  |  chain: "
          + " -> ".join(f"{n}:{m}" for n, _u, m in chain))
    print("=" * 70)
    print()

    failures = 0
    engine = get_engine()

    try:
        async with get_session_factory()() as session:
            patient = (await session.execute(select(Patient))).scalars().first()
            if patient is None:
                print("no patient in database - run scripts/seed_demo.py first")
                return 1
            events = await load_core_events(patient.id, session)
            print(f"patient: {patient.display_name}  ({len(events)} events)")
            print()

            context = build_context(
                str(patient.id),
                events,
                make_deidentifier(patient),
                display_name=patient.display_name,
                age=patient.age,
            )

            async with LLMClient(settings) as llm:
                pipeline = AnswerPipeline(
                    llm=llm, semantic_fn=lexical_similarity,
                    k=settings.retrieval_k, alpha=settings.retrieval_alpha,
                )

                for question, want, avoid in QUESTIONS:
                    started = time.perf_counter()
                    result = await pipeline.answer(context, question)
                    elapsed = (time.perf_counter() - started) * 1000

                    degraded = bool(result.llm and result.llm.degraded)
                    print(f"Q: {question}")
                    print(f"   intent={result.intent.kind}  {elapsed:.0f}ms  "
                          f"degraded={degraded}")

                    text = result.answer.text or ""
                    for line in text.splitlines():
                        print(f"   | {line}")

                    if result.answer.suppressed:
                        for item in result.answer.suppressed:
                            print(f"   SUPPRESSED: {item['text']}")

                    print(f"   citations: {len(result.answer.citations)}")
                    print()

                    if want and want not in text:
                        print(f"   FAIL: expected {want!r} in the answer")
                        failures += 1
                    if avoid and avoid in text:
                        print(f"   FAIL: {avoid!r} should not be asserted as current")
                        failures += 1
    finally:
        await engine.dispose()

    print("=" * 70)
    if failures:
        print(f"{failures} check(s) failed")
    else:
        print("all checks passed")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
