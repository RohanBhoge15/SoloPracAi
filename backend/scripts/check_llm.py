"""Verify the OpenRouter key and inspect the model's real output shape.

Run this the moment a key is pasted. It answers three questions that the
rest of the system depends on:

  1. Is the key valid, and is the model reachable on the free tier?
  2. Does the model return clean JSON, or prose wrapped in markdown?
  3. Does it preserve the de-identification placeholders?

The third matters more than it looks. The whole outbound design assumes
`[PT_01]` and `[VAL_K_01]` survive the round trip so exact clinical
values can be re-injected locally. If the model paraphrases or drops
them, every answer loses its numbers.

Usage:
    python scripts/check_llm.py                      # primary (OpenCode)
    python scripts/check_llm.py --provider groq
    python scripts/check_llm.py --provider openrouter --key sk-or-...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Mirrors what the production pipeline actually sends: de-identified
# records with relative-time placeholders and no calendar anchor.
SAMPLE_RECORDS = """[e1] Lab result ([T_MINUS_14D])
  test: Potassium
  value: 4.8
  unit: mEq/L
  STATUS: current
[e2] Lab result ([T_MINUS_11D])
  test: Potassium
  value: 2.1
  unit: mEq/L
  STATUS: SUPERSEDED - a later record corrects this one. Do not state this as a current fact.
[e3] Medication ([T_MINUS_120D])
  drug: Telmisartan
  dose: 40 mg
  frequency: once daily
  STATUS: current"""

QUESTION = "What is the current potassium level for [PT_01]?"


async def check(key: str, provider: str) -> int:
    # Pin one link so the check exercises exactly the provider asked for.
    # Without this, a healthy primary would mask a broken OpenRouter (or
    # vice versa), and the verdict would say nothing about the link named.
    # The key is (re)injected so an explicit --key wins over .env.
    os.environ[{
        "opencode": "OPENCODE_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }[provider]] = key
    os.environ["LLM_PRIMARY"] = provider
    os.environ["LLM_FALLBACK_ENABLED"] = "false"
    os.environ["ENFORCE_DEIDENTIFICATION"] = "true"

    import httpx

    from app.core.config import get_settings
    from app.services.llm import LLMClient, SYSTEM_PROMPT, parse_claims

    get_settings.cache_clear()
    settings = get_settings()
    chain = settings.llm_provider_chain
    if not chain or chain[0][0] != provider:
        print(f"provider {provider!r} is not configured (no key in this env)")
        return 1
    model = chain[0][2]

    print("=" * 66)
    print("SOLUM - LLM verification")
    print("=" * 66)
    print(f"key    : {key[:12]}...{key[-4:] if len(key) > 16 else ''}")
    print(f"model  : {model}")
    print(f"temp   : {settings.llm_temperature}")
    print()

    # ── 1. reachability ───────────────────────────────────────────
    print("[1/3] checking key and model availability ...")

    prompt = (
        "The physician is asking about the present. Use only records marked "
        "current.\n\nRECORDS:\n"
        + SAMPLE_RECORDS
        + f"\n\nQUESTION: {QUESTION}\n\n"
        "Remember: the patient's name is [PT_01]. Use that exact placeholder."
    )

    started = time.perf_counter()
    async with LLMClient(settings) as client:
        result = await client.complete(
            prompt, system=SYSTEM_PROMPT, use_cache=False
        )
    elapsed = (time.perf_counter() - started) * 1000

    if result.degraded and not result.text:
        print(f"  FAILED: {result.model}")
        print()
        print("  Likely causes:")
        print("    - key is wrong or inactive")
        print("    - model id is wrong (free models get renamed often)")
        print("    - rate limited (free tier) - wait a minute and retry")
        return 1

    print(f"  OK - responded in {elapsed:.0f} ms")
    print(f"  model reported: {result.model}")
    print(f"  attempts: {result.attempts}, degraded: {result.degraded}")
    print()

    # ── 2. output shape ───────────────────────────────────────────
    print("[2/3] inspecting output shape")
    print()
    print("-" * 66)
    print("RAW RESPONSE")
    print("-" * 66)
    print(result.text[:1400] or "(empty)")
    print("-" * 66)
    print()

    claims = parse_claims(result.text)
    if claims:
        print(f"  PARSED OK - {len(claims)} claim(s)")
        for index, claim in enumerate(claims, 1):
            print(f"    {index}. {claim['text']}")
            print(f"       event_ids: {claim['event_ids']}")
    else:
        print("  PARSE FAILED - no claims extracted")
        print("  The chat pipeline would fall back to extractive answers.")
        print("  Fix parse_claims() in app/services/llm.py to match this shape.")
    print()

    # ── 3. placeholder preservation ───────────────────────────────
    print("[3/3] checking placeholder preservation")
    print()

    preserved = "[PT_01]" in result.text
    print(f"  [PT_01] preserved : {'YES' if preserved else 'NO'}")

    numbers_expected = {"4.8", "2.1"}
    numbers_found = {n for n in numbers_expected if n in result.text}
    stale_asserted = "2.1" in result.text and "2.1" in result.text.split("2.1")[-1][:200]

    print(f"  values echoed     : {sorted(numbers_found) or 'none'}")
    print(
        f"  stale value risk  : {'YES - it mentioned 2.1' if '2.1' in result.text else 'no'}"
    )
    print()

    # ── verdict ───────────────────────────────────────────────────
    print("=" * 66)
    if not claims:
        print("VERDICT: key works, but output shape needs adapting.")
        print("         See the RAW RESPONSE above and update parse_claims().")
    elif not preserved:
        print("VERDICT: key works and JSON parses.")
        print("         Placeholders were dropped - re-hydration needs a fallback.")
    else:
        print("VERDICT: fully working. Key valid, JSON parses, placeholders held.")
    print("=" * 66)
    print()
    print("Next: add the same questions to a real patient via")
    print("  POST /api/v1/patients/{id}/chat")
    return 0 if claims else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default=None)
    parser.add_argument("--provider", default="opencode",
                        choices=["opencode", "groq", "openrouter"])
    args = parser.parse_args()

    # The key is resolved through Settings, not os.environ: Settings reads
    # solum/.env itself, so this works from a bare shell with no exports.
    # An explicit --key wins over the file.
    key_var = {
        "opencode": "OPENCODE_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }[args.provider]
    if args.key:
        os.environ[key_var] = args.key
    from app.core.config import get_settings
    get_settings.cache_clear()
    key = getattr(get_settings(), {
        "opencode": "opencode_api_key",
        "groq": "groq_api_key",
        "openrouter": "openrouter_api_key",
    }[args.provider]) or ""
    if not key:
        print(f"no key for {args.provider} in this env (and no --key given)")
        print("  paste it in solum/.env, or pass --key")
        return 1
    if key.startswith("your-") or len(key) < 20:
        print(f"key looks like a placeholder: {key!r}")
        return 1
    return asyncio.run(check(key, args.provider))


if __name__ == "__main__":
    raise SystemExit(main())
