"""The generation arm: does better ranking change what the model says?

The retrieval table answers "did the right record reach the top". It cannot
answer the question a reviewer actually cares about, which is whether that
matters once a language model is reading the records. A ranker can put the
right record first and the answer can still come out stale, because the model
is free to prefer whichever record it finds most salient.

This module runs that experiment. It is paired: the same question is asked
twice, once with each arm's context, so the comparison controls for question
difficulty and for the model's run-to-run variation.

**The one design decision everything else hangs off: neither arm annotates
validity.** `AnswerPipeline` stamps every record with `STATUS: SUPERSEDED`,
and that annotation is a defence *independent* of the ranker. If both arms
carried it, the model would be told which record to distrust and any
difference in the answers would be attributable to the annotation rather than
to the ranking. So both arms are rendered by
`render_records_unannotated`, and the only difference between them is which
records arrived and in what order. That is what makes the result
attributable to the ranker.

**The specificity control.** Part of the sample is drawn from the `unchanged`
class, where nothing is ever corrected and staleness is therefore impossible.
A ranker that genuinely helps should move the corrected class and leave the
unchanged class alone. A ranker that merely reorders text, or that helps for
some unrelated reason, moves both. Reporting only the corrected class would
hide exactly that failure, so the control is reported alongside it.

Scoring is deliberately blunt and mechanical: did the answer assert the
retracted value instead of the corrected one? This is not a claim about
clinical quality and it is insensitive to phrasing. It is the narrowest
operationalisation of "the model repeated a retracted number" that is still
falsifiable.
"""

from __future__ import annotations

import re
import asyncio
from dataclasses import dataclass, field
from datetime import date
from math import comb
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.services.llm import LLMClient
from app.services.pipeline import render_records_unannotated
from app.services.pii import Deidentifier

from tapqa.baselines import make_dense_only
from tapqa.dataset import Query, Record
from tapqa.ours import make_graph_conditioned, to_core


#: Records shown to the model per arm. Held equal because an arm that simply
#: sees more text is not being compared against one that sees less.
K_CONTEXT = 5


#: Seconds to wait between outbound calls.
#:
#: Not politeness, arithmetic. A free endpoint will start refusing once it is
#: hit hard: the first run of this arm completed 38 calls and then every
#: subsequent call came back degraded, because both providers were refusing
#: at once (OpenCode rate-limited, OpenRouter's daily quota already spent).
#: 61 of 80 trials were lost that way. At ~5s per call the arm wants 160
#: calls inside about 13 minutes, which is well inside what the endpoint
#: tolerates when it is not asked for them back to back.
PACE_SECONDS = 2.0

#: Retries for a single call that came back empty or degraded.
#:
#: A transient refusal is not a measurement. Retrying it is what separates
#: "the model declined" from "the endpoint was busy", and collapsing the two
#: would silently convert provider load into evidence about the ranker.
CALL_ATTEMPTS = 4


async def _call(
    client: LLMClient,
    prompt: str,
    *,
    system: str,
    deid: Deidentifier,
) -> tuple:
    """One model call, paced and retried. Returns `(text, error)`.

    `error` is None on success. A persistent failure is reported as an error
    rather than as an empty answer, because an empty answer scores as "did not
    state the retracted value" and would read as a clean result.
    """
    scrubbed = deid.scrub(prompt)
    last = "no attempt made"
    for attempt in range(1, CALL_ATTEMPTS + 1):
        if attempt > 1:
            await asyncio.sleep(5.0 * attempt)
        await asyncio.sleep(PACE_SECONDS)
        try:
            result = await client.complete(
                scrubbed, system=system, use_cache=True
            )
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            continue

        text = (getattr(result, "text", "") or "").strip()
        if text and not getattr(result, "degraded", False):
            return text, None
        last = (
            f"empty or degraded answer after {attempt} attempt(s) "
            f"(model={getattr(result, 'model', '?')}, "
            f"provider={getattr(result, 'provider', '') or 'none'})"
        )
    return "", last


def _prompt(question: str, records_block: str) -> str:
    """Ask the benchmark's question verbatim against a rendered context.

    The question is not reworded. Rewriting it would mean this arm measures a
    different dataset from the retrieval arm and the two tables would not be
    comparable, which is the only reason to have both.
    """
    return (
        "Answer the question using only the clinical records below. "
        "State the value plainly. If the records disagree, say so.\n\n"
        f"RECORDS:\n{records_block}\n\nQUESTION: {question}"
    )


#: Anchor for the de-identifier's relative-time conversion, pinned rather
#: than defaulting to today. `scrub` rewrites calendar dates as offsets from
#: an anchor, so leaving it as the wall clock would make the prompt -- and
#: therefore the cache key and the recorded provenance -- differ between two
#: runs of the same benchmark. It matches the instant `ours.py` uses as its
#: notion of now.
ANCHOR = date(2026, 6, 1)


#: This arm does NOT use the shipping SYSTEM_PROMPT, which requires
#: `{"claims": [{"text": ..., "event_ids": [...]}]}`.
#:
#: That contract is for the product, where the provenance linker has to
#: resolve every claim back to a record or the answer is discarded. Here the
#: question is "does ranking change the number the model states", and forcing
#: a claims envelope measures the envelope: asked for citations to records it
#: was never given ids for, the model returns `{"claims": []}` and every
#: answer scores as a non-answer. Both arms would then read as 0% stale and
#: the experiment would report a clean null for the wrong reason.
#:
#: Deliberately no validity language here either. Telling the model that a
#: record is superseded is the defence under test, and saying so in the system
#: prompt would hand both arms the answer.
ARM_SYSTEM = (
    "You are reading de-identified clinical record excerpts and answering a "
    "question about them. Use only the records provided. State the value "
    "plainly and briefly. If the records disagree with each other, say so "
    "and give each value you see. Do not explain your reasoning and do not "
    "add clinical advice."
)


def _value_of(record: Optional[Record]) -> Optional[str]:
    """The value a record asserts, as a string, or None if it asserts none."""
    if record is None:
        return None
    payload = record.payload or {}
    for key in ("value", "dose", "result"):
        if payload.get(key) is not None:
            return str(payload[key])
    return None


#: A number as written in clinical text: optional sign, digits, optional
#: decimal part, optional thousands separators.
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _mentions(answer: str, value: Optional[str]) -> bool:
    """Whether `answer` states `value`.

    Numbers are compared *numerically*, not textually. This is not a
    refinement, it is a correctness requirement, and the earlier
    textual version was quietly wrong in a way that would have been reported
    as a result: a record holding ``52.0`` and a model answering "52.0 mg/dL"
    were scored as *not* mentioning it, because the integer branch matched
    ``52`` and then rejected it for being followed by a decimal point.

    That is not a harmless edge case. It broke in both directions at once --
    ``said_gold`` was false when the model was right, and ``said_retracted``
    was false when the model was quoting the retracted value -- so a genuinely
    stale answer scored as clean. A benchmark that cannot detect the failure
    it exists to detect reports 0% and looks like a triumph.

    Comparing floats also removes the boundary problem entirely: "9.45" is not
    9.4, and no regex guard is needed to keep them apart. Thousands
    separators are stripped for the same reason.

    Free text (a drug name) still matches on word boundaries.
    """
    if not value:
        return False
    text = str(value).strip()
    if not text:
        return False

    try:
        target = float(text.replace(",", ""))
    except ValueError:
        return (
            re.search(rf"(?<!\w){re.escape(text)}(?!\w)", answer, re.I)
            is not None
        )

    for token in _NUMBER.findall(answer):
        try:
            if float(token.replace(",", "")) == target:
                return True
        except ValueError:
            continue
    return False


def _mcnemar_exact(ours_only: int, baseline_only: int) -> Dict[str, Any]:
    """Exact two-sided binomial McNemar on the discordant pairs.

    Two-sided: the direction of the effect was not predicted in advance, so a
    one-sided test would quietly halve the p-value and overstate the result.
    The doubling is capped at 1 because the exact test can exceed it when the
    counts are small and balanced.

    Exact rather than chi-square: on a stratified subsample the discordant
    counts are small, and small counts are precisely where the chi-square
    approximation is least trustworthy.
    """
    n = ours_only + baseline_only
    if n == 0:
        return {
            "discordant": 0,
            "p": None,
            "note": "no discordant pairs; the arms never differed",
        }
    k = min(ours_only, baseline_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {
        "discordant": n,
        "ours_only_stale": ours_only,
        "baseline_only_stale": baseline_only,
        "two_sided": True,
        "p": round(min(1.0, 2 * tail), 6),
    }


@dataclass
class ArmOutcome:
    """One paired trial. Per-arm failures are kept, never dropped: an arm
    that failed more often would otherwise look better by attrition."""

    query_id: str
    kind: str
    question: str
    gold_event_id: str = ""
    ours_stale: Optional[bool] = None
    baseline_stale: Optional[bool] = None
    ours_correct: Optional[bool] = None
    baseline_correct: Optional[bool] = None
    error: Optional[str] = None
    ours_context: List[str] = field(default_factory=list)
    baseline_context: List[str] = field(default_factory=list)
    ours_answer: str = ""
    baseline_answer: str = ""

    @property
    def paired(self) -> bool:
        """Both arms produced an answer, so the pair is scoreable.

        Deliberately *not* "both staleness verdicts are known". Staleness is
        undefined on the `unchanged` control by construction -- nothing was
        ever corrected -- so a staleness-based definition would score every
        control trial as unpaired and silently discard the specificity check
        that the whole design rests on.
        """
        if self.error:
            return False
        return bool(self.ours_answer) and bool(self.baseline_answer)


def _stratified(
    cases: Sequence[Any], n_corrected: int, n_unchanged: int
) -> List[tuple]:
    """Draw the subsample, reproducibly. Returns `(case, query)` pairs.

    A `Case` holds one patient and several queries, so the sampling unit is
    the pair, not the case. Stratified by class and sorted by query id before
    slicing, so the draw is not merely "whatever the generator emitted first"
    -- which would correlate the sample with record ordering and bias it.
    """
    pairs = [(c, q) for c in cases for q in c.queries]
    picked: List[tuple] = []
    for kind, want in (("corrected", n_corrected), ("unchanged", n_unchanged)):
        pool = sorted(
            (p for p in pairs if p[1].kind == kind),
            key=lambda p: p[1].query_id,
        )
        picked += pool[:want]
    return picked


def _rate(values: Sequence[Optional[bool]]) -> Optional[float]:
    scored = [v for v in values if v is not None]
    if not scored:
        return None
    return round(sum(1 for v in scored if v) / len(scored), 4)


def _contrast(outcomes: List[ArmOutcome]) -> Dict[str, Any]:
    """How much the two arms actually differed, before asking about answers.

    A null result in this experiment is uninterpretable on its own. "Neither
    arm went stale" is consistent with two very different worlds:

      1. the rankers barely differed, so there was nothing to detect; or
      2. the rankers differed maximally and the model did not care.

    Those demand opposite conclusions about the method, and a paper that
    reports the null without this is asking the reader to guess. So the
    retrieval-level contrast is measured and reported next to the
    generation-level outcome, every time.
    """
    scored = [o for o in outcomes if o.ours_context and o.baseline_context]
    if not scored:
        return {"trials": 0}

    top1_same = sum(
        1 for o in scored if o.ours_context[0] == o.baseline_context[0]
    )
    jaccards = []
    for o in scored:
        a, b = set(o.ours_context), set(o.baseline_context)
        union = a | b
        jaccards.append(len(a & b) / len(union) if union else 1.0)

    ours_gold = sum(
        1 for o in scored if o.ours_context[0] == o.gold_event_id
    )
    base_gold = sum(
        1 for o in scored if o.baseline_context[0] == o.gold_event_id
    )

    return {
        "trials": len(scored),
        "top1_identical": top1_same,
        "mean_context_jaccard": round(sum(jaccards) / len(jaccards), 4),
        "top1_is_gold_ours": ours_gold,
        "top1_is_gold_baseline": base_gold,
        "note": (
            "Retrieval-level contrast between the arms, independent of what "
            "the model said. top1_identical=0 means the arms disagreed on "
            "first place on every trial, so a null on the answers is a real "
            "null rather than an absence of contrast."
        ),
    }


def _summarise(
    outcomes: List[ArmOutcome], k_context: int
) -> Dict[str, Any]:
    """Reduce trials to the numbers the paper reports."""
    corrected = [o for o in outcomes if o.kind == "corrected"]
    unchanged = [o for o in outcomes if o.kind == "unchanged"]
    paired = [o for o in corrected if o.paired]

    ours_only = sum(1 for o in paired if o.ours_stale and not o.baseline_stale)
    baseline_only = sum(
        1 for o in paired if o.baseline_stale and not o.ours_stale
    )

    return {
        "k_context": k_context,
        "arms": {
            "ours": "validity-aware ranker (shipping retriever)",
            "baseline": "dense-only, no temporal term",
        },
        "annotated": False,
        "contrast": {
            "all": _contrast(outcomes),
            "corrected": _contrast(corrected),
            "unchanged_control": _contrast(unchanged),
        },
        "corrected": {
            "trials": len(corrected),
            "paired": len(paired),
            "ours_stale_rate": _rate([o.ours_stale for o in corrected]),
            "baseline_stale_rate": _rate([o.baseline_stale for o in corrected]),
            "ours_accuracy": _rate([o.ours_correct for o in corrected]),
            "baseline_accuracy": _rate([o.baseline_correct for o in corrected]),
            "mcnemar": _mcnemar_exact(ours_only, baseline_only),
        },
        "unchanged_control": {
            "trials": len(unchanged),
            "paired": sum(1 for o in unchanged if o.paired),
            "ours_stale_rate": _rate([o.ours_stale for o in unchanged]),
            "baseline_stale_rate": _rate([o.baseline_stale for o in unchanged]),
            "ours_accuracy": _rate([o.ours_correct for o in unchanged]),
            "baseline_accuracy": _rate([o.baseline_correct for o in unchanged]),
        },
        "errors": [
            {"query_id": o.query_id, "error": o.error}
            for o in outcomes
            if o.error
        ],
        # The trials themselves, verbatim. Kept because an aggregate that
        # cannot be re-examined is not evidence: when the scorer turned out to
        # be wrong, the only way to find out was to re-run and read the model's
        # actual words, since the first report had thrown them away. Every
        # verdict below can now be checked by hand against the text that
        # produced it.
        "trials": [
            {
                "query_id": o.query_id,
                "kind": o.kind,
                "question": o.question,
                "ours_answer": o.ours_answer,
                "baseline_answer": o.baseline_answer,
                "ours_stale": o.ours_stale,
                "baseline_stale": o.baseline_stale,
                "ours_correct": o.ours_correct,
                "baseline_correct": o.baseline_correct,
                "ours_context": o.ours_context,
                "baseline_context": o.baseline_context,
            }
            for o in outcomes
        ],
    }


async def run_arm(
    cases: Sequence[Any],
    embedder: Optional[Callable[..., Any]],
    client: LLMClient,
    *,
    n_corrected: int = 40,
    n_unchanged: int = 40,
    k_context: int = K_CONTEXT,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Run the paired generation comparison and return a report."""
    sample = _stratified(cases, n_corrected, n_unchanged)
    ours = make_graph_conditioned(embedder=embedder)
    baseline = make_dense_only(embedder)

    by_id: Dict[str, Record] = {}
    for case in cases:
        for record in case.records:
            by_id[record.event_id] = record

    outcomes: List[ArmOutcome] = []
    total = len(sample)

    for i, (case, query) in enumerate(sample, start=1):
        outcome = ArmOutcome(
            query_id=query.query_id,
            kind=query.kind,
            question=query.text,
            gold_event_id=query.gold_event_id,
        )
        outcomes.append(outcome)

        try:
            # Both factories *return* the ranker function; neither returns an
            # object with a `.ranker` attribute.
            ours_ids = list(ours(query, case.records).ids)[:k_context]
            base_ids = list(baseline(query, case.records))[:k_context]
        except Exception as exc:  # noqa: BLE001
            outcome.error = f"ranking: {type(exc).__name__}: {exc}"
            if on_progress:
                on_progress(i, total)
            continue

        outcome.ours_context = ours_ids
        outcome.baseline_context = base_ids

        gold = by_id.get(query.gold_event_id)
        gold_value = _value_of(gold)
        retracted_value = None
        if gold is not None and gold.supersedes_id:
            retracted_value = _value_of(by_id.get(gold.supersedes_id))

        def block(ids: Sequence[str]) -> str:
            return render_records_unannotated(
                [to_core(by_id[e]) for e in ids if e in by_id]
            )

        # The rendered records carry calendar dates, so the prompt goes
        # through the same de-identifier the product uses. Without this the
        # outbound PII gate blocks the request -- `DeidentificationBlocked` --
        # which is the gate working correctly, not a workaround being needed.
        # The anchor is pinned so offsets, and therefore cache keys, are
        # reproducible across runs.
        deid = Deidentifier(patient_id=case.patient_id, today=ANCHOR)

        for arm, ids in (("ours", ours_ids), ("baseline", base_ids)):
            text, failure = await _call(
                client,
                _prompt(query.text, block(ids)),
                system=ARM_SYSTEM,
                deid=deid,
            )
            if failure is not None:
                outcome.error = f"{arm}: {failure}"
                continue

            said_gold = _mentions(text, gold_value)
            # Stale means the retracted value was asserted *instead of* the
            # corrected one. Quoting both, with the correction present, is the
            # model behaving correctly and is not counted as stale.
            said_retracted = _mentions(text, retracted_value)
            stale = said_retracted and not said_gold

            if arm == "ours":
                outcome.ours_answer = text
                outcome.ours_correct = said_gold
                outcome.ours_stale = stale if retracted_value is not None else None
            else:
                outcome.baseline_answer = text
                outcome.baseline_correct = said_gold
                outcome.baseline_stale = (
                    stale if retracted_value is not None else None
                )

        if on_progress:
            on_progress(i, total)

    return _summarise(outcomes, k_context)