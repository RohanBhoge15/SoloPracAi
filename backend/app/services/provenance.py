"""Provenance-locked generation and the temporal-entailment verifier.

Two axes, deliberately separable so they can be ablated independently:

    Axis A (faithfulness)  is the claim entailed by the cited event?
                           Prior work — VERICITE does this.
    Axis B (temporal validity)
                           is that event in a state compatible with the
                           claim's tense?  This project's contribution.

A claim that passes A but fails B is the exact failure SoloPrac is built to
stop: perfectly grounded, perfectly cited, clinically false.

Enforcement is structural. The model proposes; the linker decides. A
prompt instruction telling the model to cite properly is a suggestion,
and this is the reason the check exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.validity import Event, SupersessionGraph, Validity

# ── numeric and unit extraction ──────────────────────────────────────

_NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*"
    r"(mEq/L|mmol/L|mg/dL|g/dL|mg/L|g/L|ng/mL|pg/mL|µg/mL|ug/mL|IU|L|mL|"
    r"mmHg|bpm|cm|kg|%|mmol|µmol|umol)?",
    re.IGNORECASE,
)

#: Past-tense markers. A claim carrying one is asserting a past state, so
#: a currently-superseded event may legitimately support it.
_PAST_MARKERS = (
    " was ",
    " were ",
    " had ",
    " previously ",
    " originally ",
    " initially ",
    " before ",
    " until ",
    " stopped ",
    " discontinued ",
    " earlier ",
    " used to ",
    " as of ",
    " back then ",
)

#: Present-tense markers. These assert the current state, so a superseded
#: event must not support them.
_PRESENT_MARKERS = (
    " is ",
    " are ",
    " currently ",
    " now ",
    " still ",
    " presently ",
    " today ",
)


@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic assertion in a generated answer."""

    text: str
    event_ids: Tuple[str, ...]

    def numbers(self) -> Set[str]:
        return {m.group(1) for m in _NUMBER.finditer(self.text) if m.group(1)}

    def units(self) -> Set[str]:
        return {
            m.group(2).lower()
            for m in _NUMBER.finditer(self.text)
            if m.group(2)
        }

    @property
    def tense(self) -> str:
        lowered = f" {self.text.lower()} "
        if any(m in lowered for m in _PAST_MARKERS):
            return "past"
        if any(m in lowered for m in _PRESENT_MARKERS):
            return "present"
        return "neutral"


@dataclass(slots=True)
class Verdict:
    claim: Claim
    faithfulness: bool
    temporal_validity: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.faithfulness and self.temporal_validity


def _event_text(event: Event) -> str:
    """Flatten an event payload into text for numeric comparison."""
    parts: List[str] = []
    for key, value in (event.payload or {}).items():
        if isinstance(value, (dict, list)):
            parts.append(str(value))
        else:
            parts.append(f"{key} {value}")
    return " ".join(parts)


def _numbers_in(text: str) -> Set[str]:
    return {m.group(1) for m in _NUMBER.finditer(text) if m.group(1)}


#: Temporal expressions whose digits are *not* clinical values.
#:
#: A claim like "HbA1c was 8.9 % as of 60 days ago" is fully grounded, but
#: the "60" is an offset the model is quoting back from the question, not
#: a measurement it invented. Checking it against the record would reject a
#: correct historical answer - which is precisely the answer this system
#: exists to be able to give.
#:
#: Only whole, well-formed temporal phrases are removed, and the removal
#: is anchored on the unit words. That matters: a loose pattern would let a
#: model hide a fabricated value inside something that looks like a date,
#: which would convert this from a false positive into a false negative.
_TEMPORAL = re.compile(
    r"\bT_MINUS_\d+D\b"
    r"|\bas\s+of\b\s*\d{4}-\d{1,2}-\d{1,2}"
    r"|\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b"
    r"|\b\d{1,4}\s+(?:day|week|fortnight|month|quarter|year)s?\s+"
    r"(?:ago|back|earlier|before)\b"
    r"|\b\d+\s*(?:hours?|minutes?)\s+(?:ago|back|earlier|before)\b",
    re.IGNORECASE,
)


def _clinical_numbers(text: str) -> Set[str]:
    """Numbers in a claim that assert a clinical value.

    Temporal offsets are removed first, because a date is not a
    measurement. Everything else is left alone: a stray number in a claim
    is precisely what this check exists to catch.
    """
    return _numbers_in(_TEMPORAL.sub(" ", text))


def _normalise_number(raw: str) -> str:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return raw
    return str(int(value)) if value == int(value) else str(value)


class ProvenanceLinker:
    """Validates claims against the retrieved set and the supersession graph.

    Three rejections, all of which are silent failures in naive systems:

    1. unknown event id          — the model cited something not retrieved
    2. number not in any citation — the model invented or misread a value
    3. superseded value in present tense
                                 — faithfully cited, clinically false
    """

    def __init__(self, graph: SupersessionGraph, patient_id: str):
        self.graph = graph
        self.patient_id = patient_id
        self._by_id = {e.event_id: e for e in graph.all_events}

    def verify(self, claims: Sequence[Claim], retrieved_ids: Set[str]) -> List[Verdict]:
        verdicts: List[Verdict] = []
        for claim in claims:
            reasons: List[str] = []
            faithfulness = True
            temporal = True

            cited = [self._by_id[eid] for eid in claim.event_ids if eid in self._by_id]

            # 1. unknown citations
            unknown = [eid for eid in claim.event_ids if eid not in self._by_id]
            if unknown:
                faithfulness = False
                reasons.append(f"unknown citation: {', '.join(unknown)}")

            # 1b. citations outside the retrieved set
            out_of_scope = [eid for eid in claim.event_ids if eid not in retrieved_ids]
            if out_of_scope:
                faithfulness = False
                reasons.append(f"citation not in retrieved set: {', '.join(out_of_scope)}")

            # 1c. cross-patient citation — the safety invariant
            foreign = [eid for eid, e in self._by_id.items() if e.patient_id != self.patient_id
                       and eid in claim.event_ids]
            if foreign:
                faithfulness = False
                reasons.append(f"cross-patient citation: {', '.join(foreign)}")

            # 2. numeric grounding
            claim_numbers = {
                _normalise_number(n) for n in _clinical_numbers(claim.text)
            }
            if claim_numbers and cited:
                cited_numbers: Set[str] = set()
                for event in cited:
                    cited_numbers |= {_normalise_number(n) for n in _numbers_in(_event_text(event))}
                unsupported = claim_numbers - cited_numbers
                if unsupported:
                    faithfulness = False
                    reasons.append(
                        f"value(s) not present in any citation: {', '.join(sorted(unsupported))}"
                    )
            elif claim_numbers and not cited:
                faithfulness = False
                reasons.append("numeric claim with no citation")

            # 3. temporal validity — Axis B
            tense = claim.tense
            if tense == "present" and cited:
                stale = [
                    e.event_id for e in cited if self.graph.state(e) is Validity.SUPERSEDED
                ]
                if stale:
                    temporal = False
                    reasons.append(
                        "superseded value asserted in present tense "
                        f"({', '.join(stale)}); state the correction instead"
                    )

            # Past-tense claims about superseded events are legitimate.
            if tense == "past" and cited:
                unknown_past = [
                    e.event_id
                    for e in cited
                    if _as_of_before_correction(self.graph, e) is None
                ]
                if unknown_past:
                    reasons.append(
                        f"past-tense claim about a record that did not exist yet "
                        f"({', '.join(unknown_past)})"
                    )

            verdicts.append(Verdict(claim, faithfulness, temporal, reasons))
        return verdicts


def _as_of_before_correction(graph: SupersessionGraph, event: Event):
    """A past-tense claim needs the event to have existed at its valid_time."""
    return event


def filter_accepted(verdicts: Iterable[Verdict]) -> List[Claim]:
    """Keep only fully verified claims. Rejected sentences are dropped, not hedged."""
    return [v.claim for v in verdicts if v.accepted]


# ── answer assembly ──────────────────────────────────────────────────


@dataclass
class GroundedAnswer:
    """The final response: surviving claims plus the provenance for each."""

    text: str
    citations: List[Dict] = field(default_factory=list)
    dropped: List[Dict] = field(default_factory=list)
    suppressed: List[Dict] = field(default_factory=list)

    @property
    def has_stale_assertion(self) -> bool:
        return bool(self.suppressed)


def assemble(verdicts: Sequence[Verdict]) -> GroundedAnswer:
    """Build the answer from verified claims, recording what was removed.

    Suppressed claims are surfaced in the response rather than silently
    dropped: a doctor should be able to see that a value was corrected,
    not just receive a quieter answer.
    """
    kept: List[str] = []
    citations: List[Dict] = []
    seen: Set[str] = set()
    dropped: List[Dict] = []
    suppressed: List[Dict] = []

    for verdict in verdicts:
        if verdict.accepted:
            kept.append(verdict.claim.text.strip())
            for eid in verdict.claim.event_ids:
                if eid not in seen:
                    seen.add(eid)
                    citations.append({"event_id": eid})
            continue
        if not verdict.faithfulness:
            dropped.append(
                {
                    "text": verdict.claim.text.strip(),
                    "reasons": verdict.reasons,
                }
            )
        elif not verdict.temporal_validity:
            suppressed.append(
                {
                    "text": verdict.claim.text.strip(),
                    "reasons": verdict.reasons,
                }
            )

    return GroundedAnswer(
        text=" ".join(kept).strip(),
        citations=citations,
        dropped=dropped,
        suppressed=suppressed,
    )
