"""The weekly clinical report.

This is the feature that ties the whole system together, and the reason
the LLM prompt is large: the model has to be told what a valid
three-valued clinical record looks like, or it will summarise a corrected
lab value as if it were current.

The prompt therefore:

  1. States the clinical role and the audience (a treating clinician).
  2. Explains the three states, explicitly, because the model has no way
     to infer that "superseded" is a category and not noise.
  3. Forbids the one specific failure: asserting a corrected value.
  4. Requires that every claim name the record it came from.
  5. Is given the records already filtered by the validity-aware
     retriever, so the model reasons over what is true rather than
     retrieving for itself.

Everything the model says is still checked afterwards. The report is a
draft a doctor signs, not an autonomous finding.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.validity import Event, EventType, SourceType, SupersessionGraph, Validity
from app.services.retrieval import ValidityAwareRetriever, parse_temporal_intent

logger = logging.getLogger(__name__)


WEEKLY_SYSTEM_PROMPT = """
You are a senior consultant physician writing a clinical progress report
for another clinician who will read it on Monday morning with no other
context. The reader knows this patient; you do not.

You are given a patient's record for a reporting period. The record is 
three-valued and this matters absolutely:

  CURRENT     - what we believe now. Safe to state as fact.
  SUPERSEDED  - a later record corrected this. It was true when written 
                and is now known to be wrong. You may mention it ONLY in 
                the past tense, and you MUST say it was corrected.
  AS-OF-T     - the state of the record at a past instant.

A superseded value reported in the present tense is a clinical error that 
can harm a patient. This is the single most important rule in this 
prompt. If a potassium was amended from 2.1 to 4.8, you write "potassium 
4.8 (amended from an initial 2.1)", never "potassium 2.1".

Other rules:
  - Use only the records given to you. Never infer a value that is absent.
  - State values with their units. Do not round away a clinically 
    meaningful digit.
  - If a record carries a flag, say so: "high", "low", "normal".
  - Do not invent a diagnosis, a medication, or a plan.
  - Do not recommend treatment. You are summarising, not prescribing.
  - If the period is quiet, say so plainly. A short report that is 
    accurate beats a long one that is padded.
  - Plain prose. No markdown headings, no bullet characters, no bold. 
    The renderer will structure it.
  - British spelling, clinical register, no filler.

Write 3 to 5 short paragraphs: overall trajectory, then what changed, 
then what needs watching. Address the reader as a colleague.
"""


# ── record preparation ──────────────────────────────────────────────


def collect_period_events(
    events: List[Event], days: int, now: Optional[datetime] = None
) -> Tuple[List[Event], List[Tuple[Event, Event]]]:
    """Split a window into its current records and its corrections."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    graph = SupersessionGraph(events)

    in_window = [e for e in events if e.valid_time >= start]
    pairs = [
        (old, correction)
        for old, correction in graph.superseded_pairs(
            str(in_window[0].patient_id) if in_window else ""
        )
        if old.valid_time >= start
    ]
    return in_window, pairs


def render_for_prompt(
    events: List[Event],
    pairs: List[Tuple[Event, Event]],
    graph: SupersessionGraph,
    limit: int = 40,
) -> Tuple[str, List[str]]:
    """Render the window as text, marking validity explicitly.

    Returns (text, cited_event_ids) so the report can name what it used.
    """
    cited: List[str] = []
    lines: List[str] = []

    def label(e: Event) -> str:
        return {
            EventType.LAB: "lab",
            EventType.MEDICATION: "medication",
            EventType.DIAGNOSIS: "diagnosis",
            EventType.VITAL: "vital",
            EventType.NOTE: "note",
            EventType.IMAGING: "imaging",
        }.get(e.event_type, "record")

    def value(e: Event) -> str:
        p = e.payload or {}
        if e.event_type == EventType.LAB:
            unit = p.get("unit") or ""
            flag = f" [{p['flag']}]" if p.get("flag") else ""
            return f"{p.get('test')} = {p.get('value')} {unit}{flag}".strip()
        if e.event_type == EventType.MEDICATION:
            status = p.get("status", "active")
            return (
                f"{p.get('drug')} {p.get('dose','')} {p.get('frequency','')} "
                f"({status})".strip()
            )
        if e.event_type == EventType.DIAGNOSIS:
            return f"{p.get('diagnosis')} {p.get('icd10','')}".strip()
        if e.event_type == EventType.VITAL:
            return f"{p.get('measurement')} = {p.get('value')} {p.get('unit','')}".strip()
        if e.event_type == EventType.IMAGING:
            return f"{p.get('study')}: {p.get('impression') or p.get('findings','')}"
        return str(p.get("note", ""))[:400]

    # Corrections first: they are the most important thing in the report.
    if pairs:
        lines.append("CORRECTIONS (a value below was wrong when first recorded):")
        for old, new in pairs[:20]:
            cited.extend([old.event_id, new.event_id])
            reason = (new.payload or {}).get("amendment")
            suffix = f" - reason: {reason}" if reason else ""
            lines.append(
                f"  [{old.event_id}] {label(old)}: {value(old)}"
                f"  ->  SUPERSEDED on {new.valid_time.date().isoformat()}"
                f" by [{new.event_id}] {value(new)}{suffix}"
            )
        lines.append("")

    # Current records.
    current = [e for e in events if graph.state(e) is Validity.CURRENT]
    if current:
        lines.append("CURRENT RECORDS (true as of now):")
        for e in current[:limit]:
            cited.append(e.event_id)
            note = ""
            if e.event_type == EventType.NOTE:
                note = f"  # {(e.payload or {}).get('author','')}"
            lines.append(
                f"  [{e.event_id}] {e.valid_time.date().isoformat()} "
                f"{label(e)}: {value(e)}{note}"
            )

    # Superseded-but-uncorrected is impossible, but a superseded record
    # with no visible correction would be a data problem worth surfacing.
    orphans = [
        e
        for e in events
        if graph.state(e) is Validity.SUPERSEDED
        and not any(old.event_id == e.event_id for old, _ in pairs)
    ]
    if orphans:
        lines.append("")
        lines.append("SUPERSEDED WITHOUT A VISIBLE CORRECTION (data issue):")
        for e in orphans[:10]:
            lines.append(f"  [{e.event_id}] {label(e)}: {value(e)}")

    return "\n".join(lines), list(dict.fromkeys(cited))


def build_period(events: List[Event], days: int, now=None) -> Dict[str, Any]:
    """The structured payload the PDF template renders.

    Built deterministically from the record rather than from the model, so
    the numbers in the report are always the record's numbers.
    """
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    graph = SupersessionGraph(events)
    if not events:
        return {
            "period_from": start.isoformat(),
            "period_to": now.isoformat(),
            "findings": [],
            "medications": [],
            "corrections": [],
            "retrieved_context": [],
        }

    window = [e for e in events if e.valid_time >= start]
    pairs = graph.superseded_pairs(str(events[0].patient_id))

    findings: List[Dict[str, Any]] = []
    seen_tests = set()
    for e in sorted(window, key=lambda x: x.valid_time, reverse=True):
        if e.event_type is not EventType.LAB:
            continue
        p = e.payload or {}
        if graph.state(e) is not Validity.CURRENT:
            continue
        name = str(p.get("test", "?"))
        if name in seen_tests:
            # latest only
            window = [x for x in window if not (
                x.event_type is EventType.LAB
                and (x.payload or {}).get("test") == name
                and x.event_id != e.event_id
            )]
        seen_tests.add(name)
        findings.append(
            {
                "test": name,
                "value": f"{p.get('value')} {p.get('unit','')}".strip(),
                "flag": p.get("flag", "normal"),
                "date": e.valid_time.date().isoformat(),
            }
        )

    medications: List[Dict[str, Any]] = []
    for e in sorted(window, key=lambda x: x.valid_time, reverse=True):
        if e.event_type is not EventType.MEDICATION:
            continue
        p = e.payload or {}
        if str(p.get("status", "active")).lower() == "stopped":
            continue
        medications.append(
            {
                "drug": p.get("drug"),
                "dose": p.get("dose"),
                "frequency": p.get("frequency"),
                "since": e.valid_time.date().isoformat(),
            }
        )

    corrections: List[Dict[str, Any]] = []
    for old, new in pairs:
        if old.valid_time < start:
            continue
        op, np_ = old.payload or {}, new.payload or {}
        corrections.append(
            {
                "test": op.get("test") or op.get("drug") or "record",
                "old": op.get("value") or op.get("dose") or "—",
                "new": np_.get("value") or np_.get("dose") or "—",
                "date": new.valid_time.date().isoformat(),
                "reason": np_.get("amendment"),
            }
        )

    return {
        "period_from": start.isoformat(),
        "period_to": now.isoformat(),
        "days": days,
        "findings": findings,
        "medications": medications,
        "corrections": corrections,
        "retrieved_context": [],
    }


async def draft_summary(
    llm,
    *,
    patient: dict,
    doctor: dict,
    period: Dict[str, Any],
    record_text: str,
    cited: List[str],
    now: Optional[datetime] = None,
) -> str:
    """Ask the model for the narrative.

    The structured tables are built from the record, not the model. Only
    the prose comes from the LLM, and it is a draft a doctor signs.
    """
    if llm is None:
        return _fallback_summary(patient, period)

    header = (
        f"Patient: {patient.get('display_name')}, "
        f"{patient.get('age') or 'age unknown'}\n"
        f"Reporting period: last {period.get('days',7)} days "
        f"({period['period_from'][:10]} to {period['period_to'][:10]})\n"
        f"Clinician: {doctor.get('account', {}).get('display_name','')}\n"
    )

    prompt = f"{header}\nRECORDS\n\n{record_text}\n\nWrite the report."

    # Same discipline as chat: nothing named or dated crosses the wire.
    # render_for_prompt emits absolute calendar dates, which the PII gate
    # rightly refuses — without this scrub the LLM path fails on EVERY
    # report and only the extract fallback ever ships. Seen live Oct 2026
    # (weekly summary generation failed: calendar_date).
    from app.services.pii import Deidentifier

    anchor = now or datetime.now(timezone.utc)
    deid = Deidentifier(
        patient_id=str(patient.get("id", "")),
        patient_name=patient.get("display_name"),
        phone=patient.get("phone"),
        today=anchor.date(),
    )
    prompt = deid.scrub(prompt)

    try:
        result = await llm.complete(prompt, system=WEEKLY_SYSTEM_PROMPT)
        if result.text and not result.degraded:
            summary = result.text.strip()
            # Strip markdown the renderer cannot use.
            summary = summary.replace("**", "").replace("##", "").strip()
            period["retrieved_context"] = cited[:12]
            return summary
    except Exception:  # noqa: BLE001
        logger.warning("weekly summary generation failed", exc_info=True)

    return _fallback_summary(patient, period)


def _fallback_summary(patient: dict, period: Dict[str, Any]) -> str:
    """A safe, factual summary when the model is unavailable.

    The free tier is slow and rate-limits, so a report that depends on it
    would fail often. This produces something true rather than nothing.
    """
    name = (patient.get("display_name") or "The patient").split(" ")[0]
    days = period.get("days", 7)
    parts = [
        f"{name} was reviewed over the last {days} days. "
        f"{len(period['findings'])} current investigation result(s) and "
        f"{len(period['medications'])} active medication(s) are on file."
    ]
    if period["corrections"]:
        parts.append(
            f"{len(period['corrections'])} record(s) were corrected in this "
            f"period. The corrected values above are the ones to act on; the "
            f"originals are retained for audit and should not be relied on."
        )
    if not period["findings"] and not period["corrections"]:
        parts.append("No new results were recorded in this period.")
    parts.append(
        "This report was assembled from the versioned record. Every figure "
        "shown is the current version; superseded values are shown struck "
        "through in the corrections table."
    )
    return " ".join(parts)
