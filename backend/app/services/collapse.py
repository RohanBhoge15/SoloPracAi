"""Collapse repeated entities so an answer lists a drug once.

A longitudinal record contains three Metformin rows because the dose was
changed twice. A physician asking "what is the patient on" wants one
line per drug with the current dose, plus a note where the record shows
a change -  not three contradictory-looking rows.

This is presentation logic, kept separate from retrieval: what was found
is unchanged, only how it reads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.core.validity import Event, EventType, SourceType, Validity


def _rank_key(event: Event) -> Tuple[float, float]:
    """Ordering within one entity; highest sorts first.

    A medication's *status* outranks its timestamp: a newer "stopped" row
    must beat an older "active" one, or the answer tells a physician the
    patient is still taking a drug that was discontinued.

    Implemented as a penalty rather than a boolean so that a drug with no
    active record at all still surfaces - just last.
    """
    penalty = 0.0
    if event.event_type == EventType.MEDICATION:
        status = str((event.payload or {}).get("status", "active")).lower()
        if status not in ("active", "current", "ongoing", ""):
            penalty = 1.0
    return (event.valid_time.timestamp() - penalty, event.valid_time.timestamp())


def _identity(event: Event) -> Optional[str]:
    """The thing a physician thinks of as 'the same record'."""
    payload = event.payload or {}
    if event.event_type == EventType.MEDICATION:
        drug = payload.get("drug")
        return f"med:{str(drug).lower().strip()}" if drug else None
    if event.event_type == EventType.LAB:
        test = payload.get("test")
        return f"lab:{str(test).lower().strip()}" if test else None
    if event.event_type == EventType.DIAGNOSIS:
        # Per disease, not one bucket. The old constant "diagnosis:active"
        # merged every disease into a single row, so recording Yellow Fever
        # would have hidden Type 2 Diabetes Mellitus behind "newest wins".
        diagnosis = str(payload.get("diagnosis") or "").lower().strip()
        return f"diagnosis:{diagnosis}" if diagnosis else None
    if event.event_type == EventType.VITAL:
        measurement = payload.get("measurement")
        return f"vital:{str(measurement).lower().strip()}" if measurement else None
    if event.event_type == EventType.NOTE:
        # Identical note text filed repeatedly (a standing note re-saved at
        # each visit) is one row, newest first. Different text stays
        # separate - two different notes are two facts, not two versions.
        note = str(payload.get("note") or "").lower().strip()
        return f"note:{note}" if note else None
    if event.event_type == EventType.IMAGING:
        study = payload.get("study")
        return f"imaging:{str(study).lower().strip()}" if study else None
    return None


def collapse(
    items: List[Tuple[Any, Validity]]
) -> Tuple[List[Tuple[Any, Validity]], List[str]]:
    """Reduce (event, validity) pairs to one row per entity.

    Returns (kept, notes). The kept row is the most recent surviving
    version; notes describe anything that was superseded along the way.
    """
    grouped: Dict[str, List[Tuple[Any, Validity]]] = {}
    order: List[str] = []
    ungrouped: List[Tuple[Any, Validity]] = []

    for event, validity in items:
        key = _identity(event)
        if key is None:
            ungrouped.append((event, validity))
            continue
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((event, validity))

    kept: List[Tuple[Any, Validity]] = []
    notes: List[str] = []

    for key in order:
        # Newest active version wins. A discontinued drug sorts last and is
        # only chosen when no active record exists for it at all.
        rows = sorted(grouped[key], key=lambda pair: _rank_key(pair[0]), reverse=True)
        current = [r for r in rows if r[1] is Validity.CURRENT]
        pool = current or rows
        kept.append(pool[0])

        # Only note a change when the record actually shows a prior version.
        if current and len(rows) > 1:
            kept_event = pool[0][0]
            others = [r for r in rows if r[0].event_id != kept_event.event_id]
            dates = sorted(
                {r[0].valid_time.date().isoformat() for r in others}, reverse=True
            )
            notes.append(
                f"{_label(kept_event)} - record shows {len(rows)} versions; "
                f"most recent change {dates[0]}"
            )

    kept.extend(ungrouped)
    kept.sort(key=lambda pair: pair[0].valid_time, reverse=True)
    return kept, notes


def _label(event: Event) -> str:
    payload = event.payload or {}
    return str(
        payload.get("drug")
        or payload.get("test")
        or payload.get("measurement")
        or event.event_type
    )
