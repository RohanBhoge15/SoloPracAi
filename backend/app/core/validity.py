"""Three-valued validity - the research core of SoloPrac.

This module is deliberately pure: no database, no network, no framework.
Everything about the paper's central claim (validity is derived from the
supersession graph, never stored) is decided here, so it can be tested in
isolation and reasoned about without infrastructure.

Three states:
    CURRENT     no active correction points at this event
    SUPERSEDED  a correction points at this event
    AS_OF_T     valid at an explicitly requested instant

The critical asymmetry: supersession is resolved *relative to an instant*.
A record corrected in March is CURRENT as of February and SUPERSEDED now.
This is what preserves historical recall, and it is the property that
recency-decay baselines cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set


class Validity(str, Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class EventType(str, Enum):
    LAB = "lab"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    VITAL = "vital"
    NOTE = "note"
    IMAGING = "imaging"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class SourceType(str, Enum):
    MANUAL = "manual"
    OCR = "ocr"
    AI_SUGGESTION = "ai_suggestion"
    CORRECTION = "correction"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def _as_aware(value: datetime) -> datetime:
    """Normalise to timezone-aware UTC so comparisons never raise."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable clinical fact. Never mutated, never deleted."""

    event_id: str
    patient_id: str
    event_type: EventType
    valid_time: datetime
    payload: Dict
    source_type: SourceType = SourceType.MANUAL
    confidence: Optional[float] = None
    supersedes_id: Optional[str] = None
    recorded_at: Optional[datetime] = None
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source_type is SourceType.CORRECTION and not self.supersedes_id:
            raise ValueError(
                f"correction {self.event_id} must declare supersedes_id; "
                "an uncorrected correction is not a correction"
            )
        if self.source_type is not SourceType.CORRECTION and self.supersedes_id:
            raise ValueError(
                f"non-correction {self.event_id} cannot declare supersedes_id; "
                "only corrections supersede"
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence {self.confidence} for {self.event_id} outside [0,1]"
            )

    @property
    def is_correction(self) -> bool:
        return self.source_type is SourceType.CORRECTION


def content_hash(event: Event) -> str:
    """Deterministic tamper-evidence hash.

    Uses sorted-key JSON so the digest is stable across interpreter runs —
    a hash that varies between runs proves nothing.
    """
    import hashlib
    import json

    canonical = json.dumps(
        {
            "event_type": str(event.event_type),
            "payload": event.payload,
            "source_type": str(event.source_type),
            "supersedes_id": event.supersedes_id,
            "valid_time": _as_aware(event.valid_time).isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SupersessionGraph:
    """Derives three-valued validity from an append-only event log.

    The graph is a DAG: corrections point backwards at what they correct.
    Chains are supported (a correction may itself be corrected) and validity
    is transitive through them.
    """

    def __init__(self, events: Iterable[Event]):
        by_id: Dict[str, Event] = {}
        for event in events:
            if event.event_id in by_id:
                raise ValueError(f"duplicate event_id {event.event_id}")
            by_id[event.event_id] = event
        self._by_id = by_id

        # children[c] = corrections that supersede c
        children: Dict[str, List[Event]] = {}
        for event in by_id.values():
            if event.supersedes_id:
                parent = by_id.get(event.supersedes_id)
                if parent is None:
                    raise ValueError(
                        f"event {event.event_id} supersedes unknown "
                        f"{event.supersedes_id}"
                    )
                if parent.patient_id != event.patient_id:
                    raise ValueError(
                        f"cross-patient supersession: {event.event_id} "
                        f"({event.patient_id}) supersedes {parent.event_id} "
                        f"({parent.patient_id})"
                    )
                children.setdefault(event.supersedes_id, []).append(event)
        self._children = children

    # ── core queries ──────────────────────────────────────────────────

    def corrections_to(self, event_id: str) -> List[Event]:
        return list(self._children.get(event_id, ()))

    def state(self, event: Event) -> Validity:
        """Current state of an event (no temporal argument)."""
        return Validity.SUPERSEDED if self._children.get(event.event_id) else Validity.CURRENT

    def state_as_of(self, event: Event, instant: datetime) -> Validity:
        """State of an event *as of* an instant.

        An event is SUPERSEDED at `instant` only if a correction whose own
        valid_time is at or before `instant` points at it. A correction
        recorded later does not retroactively invalidate what was believed
        at the time — this is exactly the medico-legal property the previous
        prototype never specified.
        """
        if not self._children.get(event.event_id):
            return Validity.CURRENT
        at = _as_aware(instant)
        for correction in self._children[event.event_id]:
            if _as_aware(correction.valid_time) <= at:
                return Validity.SUPERSEDED
        return Validity.CURRENT

    def effective_version(self, event: Event, instant: datetime) -> Optional[Event]:
        """Walk the correction chain forward to the version valid at `instant`."""
        at = _as_aware(instant)
        current = event
        seen: Set[str] = set()
        while current.event_id not in seen:
            seen.add(current.event_id)
            if _as_aware(current.valid_time) > at:
                return None
            successor = None
            for correction in self._children.get(current.event_id, ()):
                if _as_aware(correction.valid_time) <= at:
                    if successor is None or _as_aware(
                        correction.valid_time
                    ) > _as_aware(successor.valid_time):
                        successor = correction
            if successor is None:
                return current
            current = successor
        return None

    # ── projections ───────────────────────────────────────────────────

    def current_view(self, patient_id: str) -> List[Event]:
        """Events that are true now, for one patient."""
        return sorted(
            (
                e
                for e in self._by_id.values()
                if e.patient_id == patient_id and self.state(e) is Validity.CURRENT
            ),
            key=lambda e: _as_aware(e.valid_time),
            reverse=True,
        )

    def as_of_view(self, patient_id: str, instant: datetime) -> List[Event]:
        """Events that were true at `instant`, for one patient.

        Resolves supersession *relative to the requested instant*, not to now.
        This single method is what a recency-decay baseline cannot replicate.
        """
        at = _as_aware(instant)
        return sorted(
            (
                e
                for e in self._by_id.values()
                if e.patient_id == patient_id
                and _as_aware(e.valid_time) <= at
                and self.state_as_of(e, at) is Validity.CURRENT
            ),
            key=lambda e: _as_aware(e.valid_time),
            reverse=True,
        )

    def history(self, patient_id: str) -> List[Event]:
        """Every event for a patient, both states, chronological."""
        return sorted(
            (e for e in self._by_id.values() if e.patient_id == patient_id),
            key=lambda e: _as_aware(e.valid_time),
        )

    def superseded_pairs(self, patient_id: str) -> List[tuple[Event, Event]]:
        """(superseded, correction) pairs — drives the timeline UI."""
        pairs = []
        for event in self._by_id.values():
            if event.patient_id != patient_id:
                continue
            for correction in self._children.get(event.event_id, ()):
                pairs.append((event, correction))
        pairs.sort(key=lambda p: _as_aware(p[0].valid_time), reverse=True)
        return pairs

    def chain(self, event_id: str) -> List[Event]:
        """Full supersession chain for one event, oldest first."""
        event = self._by_id[event_id]
        out = [event]
        seen = {event_id}
        while True:
            nxt = [c for c in self._children.get(out[-1].event_id, ())]
            if not nxt:
                return out
            nxt.sort(key=lambda e: _as_aware(e.valid_time))
            chosen = nxt[0]
            if chosen.event_id in seen:
                return out
            seen.add(chosen.event_id)
            out.append(chosen)

    # ── stats ─────────────────────────────────────────────────────────

    @property
    def all_events(self) -> List[Event]:
        return list(self._by_id.values())

    def stats(self, patient_id: str) -> Dict[str, int]:
        events = [e for e in self._by_id.values() if e.patient_id == patient_id]
        return {
            "total": len(events),
            "current": sum(1 for e in events if self.state(e) is Validity.CURRENT),
            "superseded": sum(
                1 for e in events if self.state(e) is Validity.SUPERSEDED
            ),
            "corrections": sum(1 for e in events if e.is_correction),
        }
