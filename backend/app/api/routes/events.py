"""Event routes — the append-only ledger.

There is no PUT and no DELETE on events. A mistake is corrected by
appending a correction, which is what makes the record time-travellable.
"""

from __future__ import annotations

import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_patient, load_core_events
from app.api.schemas import EventCreate, EventOut
from app.core.validity import Event as CoreEvent
from app.core.validity import content_hash
from app.core.validity import EventType, SourceType
from app.db.models import Event, Patient
from app.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patients/{patient_id}/events", tags=["events"])


def _to_core(body: EventCreate, patient_id: UUID) -> CoreEvent:
    return CoreEvent(
        event_id="",
        patient_id=str(patient_id),
        event_type=EventType(body.event_type),
        valid_time=body.valid_time,
        payload=body.payload,
        source_type=SourceType(body.source_type),
        confidence=body.confidence,
        supersedes_id=str(body.supersedes_id) if body.supersedes_id else None,
    )


@router.get("", response_model=List[EventOut])
async def list_events(
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    rows = (
        await session.execute(
            select(Event)
            .where(Event.patient_id == patient.id)
            .order_by(Event.valid_time.desc())
        )
    ).scalars().all()
    return [_enrich(row, rows) for row in rows]


@router.post("", response_model=EventOut, status_code=status.HTTP_201_CREATED)
async def create_event(
    body: EventCreate,
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    # Validate through the core type so the invariants are enforced in one
    # place rather than duplicated here.
    try:
        core = _to_core(body, patient.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if core.supersedes_id:
        target_id = UUID(core.supersedes_id)
        target = await session.get(Event, target_id)
        if target is None:
            raise HTTPException(status_code=404, detail="superseded event not found")
        if target.patient_id != patient.id:
            raise HTTPException(
                status_code=403, detail="cannot supersede another patient's event"
            )
        # A record can be superseded exactly once. To revise a correction,
        # supersede the correction itself - otherwise two corrections would
        # claim the same predecessor and the chain becomes ambiguous.
        already = (
            await session.execute(
                select(Event.id).where(
                    Event.supersedes_id == target_id,
                    Event.source_type == "correction",
                )
            )
        ).first()
        if already is not None:
            raise HTTPException(
                status_code=409,
                detail="record is already superseded; correct the correction instead",
            )

    row = Event.from_core(core)
    row.content_hash = content_hash(core)
    session.add(row)
    await session.flush()
    await session.refresh(row)

    # Index immediately. A record absent from the vector store cannot be
    # found by paraphrase, so a doctor who files a glucose result and then
    # asks "what is her blood sugar" would get the weaker token-overlap
    # path for the one record they most expect to be found. One forward
    # pass is tens of milliseconds on CPU, which is cheaper than the
    # confusion it prevents.
    await _index_now(session, patient.id, [row])
    return _enrich(row, [row])


@router.post("/bulk", response_model=List[EventOut], status_code=status.HTTP_201_CREATED)
async def create_events_bulk(
    body: List[EventCreate],
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    created: List[Event] = []
    for item in body:
        try:
            core = _to_core(item, patient.id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        row = Event.from_core(core)
        row.content_hash = content_hash(core)
        session.add(row)
        created.append(row)
    await session.flush()
    # Batched: one encode batch rather than N, and the superseded states
    # are recomputed from the whole log so a new correction relabels its
    # predecessor in the index too.
    await _index_now(session, patient.id, created)
    return [_enrich(row, created) for row in created]


async def _index_now(
    session: AsyncSession, patient_id: UUID, rows: List[Event]
) -> None:
    """Write new records to the vector index, best effort.

    Never raises. The index is a cache, so failing to update it must not
    fail the write that preceded it — a doctor filing a result cannot be
    blocked because Qdrant is briefly unreachable. The event log remains
    authoritative and `POST /semantic/reindex` rebuilds the index.
    """
    import asyncio

    from app.core.validity import SupersessionGraph
    from app.services.hybrid import index_events

    try:
        # The full log, because a correction's validity label depends on
        # what supersedes it, and the predecessor's label changes too.
        events = await load_core_events(patient_id, session)
        graph = SupersessionGraph(events)
        await asyncio.to_thread(index_events, events, graph)
    except Exception:  # noqa: BLE001
        logger.warning(
            "vector index not updated for patient %s; "
            "the record is saved and the index can be rebuilt with "
            "POST /api/v1/semantic/reindex",
            patient_id,
            exc_info=True,
        )


def _enrich(row: Event, all_rows: List[Event]) -> dict:
    """Attach derived validity. Never stored — computed on read."""
    superseded_ids = {
        r.supersedes_id
        for r in all_rows
        if r.supersedes_id is not None and r.source_type == "correction"
    }
    is_superseded = row.id in superseded_ids
    return {
        "id": row.id,
        "event_type": row.event_type,
        "source_type": row.source_type,
        "valid_time": row.valid_time,
        "recorded_at": row.recorded_at,
        "payload": row.payload or {},
        "confidence": row.confidence,
        "supersedes_id": row.supersedes_id,
        "content_hash": row.content_hash,
        "is_superseded": is_superseded,
        "validity": "superseded" if is_superseded else "current",
    }
