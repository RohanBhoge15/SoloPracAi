"""Timeline routes — the three-valued view a physician actually reads."""

from __future__ import annotations

from datetime import datetime
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_patient
from app.api.routes.events import _enrich
from app.api.schemas import AsOfOut, EventOut, SupersededPair, TimelineOut
from app.core.validity import Validity
from app.db.models import Event, Patient
from app.db.session import get_session
from app.services.llm import RedisCache

router = APIRouter(prefix="/patients/{patient_id}", tags=["timeline"])


@router.get("/timeline", response_model=TimelineOut)
async def timeline(
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """The primary surface: current records, full history, correction pairs."""
    rows = (
        await session.execute(
            select(Event).where(Event.patient_id == patient.id).order_by(Event.valid_time)
        )
    ).scalars().all()
    by_id = {r.id: r for r in rows}
    graph = await _graph(patient, session)
    enriched = [_enrich(r, rows) for r in rows]

    current = [e for e in enriched if e["validity"] == "current"]
    current.sort(key=lambda e: e["valid_time"], reverse=True)
    history = sorted(enriched, key=lambda e: e["valid_time"], reverse=True)

    pairs: List[SupersededPair] = []
    for old, correction in graph.superseded_pairs(str(patient.id)):
        old_row, new_row = by_id.get(_uuid_of(old)), by_id.get(_uuid_of(correction))
        if old_row is not None and new_row is not None:
            pairs.append(
                SupersededPair(
                    superseded=EventOut(**_enrich(old_row, rows)),
                    correction=EventOut(**_enrich(new_row, rows)),
                )
            )

    return TimelineOut(
        patient_id=patient.id,
        current=[EventOut(**e) for e in current],
        history=[EventOut(**e) for e in history],
        pairs=pairs,
        stats=graph.stats(str(patient.id)),
    )


@router.get("/as-of/{instant}", response_model=AsOfOut)
async def as_of(
    instant: datetime,
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """What was true at a given instant.

    Resolves supersession relative to the requested moment, not to now.
    This is the endpoint that a recency-decay baseline cannot replicate.
    """
    rows = (
        await session.execute(
            select(Event).where(Event.patient_id == patient.id).order_by(Event.valid_time)
        )
    ).scalars().all()
    graph = await _graph(patient, session)

    wanted = {e.event_id for e in graph.as_of_view(str(patient.id), instant)}
    events = [
        EventOut(**_enrich(r, rows))
        for r in rows
        if str(r.id) in wanted
    ]
    events.sort(key=lambda e: e.valid_time, reverse=True)
    return AsOfOut(patient_id=patient.id, instant=instant, events=events)


@router.get("/stats", tags=["timeline"])
async def stats(
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    graph = await _graph(patient, session)
    return {"patient_id": patient.id, **graph.stats(str(patient.id))}


async def _graph(patient: Patient, session: AsyncSession):
    from app.api.deps import build_graph

    return await build_graph(patient.id, session)


def _uuid_of(core_event) -> UUID:
    return UUID(core_event.event_id)
