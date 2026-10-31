"""Shared route dependencies."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.validity import Event as CoreEvent
from app.core.validity import SupersessionGraph
from app.db.identity import Doctor
from app.db.models import Event, Patient
from app.db.session import get_session  # noqa: F401  (re-exported for routes)
from app.services.pii import Deidentifier

# `current_doctor` lives in the auth route module because it is part of the
# auth surface. Imported here rather than duplicated: two implementations of
# "who is calling" is how they drift, and the drift is a security bug.
from app.api.routes.auth import current_doctor  # noqa: E402


async def get_patient(
    patient_id: UUID,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
) -> Patient:
    """Resolve a patient id, enforcing that it belongs to the caller.

    This used to be a bare `session.get(Patient, patient_id)`, which meant
    any authenticated doctor could name any patient id and read that
    practice's entire record - timeline, events, documents, and a chat
    answer quoting the values. It was also a write path, not just a read.

    The distinction that matters: a patient CAN legitimately belong to
    several doctors, which is what email linking means. But each of those
    doctors gets their own `Patient` row (see `auth.py` linking), so
    `patient.doctor_id` is exactly the right test. The person themself is
    one identity across all rows, and reaches them through the
    `current_patient` / `/patient/*` surface, not through here.

    404 rather than 403 for someone else's row: a 403 confirms the id
    exists, which turns this endpoint into an enumeration oracle.
    """
    patient = await session.get(Patient, patient_id)
    if patient is None or patient.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="patient not found")
    return patient


async def load_core_events(
    patient_id: UUID, session: AsyncSession
) -> List[CoreEvent]:
    """Load a patient's full event log and convert to the pure core type.

    The whole log is loaded rather than a window. Validity is derived from
    the graph, so a partial load could resolve a correction chain
    incorrectly — correctness over convenience.
    """
    rows = (
        await session.execute(
            select(Event).where(Event.patient_id == patient_id).order_by(Event.valid_time)
        )
    ).scalars().all()
    return [row.to_core() for row in rows]


async def build_graph(patient_id: UUID, session: AsyncSession) -> SupersessionGraph:
    return SupersessionGraph(await load_core_events(patient_id, session))


def make_deidentifier(patient: Patient) -> Deidentifier:
    return Deidentifier(
        patient_id=str(patient.id),
        patient_name=patient.display_name,
        age=patient.age,
        phone=patient.phone,
        email=patient.email,
        address=patient.address,
    )


PatientDep = Depends(get_patient)
