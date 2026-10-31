"""Patient routes."""

from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_patient
from app.api.routes.auth import current_doctor
from app.api.schemas import PatientCreate, PatientOut
from app.db.identity import Doctor
from app.db.models import Patient
from app.db.session import get_session

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("", response_model=List[PatientOut])
async def list_patients(
    doctor: Doctor = Depends(current_doctor),
    search: str | None = Query(default=None, description="name substring"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """This doctor's patients, and only theirs.

    Scoping by `doctor_id` is the whole point. Before this was enforced
    the query selected every patient row in the database, so any caller
    could read the full patient list of every practice on the server.
    """
    stmt = select(Patient).where(Patient.doctor_id == doctor.id)
    if search:
        stmt = stmt.where(Patient.display_name.ilike(f"%{search}%"))
    stmt = stmt.order_by(Patient.created_at.desc()).limit(limit).offset(offset)
    return (await session.execute(stmt)).scalars().all()


@router.post("", response_model=PatientOut, status_code=status.HTTP_201_CREATED)
async def create_patient(
    body: PatientCreate,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    """Register a patient against the signed-in doctor.

    The token is required. It used to be optional, and an absent or
    invalid token quietly produced an *unowned* patient row - a record
    belonging to no doctor, visible to no one, that still consumed a row
    and still accepted a real patient's name, age, phone and email from
    anyone who could reach the port. An unowned row is not a neutral
    outcome; it is a silent write of clinical data with no owner and no
    audit trail.
    """
    from app.core.security import normalise_email

    values = body.model_dump(exclude_none=True)
    # Normalise on write: `A@x.com` and `a@x.com` must be one person or
    # the auto-linking guarantee fails silently.
    if values.get("email"):
        values["email"] = normalise_email(values["email"]) or values["email"]

    patient = Patient(
        **values,
        doctor_id=doctor.id,
        origin="doctor_created",
    )
    session.add(patient)
    await session.flush()
    await session.refresh(patient)
    return patient


@router.get("/{patient_id}/photo")
async def patient_photo(
    patient_id: UUID,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    """A presigned URL for a patient's picture, for their own doctor.

    Same reasoning as `/documents/photo/url`: an `<img>` cannot send an
    Authorization header, so the browser cannot point the tag at an
    authenticated endpoint. The client fetches the URL here and gives the
    `<img>` the object-storage one.

    Scoped with the same `doctor_id` check as the rest of this router. A
    patient's picture is their identity document, not decoration, so it
    must not be readable by any signed-in user who can guess a uuid.
    """
    patient = await session.get(Patient, patient_id)
    if patient is None or patient.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="patient not found")
    if not patient.photo_path:
        raise HTTPException(status_code=404, detail="no profile picture")

    from app.core.config import get_settings
    from app.services.storage import get_storage

    url = await get_storage().presigned_url(
        patient.photo_path, ttl=get_settings().avatar_url_ttl_seconds
    )
    return {"url": url}


@router.get("/{patient_id}", response_model=PatientOut)
async def get_one(patient: Patient = Depends(get_patient)):
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: UUID,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    """Delete one of this doctor's own patients.

    Authenticated *and* ownership-checked. An unauthenticated delete on a
    records system is destructive and irreversible, which is strictly
    worse than the unauthenticated create this was found alongside.
    """
    patient = await session.get(Patient, patient_id)
    if patient is None or patient.doctor_id != doctor.id:
        # 404 rather than 403: another practice's patient id must not be
        # confirmed to exist, or the endpoint becomes an enumeration oracle.
        raise HTTPException(status_code=404, detail="patient not found")
    await session.delete(patient)
    return None
