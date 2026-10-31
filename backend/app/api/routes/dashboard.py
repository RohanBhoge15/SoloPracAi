"""The doctor's home screen.

Two things live here: counts the doctor actually looks at, and a bounded
list of who they have seen. Both are scoped to the signed-in doctor
rather than the whole table, so two practices on one server cannot see
each other's numbers.

The counts are deliberately simple COUNT queries over indexed columns
rather than anything clever. A solo practice holds hundreds of patients,
not millions, and a dashboard that needs a pre-aggregation table is a
dashboard that lies the first time someone forgets to backfill it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import current_doctor
from app.db.documents import Document
from app.db.identity import Doctor
from app.db.models import Event, Patient
from app.db.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class Kpis(BaseModel):
    patients: int
    patients_added_7d: int
    records: int
    records_added_today: int
    records_added_7d: int
    documents: int
    prescriptions: int
    reports: int
    certificates: int
    invoices: int
    documents_sent_today: int
    #: Scans that OCR could not confidently read and that are waiting for
    #: a human. Surfaced because a silently growing review queue is how
    #: records go missing.
    awaiting_review: int


class PatientRow(BaseModel):
    id: str
    display_name: str
    age: Optional[int]
    phone: Optional[str]
    email: Optional[str]
    photo_path: Optional[str]
    record_count: int
    last_activity: Optional[str]
    created_at: str


class DashboardOut(BaseModel):
    doctor_name: str
    specialty: Optional[str]
    clinic_name: Optional[str]
    kpis: Kpis
    recent_patients: List[PatientRow]
    #: The total, so the UI can say "showing 8 of 63" instead of implying
    #: this list is everything.
    total_patients: int


@router.get("", response_model=DashboardOut)
async def dashboard(
    limit: int = Query(default=8, ge=1, le=50),
    search: str = Query(default="", max_length=120),
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week = today - timedelta(days=7)

    # A doctor's patients are the rows carrying their doctor_id. Patients
    # with a null doctor_id predate identity linking and belong to nobody in
    # particular, so they are excluded rather than guessed at.
    scoped = Patient.doctor_id == doctor.id

    patient_ids = select(Patient.id).where(scoped).scalar_subquery()

    async def count_rows(stmt) -> int:
        return int((await session.execute(stmt)).scalar() or 0)

    total_patients = await count_rows(
        select(func.count(Patient.id)).where(scoped)
    )
    patients_7d = await count_rows(
        select(func.count(Patient.id)).where(
            scoped, Patient.created_at >= week
        )
    )
    records = await count_rows(
        select(func.count(Event.id)).where(Event.patient_id.in_(patient_ids))
    )
    records_today = await count_rows(
        select(func.count(Event.id)).where(
            Event.patient_id.in_(patient_ids), Event.recorded_at >= today
        )
    )
    records_7d = await count_rows(
        select(func.count(Event.id)).where(
            Event.patient_id.in_(patient_ids), Event.recorded_at >= week
        )
    )

    doc_scoped = Document.doctor_id == doctor.id

    async def count_docs(kind: Optional[str] = None) -> int:
        stmt = select(func.count(Document.id)).where(doc_scoped)
        if kind is not None:
            stmt = stmt.where(Document.kind == kind)
        return await count_rows(stmt)

    documents = await count_docs()
    prescriptions = await count_docs("prescription")
    reports = await count_docs("weekly_report")
    certificates = await count_docs("certificate")
    invoices = await count_docs("invoice")
    sent_today = await count_rows(
        select(func.count(Document.id)).where(
            doc_scoped, Document.created_at >= today
        )
    )
    awaiting_review = await count_rows(
        select(func.count(Document.id)).where(
            doc_scoped, Document.committed.is_(False)
        )
    )

    # One grouped query for record counts and last activity, rather than a
    # query per patient. Rows are three columns wide, so dict() over them
    # would fail; they are unpacked explicitly.
    activity: Dict[UUID, tuple] = {}
    if total_patients:
        for row in (
            await session.execute(
                select(
                    Event.patient_id,
                    func.count(Event.id),
                    func.max(Event.recorded_at),
                )
                .where(Event.patient_id.in_(patient_ids))
                .group_by(Event.patient_id)
            )
        ).all():
            activity[row[0]] = (row[1], row[2])

    stmt = select(Patient).where(scoped)
    if search.strip():
        # Substring match on name, phone or email. Deliberately not
        # case-sensitive in the database: a doctor typing a phone number
        # should not have to match its capitalisation.
        needle = f"%{search.strip().lower()}%"
        stmt = stmt.where(
            func.lower(Patient.display_name).like(needle)
            | func.lower(func.coalesce(Patient.phone, "")).like(needle)
            | func.lower(func.coalesce(Patient.email, "")).like(needle)
        )
    # Most recently active first: a doctor opening this wants the person
    # they just saw, not the alphabetically first.
    rows = list(
        (
            await session.execute(
                stmt.order_by(Patient.created_at.desc()).limit(limit)
            )
        ).scalars().all()
    )

    recent: List[PatientRow] = []
    for p in rows:
        count, last = activity.get(p.id, (0, None))
        recent.append(
            PatientRow(
                id=str(p.id),
                display_name=p.display_name,
                age=p.age,
                phone=p.phone,
                email=p.email,
                photo_path=p.photo_path,
                record_count=int(count or 0),
                last_activity=last.isoformat() if last else None,
                created_at=p.created_at.isoformat(),
            )
        )

    account = doctor.account
    return DashboardOut(
        doctor_name=account.display_name if account else "Doctor",
        specialty=doctor.specialty,
        clinic_name=doctor.clinic_name,
        total_patients=total_patients,
        kpis=Kpis(
            patients=total_patients,
            patients_added_7d=patients_7d,
            records=records,
            records_added_today=records_today,
            records_added_7d=records_7d,
            documents=documents,
            prescriptions=prescriptions,
            reports=reports,
            certificates=certificates,
            invoices=invoices,
            documents_sent_today=sent_today,
            awaiting_review=awaiting_review,
        ),
        recent_patients=recent,
    )
