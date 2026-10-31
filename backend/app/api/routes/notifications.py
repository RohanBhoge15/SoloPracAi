"""Notification routes and the patient's inbox.

Two audiences, one stream. A doctor's badge and a patient's bell read
the same Redis stream, filtered by recipient. Keeping them separate
would mean two delivery paths for one event, and the two would disagree.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import current_account, current_doctor
from app.core.config import get_settings
from app.db.documents import Document
from app.db.identity import Account, Doctor
from app.db.models import Event, Patient
from app.db.session import get_session
from app.services.notifications import get_bus
from app.services.storage import get_storage

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(tags=["notifications"])


# ── schemas ────────────────────────────────────────────────────────


class NotificationOut(BaseModel):
    id: str
    kind: str
    title: str
    body: str
    payload: Dict[str, Any]
    created_at: str
    read: bool = False


class InboxItem(BaseModel):
    document_id: str
    kind: str
    title: str
    reference: str
    size_bytes: Optional[int]
    issued_at: Optional[str]
    read: bool
    #: Present when the document has been superseded by a later issue,
    #: so the patient is not misled by an out-of-date prescription.
    superseded: bool = False
    doctor_name: Optional[str] = None
    doctor_specialty: Optional[str] = None


class ReadRequest(BaseModel):
    ids: List[str] = Field(default_factory=list)


# ── notifications ──────────────────────────────────────────────────


@router.get("/notifications")
async def list_notifications(
    limit: int = Query(default=30, ge=1, le=100),
    account: Account = Depends(current_account),
):
    items = await get_bus().read(str(account.id), limit=limit)
    return [n.as_dict() for n in items]


@router.get("/notifications/unread-count")
async def unread_count(account: Account = Depends(current_account)):
    """The bell's number.

    Counted from the notifications actually marked unread, not from the
    length of the list. Returning `len(items)` counted everything ever
    published to this account, so the badge was a running total that never
    fell - and a patient who read every notification still saw a number.
    """
    bus = get_bus()
    unread = await bus.unread_count(str(account.id))
    total = len(await bus.read(str(account.id), limit=100))
    return {"unread": unread, "total": total}


@router.post("/notifications/read")
async def mark_read(
    body: ReadRequest,
    account: Account = Depends(current_account),
):
    if not body.ids:
        return {"marked": 0}
    marked = await get_bus().mark_read(str(account.id), body.ids)
    return {"marked": marked}


@router.get("/notifications/health")
async def notifications_health():
    return await get_bus().health()


# ── the patient's inbox ────────────────────────────────────────────


@router.get("/patient/inbox", response_model=List[InboxItem])
async def inbox(
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """Everything addressed to this patient, across every doctor.

    Superseded documents stay in the list but are flagged: a patient
    holding a paper prescription that was later reissued should be able
    to see that it was replaced, not find it silently missing.
    """
    if account.role != "patient":
        raise HTTPException(
            status_code=403, detail="inbox is for patient accounts"
        )

    # Every record this account owns, across all doctors. Using only the
    # newest would silently hide documents from a patient's other
    # clinicians, which is the opposite of what the portal promises.
    patients = await _account_patients(session, account)
    if not patients:
        raise HTTPException(
            status_code=404,
            detail="no clinical record linked to this account yet",
        )
    patient_ids = [p.id for p in patients]

    rows = (
        await session.execute(
            select(Document)
            .where(
                Document.patient_id.in_(patient_ids),
                Document.issued_to_patient.is_(True),
            )
            .order_by(Document.created_at.desc())
        )
    ).scalars().all()

    # Doctor details for the notification chips: "sent by X" with a face.
    doctor_ids = {d.doctor_id for d in rows if d.doctor_id}
    doctors: Dict[UUID, Dict[str, Any]] = {}
    if doctor_ids:
        doc_rows = (
            await session.execute(select(Doctor).where(Doctor.id.in_(doctor_ids)))
        ).scalars().all()
        for d in doc_rows:
            doc_account = await session.get(Account, d.account_id)
            doctors[d.id] = {
                "name": doc_account.display_name if doc_account else "Doctor",
                "specialty": d.specialty,
                "photo_path": doc_account.photo_path if doc_account else None,
            }

    out: List[InboxItem] = []
    for d in rows:
        info = doctors.get(d.doctor_id, {}) if d.doctor_id else {}
        out.append(
            InboxItem(
                document_id=str(d.id),
                kind=d.kind,
                title=d.title or d.kind,
                reference=(d.content or {}).get("reference", str(d.id)[:8]),
                size_bytes=d.size_bytes,
                issued_at=d.issued_at.isoformat() if d.issued_at else None,
                read=d.read_at is not None,
                superseded=d.superseded_by_id is not None,
                doctor_name=info.get("name"),
                doctor_specialty=info.get("specialty"),
            )
        )
    return out


@router.get("/patient/documents/{document_id}/open")
async def open_document(
    document_id: UUID,
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """A presigned link to one of this patient's own documents.

    Ownership is checked against the patient's own record rather than
    trusted from the id, because a presigned URL is a bearer token.
    """
    doc = await session.get(Document, document_id)
    if doc is None or not doc.object_key:
        raise HTTPException(status_code=404, detail="document not found")

    if account.role == "patient":
        # Checked against every clinical record this account owns, not just
        # the newest one: a patient who has seen two doctors has two
        # Patient rows, and documents under both.
        owned = {row.id for row in await _account_patients(session, account)}
        if doc.patient_id not in owned:
            raise HTTPException(
                status_code=403, detail="that document is not yours"
            )
    elif account.role == "doctor":
        # Not merely "some doctor issued this". Two practices sharing one
        # server must not be able to read each other's prescriptions, and a
        # document with no issuing doctor belongs to nobody.
        doctor = await current_doctor(account=account, session=session)
        if doc.doctor_id is None or doc.doctor_id != doctor.id:
            raise HTTPException(
                status_code=403, detail="that document is not yours"
            )
    else:
        raise HTTPException(status_code=403, detail="unrecognised account role")

    if doc.read_at is None:
        doc.read_at = datetime.now(timezone.utc)
        await session.flush()

    url = await get_storage().presigned_url(doc.object_key)
    return {
        "url": url,
        "filename": f"{(doc.content or {}).get('reference', str(doc.id)[:8])}-{doc.kind}.pdf",
        "content_type": doc.content_type or "application/pdf",
        "size_bytes": doc.size_bytes,
        "expires_in": settings.download_url_ttl_seconds,
    }


@router.get("/patient/records")
async def patient_records(
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """Every doctor this patient has seen, and their record counts.

    A patient's identity is one account but their records are several,
    and the inbox deliberately spans all of them. This endpoint is where
    the portal shows that.
    """
    if account.role != "patient":
        raise HTTPException(status_code=403, detail="patient account required")

    patients = await _account_patients(session, account)
    if not patients:
        return []

    # One query for every doctor's record rather than a loop, so the count
    # cost does not scale with the number of clinicians.
    counts = dict(
        (
            await session.execute(
                select(Event.patient_id, func.count(Event.id))
                .where(Event.patient_id.in_([p.id for p in patients]))
                .group_by(Event.patient_id)
            )
        ).all()
    )

    # Resolve the display names of every doctor involved in one pass.
    doctor_ids = {p.doctor_id for p in patients if p.doctor_id}
    details: Dict[UUID, Dict[str, Any]] = {}
    if doctor_ids:
        doctor_rows = (
            await session.execute(
                select(Doctor).where(Doctor.id.in_(doctor_ids))
            )
        ).scalars().all()
        account_ids = [d.account_id for d in doctor_rows]
        name_by_account: Dict[UUID, str] = {}
        # The account id as well as the name: the portal needs it to ask
        # for the doctor's picture. Sending the picture itself would mean
        # embedding a presigned URL that expires inside a list the client
        # caches, so the id is sent and the client fetches a fresh one.
        account_id_by_doctor: Dict[UUID, UUID] = {}
        has_photo: Dict[UUID, bool] = {}
        if account_ids:
            for acct in (
                await session.execute(
                    select(Account).where(Account.id.in_(account_ids))
                )
            ).scalars().all():
                name_by_account[acct.id] = acct.display_name
                has_photo[acct.id] = bool(acct.photo_path)
        for d in doctor_rows:
            account_id_by_doctor[d.id] = d.account_id
            details[d.id] = {
                "name": name_by_account.get(d.account_id),
                "specialty": d.specialty,
                "account_id": str(d.account_id) if d.account_id else None,
                # A hint, not the image. False avoids a pointless request
                # for the many doctors who never set a picture.
                "doctor_has_photo": has_photo.get(d.account_id, False),
            }

    out = []
    for p in patients:
        info = details.get(p.doctor_id, {}) if p.doctor_id else {}
        out.append(
            {
                "patient_id": str(p.id),
                "display_name": p.display_name,
                "doctor_name": info.get("name"),
                "specialty": info.get("specialty"),
                "doctor_account_id": info.get("account_id"),
                "doctor_has_photo": info.get("doctor_has_photo", False),
                "event_count": int(counts.get(p.id, 0)),
                "since": p.created_at.isoformat() if p.created_at else None,
            }
        )
    return out


async def _account_patients(
    session: AsyncSession, account: Account
) -> List[Patient]:
    """Every clinical record linked to this account, newest first.

    A patient account is one identity but may have several Patient rows,
    one per doctor who has seen them. Anything that needs "this patient's
    records" must span all of them, or it will quietly under-report.
    """
    return list(
        (
            await session.execute(
                select(Patient)
                .where(Patient.user_id == account.id)
                .order_by(Patient.created_at.desc())
            )
        ).scalars().all()
    )
