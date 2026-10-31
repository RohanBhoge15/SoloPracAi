"""Document routes: generate, store, list, download.

Every generated document is:
  1. rendered to PDF,
  2. stored in object storage,
  3. recorded as a row,
  4. written into the clinical event log as a `document` event, so the
     record knows a prescription exists and can answer questions about it.

A regenerated prescription supersedes the previous one rather than
replacing it, keeping the same append-only discipline as lab corrections.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_patient
from app.api.routes.auth import current_account, current_doctor
from app.core.config import get_settings
from app.core.validity import Event as CoreEvent
from app.core.validity import EventType, SourceType, content_hash
from app.db.documents import Document
from app.db.identity import Account, Doctor
from app.db.models import Event, Patient
from app.db.session import get_session
from app.services.pdf import PDFError, pdf_available, render_html_to_pdf
from app.services.storage import get_storage

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/documents", tags=["documents"])


# ── schemas ────────────────────────────────────────────────────────


class PrescriptionItemIn(BaseModel):
    drug: str = Field(min_length=1, max_length=200)
    dose: str = Field(default="", max_length=80)
    frequency: str = Field(default="", max_length=80)
    duration: str = Field(default="", max_length=80)
    route: str = Field(default="", max_length=40)
    timing: str = Field(default="", max_length=200)
    instruction: str = Field(default="", max_length=400)


class PrescriptionIn(BaseModel):
    patient_id: UUID
    items: List[PrescriptionItemIn] = Field(min_length=1, max_length=30)
    notes: str = Field(default="", max_length=4000)
    follow_up_days: Optional[int] = Field(default=None, ge=1, le=365)
    #: What the drugs are for. Accepted but previously discarded before
    #: the document was rendered, so the patient received a list of drugs
    #: with no indication attached.
    diagnosis: str = Field(default="", max_length=500)
    #: Issue to the patient's inbox immediately.
    send_to_patient: bool = True
    supersedes_id: Optional[UUID] = None


class CertificateIn(BaseModel):
    patient_id: UUID
    certificate_type: str = Field(default="Medical Certificate", max_length=80)
    body_text: str = Field(min_length=1, max_length=6000)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    send_to_patient: bool = True


class InvoiceLineIn(BaseModel):
    description: str = Field(min_length=1, max_length=200)
    quantity: int = Field(default=1, ge=1, le=999)
    unit_price: int = Field(default=0, ge=0, le=10_000_000)


class InvoiceIn(BaseModel):
    patient_id: UUID
    lines: List[InvoiceLineIn] = Field(min_length=1, max_length=50)
    discount: int = Field(default=0, ge=0)
    tax_percent: float = Field(default=0.0, ge=0, le=100)
    paid: bool = False
    send_to_patient: bool = True


class WeeklyReportIn(BaseModel):
    patient_id: UUID
    #: Upper bound is generous: a "whole record" report over a long
    #: history is a legitimate request, not an abuse vector.
    days: int = Field(default=7, ge=1, le=3650)
    send_to_patient: bool = False


class DocumentOut(BaseModel):
    id: UUID
    patient_id: Optional[UUID]
    doctor_id: Optional[UUID]
    kind: str
    title: str
    description: Optional[str]
    filename: Optional[str]
    content_type: Optional[str]
    size_bytes: Optional[int]
    content: Dict[str, Any]
    extracted: Dict[str, Any]
    ocr_confidence: Optional[float]
    committed: bool
    issued_to_patient: bool
    issued_at: Optional[datetime]
    read_at: Optional[datetime]
    version: int
    is_current: bool
    #: Human-facing, e.g. "PRE-20260930-4F2A9C". Not a column: it lives
    #: in `content` so it can be stamped into the row and the PDF at once.
    reference: str
    created_at: datetime


# ── helpers ────────────────────────────────────────────────────────


def _reference(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{kind.upper()[:3]}-{stamp}-{uuid.uuid4().hex[:6].upper()}"


async def _doctor_payload(session: AsyncSession, doctor: Doctor) -> Dict[str, Any]:
    from app.db.identity import Account

    account = await session.get(Account, doctor.account_id)
    return {
        "id": str(doctor.id),
        "account": {
            "display_name": account.display_name if account else "",
            "email": account.email if account else "",
            "phone": account.phone if account else None,
        },
        "display_name": account.display_name if account else "",
        "specialty": doctor.specialty,
        "clinic_name": doctor.clinic_name,
        "nmc_number": doctor.nmc_number,
        "registration_state": doctor.registration_state,
    }


def _patient_payload(patient: Patient) -> Dict[str, Any]:
    return {
        "id": str(patient.id),
        "display_name": patient.display_name,
        "age": patient.age,
        "phone": patient.phone,
        "abha_number": patient.abha_number,
    }


async def _persist(
    session: AsyncSession,
    *,
    patient: Patient,
    doctor: Doctor,
    kind: str,
    title: str,
    html: str,
    content: Dict[str, Any],
    filename: str,
    send_to_patient: bool,
    description: str = "",
    extracted: Optional[Dict[str, Any]] = None,
    supersedes_id: Optional[UUID] = None,
) -> Document:
    """Render, store, record, and log. The single path for every document."""
    if not pdf_available():
        raise HTTPException(
            status_code=503,
            detail="PDF engine unavailable (weasyprint/Pango missing in this image)",
        )

    pdf = render_html_to_pdf(html)
    storage = get_storage()
    prefix = f"{kind}s/{patient.id}"
    stored = await storage.put(
        pdf, prefix, filename, content_type="application/pdf"
    )

    now = datetime.now(timezone.utc)
    reference = _reference(kind)
    doc = Document(
        patient_id=patient.id,
        doctor_id=doctor.id,
        kind=kind,
        title=title,
        description=description or None,
        object_key=stored.key,
        content_type="application/pdf",
        size_bytes=stored.size,
        # The reference is generated before the row so it can be stamped
        # into the content in one go, rather than patched afterwards.
        content={**content, "reference": reference},
        extracted=extracted or {},
        issued_to_patient=send_to_patient,
        issued_at=now if send_to_patient else None,
        supersedes_id=supersedes_id,
    )
    session.add(doc)
    await session.flush()

    if supersedes_id:
        previous = await session.get(Document, supersedes_id)
        if previous is not None:
            previous.superseded_by_id = doc.id

    # The document becomes a record version, so chat can answer questions
    # about it: "what did you prescribe last time?"
    #
    # This has to be the previous document's *event* id, not the document
    # id: the documents table and the event log are separate graphs, and
    # the events foreign key rejects a document id here.
    superseded_event: Optional[UUID] = None
    if supersedes_id:
        superseded_event = await _document_event_id(session, supersedes_id)
        if superseded_event is None:
            raise HTTPException(
                status_code=409,
                detail="the document being superseded has no record version",
            )

    payload: Dict[str, Any] = {
        "document_id": str(doc.id),
        "document_kind": kind,
        "document_reference": (doc.content or {}).get("reference"),
        "title": title,
    }
    # Items and lines arrive as plain dicts once they have been through
    # pydantic, so they are copied rather than re-serialised.
    if content.get("items"):
        payload["items"] = list(content["items"])
    if content.get("lines"):
        payload["lines"] = list(content["lines"])
    if content.get("total") is not None:
        payload["amount"] = content["total"]

    core = CoreEvent(
        event_id="",
        patient_id=str(patient.id),
        event_type=EventType.NOTE,
        valid_time=now,
        payload=payload,
        # A document that supersedes another is a correction, and the
        # event log has no separate vocabulary for that. Saying so keeps
        # the ledger honest: a re-issued prescription genuinely makes the
        # previous one no longer the current advice.
        source_type=(
            SourceType.CORRECTION if superseded_event else SourceType.MANUAL
        ),
        supersedes_id=str(superseded_event) if superseded_event else None,
    )
    row = Event.from_core(core)
    row.content_hash = content_hash(core)
    session.add(row)
    await session.flush()

    if send_to_patient:
        try:
            from app.services.notifications import notify_document_issued

            await notify_document_issued(
                session, patient=patient, doctor=doctor, document=doc
            )
        except Exception:  # noqa: BLE001
            # A notification failure must not lose the prescription.
            logger.warning(
                "notification failed for %s",
                (doc.content or {}).get("reference", doc.id),
                exc_info=True,
            )

    await session.refresh(doc)
    return doc


async def _document_event_id(session: AsyncSession, document_id: UUID):
    """The event id a document wrote into the record, if any.

    Documents and events are two graphs: `documents.supersedes_id` points
    at a document, `events.supersedes_id` at an event. Resolving between
    them here keeps the confusion in one place.
    """
    from sqlalchemy import text as sql_text

    result = await session.execute(
        sql_text(
            "SELECT id FROM events "
            "WHERE payload->>'document_id' = :did "
            "ORDER BY recorded_at ASC LIMIT 1"
        ),
        {"did": str(document_id)},
    )
    row = result.first()
    return row[0] if row else None


def _out(doc: Document) -> Dict[str, Any]:
    """Serialise a document for the API.

    Built explicitly rather than via `from_attributes`, because
    `reference` is derived from `content` and has no column behind it.
    """
    content = doc.content or {}
    return {
        "id": str(doc.id),
        "patient_id": str(doc.patient_id) if doc.patient_id else None,
        "doctor_id": str(doc.doctor_id) if doc.doctor_id else None,
        "kind": doc.kind,
        "title": doc.title or doc.filename or doc.kind,
        "description": doc.description,
        "filename": doc.filename,
        "content_type": doc.content_type,
        "size_bytes": doc.size_bytes,
        "content": content,
        "extracted": doc.extracted or {},
        "ocr_confidence": doc.ocr_confidence,
        "committed": doc.committed,
        "issued_to_patient": doc.issued_to_patient,
        "issued_at": doc.issued_at.isoformat() if doc.issued_at else None,
        "read_at": doc.read_at.isoformat() if doc.read_at else None,
        "version": doc.version,
        "is_current": doc.is_current,
        "reference": content.get("reference") or str(doc.id)[:8].upper(),
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


# ── routes ────────────────────────────────────────────────────────


@router.post("/weekly-report", response_model=DocumentOut, status_code=201)
async def create_weekly_report(
    body: WeeklyReportIn,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    """A clinician-facing summary of a period, grounded in the record.

    The tables are built deterministically from the versioned record; the
    model only writes the narrative. That split is deliberate: the numbers
    must be the record's numbers, and a slow free tier must not be able to
    invent a lab value.
    """
    from app.core.validity import SupersessionGraph
    from app.documents import templates, weekly

    patient = await session.get(Patient, body.patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")

    from app.api.deps import load_core_events
    from app.core.config import get_settings
    from app.services.llm import LLMClient

    events = await load_core_events(patient.id, session)
    if not events:
        raise HTTPException(
            status_code=400, detail="no records to report on"
        )

    graph = SupersessionGraph(events)
    period = weekly.build_period(events, body.days)
    window, pairs = weekly.collect_period_events(events, body.days)
    record_text, cited = weekly.render_for_prompt(window, pairs, graph)

    settings = get_settings()
    llm = LLMClient(settings) if settings.llm_configured else None
    d = await _doctor_payload(session, doctor)
    try:
        summary = await weekly.draft_summary(
            llm,
            patient=_patient_payload(patient),
            doctor=d,
            period=period,
            record_text=record_text,
            cited=cited,
        )
    finally:
        if llm is not None:
            await llm.__aexit__(None, None, None)

    html = templates.weekly_report(
        doctor=d,
        patient=_patient_payload(patient),
        summary=summary,
        findings=period["findings"],
        medications=period["medications"],
        corrections=period["corrections"],
        retrieved_context=period.get("retrieved_context") or cited[:12],
    )

    doc = await _persist(
        session,
        patient=patient,
        doctor=doctor,
        kind="weekly_report",
        title=f"Weekly report for {patient.display_name}",
        html=html,
        content={**period, "summary": summary},
        filename="weekly-report.pdf",
        send_to_patient=body.send_to_patient,
        description=summary[:200],
        extracted={"cited_events": cited[:20]},
    )
    return _out(doc)


@router.get("/capabilities")
async def capabilities(account: Account = Depends(current_account)):
    """What this deployment can actually do, rather than what it intends to.

    Authenticated. No patient data is exposed, but the response reports
    whether PDF rendering, object storage and the LLM are live, plus the
    bucket name. That is a service map for anyone probing the deployment,
    and nothing legitimate about it needs to be public.
    """
    storage = get_storage()
    return {
        "pdf": pdf_available(),
        "storage": await storage.health(),
        "llm": settings.llm_configured,
        "kinds": [
            "prescription", "certificate", "invoice", "weekly_report", "scan"
        ],
    }


@router.post("/prescription", response_model=DocumentOut, status_code=201)
async def create_prescription(
    body: PrescriptionIn,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    from app.documents import templates

    patient = await session.get(Patient, body.patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")

    d = await _doctor_payload(session, doctor)
    p = _patient_payload(patient)
    items = [i.model_dump() for i in body.items]

    html = templates.prescription(
        doctor=d,
        patient=p,
        items=items,
        notes=body.notes,
        follow_up_days=body.follow_up_days,
        diagnosis=body.diagnosis,
        reference="",
    )

    doc = await _persist(
        session,
        patient=patient,
        doctor=doctor,
        kind="prescription",
        title=f"Prescription for {patient.display_name}",
        html=html,
        content={
            "items": items,
            "notes": body.notes,
            "follow_up_days": body.follow_up_days,
        },
        filename="prescription.pdf",
        send_to_patient=body.send_to_patient,
        description=body.notes[:200] or None,
        supersedes_id=body.supersedes_id,
    )
    # A prescription is clinical facts, not just a PDF. The diagnosis the
    # doctor typed and every drug on it become first-class events, or the
    # record can never answer "what diseases does this patient have" and
    # "what is the patient taking" after prescribing. Seen live: Yellow
    # Fever typed into the composer existed only as ink in the PDF, and a
    # disease summary came back without it.
    now = datetime.now(timezone.utc)
    diagnosis = (body.diagnosis or "").strip()
    if diagnosis:
        core = CoreEvent(
            event_id="",
            patient_id=str(patient.id),
            event_type=EventType.DIAGNOSIS,
            valid_time=now,
            payload={"diagnosis": diagnosis},
            source_type=SourceType.MANUAL,
        )
        row = Event.from_core(core)
        row.content_hash = content_hash(core)
        session.add(row)
    for item in items:
        drug = str((item or {}).get("drug") or "").strip()
        if not drug:
            continue
        core = CoreEvent(
            event_id="",
            patient_id=str(patient.id),
            event_type=EventType.MEDICATION,
            valid_time=now,
            payload={
                "drug": drug,
                "dose": str((item or {}).get("dose") or ""),
                "frequency": str((item or {}).get("frequency") or ""),
                "status": "active",
            },
            source_type=SourceType.MANUAL,
        )
        row = Event.from_core(core)
        row.content_hash = content_hash(core)
        session.add(row)
    await session.flush()
    # Stamp the reference into the content after it exists so the
    # document row and the PDF agree.
    return _out(doc)


@router.post("/certificate", response_model=DocumentOut, status_code=201)
async def create_certificate(
    body: CertificateIn,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    from app.documents import templates

    patient = await session.get(Patient, body.patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")

    d = await _doctor_payload(session, doctor)
    html = templates.certificate(
        doctor=d,
        patient=_patient_payload(patient),
        certificate_type=body.certificate_type,
        body_text=body.body_text,
        valid_from=body.valid_from,
        valid_until=body.valid_until,
        reference="",
    )
    doc = await _persist(
        session,
        patient=patient,
        doctor=doctor,
        kind="certificate",
        title=f"{body.certificate_type} for {patient.display_name}",
        html=html,
        content={
            "certificate_type": body.certificate_type,
            "body_text": body.body_text,
            "valid_from": body.valid_from.isoformat() if body.valid_from else None,
            "valid_until": body.valid_until.isoformat() if body.valid_until else None,
        },
        filename="certificate.pdf",
        send_to_patient=body.send_to_patient,
        description=body.body_text[:200],
    )
    return _out(doc)


@router.post("/invoice", response_model=DocumentOut, status_code=201)
async def create_invoice(
    body: InvoiceIn,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    from app.documents import templates

    patient = await session.get(Patient, body.patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")

    lines = [l.model_dump() for l in body.lines]
    subtotal = sum(l["quantity"] * l["unit_price"] for l in lines)
    discount = max(0, min(body.discount, subtotal))
    taxable = subtotal - discount
    tax = int(round(taxable * body.tax_percent / 100.0))
    total = taxable + tax

    d = await _doctor_payload(session, doctor)
    html = templates.invoice(
        doctor=d,
        patient=_patient_payload(patient),
        lines=lines,
        discount=body.discount,
        tax_percent=body.tax_percent,
        paid=body.paid,
        reference="",
    )
    doc = await _persist(
        session,
        patient=patient,
        doctor=doctor,
        kind="invoice",
        title=f"Invoice for {patient.display_name}",
        html=html,
        content={
            "lines": lines,
            "subtotal": subtotal,
            "discount": discount,
            "tax": tax,
            "total": total,
            "paid": body.paid,
            "currency": "INR",
        },
        filename="invoice.pdf",
        send_to_patient=body.send_to_patient,
        description=f"Total INR {total}",
    )
    return _out(doc)


@router.get("/photo/url")
async def current_photo_url(
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """A short-lived presigned URL for the signed-in account's photo.

    The browser cannot use `/photo` directly, and this is why.

    An `<img src=...>` cannot send an `Authorization` header. It also
    cannot run JavaScript to attach one. So a redirect served from behind
    `current_account` is guaranteed to return 401 to the one client that
    actually needs it, and the `<img>` fails to load.

    That is not hypothetical: the profile picture uploaded correctly,
    `photo_path` was set, and the avatar rendered as *blank space*,
    because the image 401'd and the `onError` fallback hid it. It looked
    like the upload had done nothing.

    So the flow is the same one the PDFs already use. An authenticated
    `fetch` (which can send the header) asks for a URL, and the `<img>`
    then loads straight from object storage, where no auth is needed.
    The trade is one extra request per avatar; the benefit is that it
    works at all, and that object storage can serve the bytes.
    """
    if not account.photo_path:
        raise HTTPException(status_code=404, detail="no profile picture")
    url = await get_storage().presigned_url(
        account.photo_path, ttl=settings.avatar_url_ttl_seconds
    )
    return {"url": url, "version": account.photo_version or 1}


@router.get("/photo", include_in_schema=False)
async def current_photo(
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """Redirect to the signed-in account's own profile picture.

    Kept for API clients that *can* send the header. The browser must use
    `/photo/url` instead, for the reason documented there.
    """
    if not account.photo_path:
        raise HTTPException(status_code=404, detail="no profile picture")
    url = await get_storage().presigned_url(
        account.photo_path, ttl=settings.avatar_url_ttl_seconds
    )
    return RedirectResponse(url, status_code=307)


@router.get("/avatars/{account_id}")
async def account_photo_url(
    account_id: UUID,
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """A presigned URL for another account's avatar.

    A patient needs to see which doctor sent them a document, and an
    avatar is the least sensitive thing either party holds. Everything
    else about an account stays behind `/auth/me`.

    The id is in the path, not a query, so a caller cannot enumerate
    accounts by asking a range of them and timing the difference between
    "no such account" and "no picture". Both return 404 for that reason.

    Returns a URL rather than redirecting, for the same reason as
    `/photo/url`: an `<img>` cannot send the Authorization header, so a
    redirect behind auth 401s to the one client that needs it.
    """
    target = await session.get(Account, account_id)
    if target is None or not target.photo_path:
        raise HTTPException(status_code=404, detail="no profile picture")
    url = await get_storage().presigned_url(
        target.photo_path, ttl=settings.avatar_url_ttl_seconds
    )
    return {"url": url, "version": target.photo_version or 1}


@router.get("/avatars/{account_id}/photo", include_in_schema=False)
async def account_photo(
    account_id: UUID,
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """Redirect to another account's picture.

    Kept for API clients that *can* send the header. The browser must use
    `/avatars/{id}` instead, for the reason documented there.
    """
    target = await session.get(Account, account_id)
    if target is None or not target.photo_path:
        raise HTTPException(status_code=404, detail="no profile picture")
    url = await get_storage().presigned_url(
        target.photo_path, ttl=settings.avatar_url_ttl_seconds
    )
    return RedirectResponse(url, status_code=307)


@router.get("/patient/{patient_id}")
async def list_documents(
    patient_id: UUID,
    kind: Optional[str] = Query(default=None),
    include_superseded: bool = Query(default=True),
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """Documents for one of the calling doctor's own patients.

    `get_patient` is the authorization: it 404s unless the row belongs to
    the caller. This endpoint previously took a bare patient_id with no
    auth at all, so any caller who guessed or captured a UUID could list
    another practice's prescriptions, certificates and invoices. Those
    carry the patient's name, age and phone on the document itself.
    """
    stmt = select(Document).where(Document.patient_id == patient.id)
    if kind:
        stmt = stmt.where(Document.kind == kind)
    if not include_superseded:
        stmt = stmt.where(Document.superseded_by_id.is_(None))
    stmt = stmt.order_by(Document.created_at.desc())
    rows = (await session.execute(stmt)).scalars().all()
    return [_out(r) for r in rows]


@router.get("/{document_id}/download")
async def download_url(
    document_id: UUID,
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """A time-limited direct link.

    Served by the object store rather than proxied, so a large report does
    not occupy an API worker.

    Authorized on the way in. The presigned URL is a bearer token that
    grants full read access to the file, and these documents carry the
    patient's name, age and phone on the page, so an unauthenticated
    request that returned one was equivalent to publishing the patient's
    details to whoever held the UUID.

    The caller must be the issuing doctor, or the patient the document was
    sent to. 404 rather than 403 for either miss, so this cannot be used to
    probe which document ids exist.
    """
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    if not doc.object_key:
        raise HTTPException(status_code=404, detail="no file attached")
    if not await _may_read_document(doc, account, session):
        raise HTTPException(status_code=404, detail="document not found")

    url = await get_storage().presigned_url(doc.object_key)
    return {
        "url": url,
        "filename": f"{doc.content.get('reference', doc.id)}-{doc.kind}.pdf",
        "content_type": doc.content_type,
        "size_bytes": doc.size_bytes,
        "expires_in": settings.download_url_ttl_seconds,
    }


async def _may_read_document(
    doc: Document, account: Account, session: AsyncSession
) -> bool:
    """Whether this account may read this document.

    Two legitimate readers, and they are checked separately because they
    are different relationships:

      - the doctor who issued it, or who owns the patient row it is on
      - the patient it was sent to, matched on the account that the
        Patient row was linked to

    Anything else is refused. In particular a doctor who merely shares a
    patient *person* with the issuer does not automatically gain access to
    that doctor's documents - the rows are deliberately separate.
    """
    if account.role == "doctor":
        doctor = (
            await session.execute(
                select(Doctor).where(Doctor.account_id == account.id)
            )
        ).scalars().first()
        if doctor is None:
            return False
        if doc.doctor_id == doctor.id:
            return True
        owner = await session.get(Patient, doc.patient_id)
        return owner is not None and owner.doctor_id == doctor.id

    if account.role == "patient":
        owned = (
            await session.execute(
                select(Patient).where(Patient.email == account.email)
            )
        ).scalars().all()
        return any(row.id == doc.patient_id for row in owned)

    return False


@router.get("/{document_id}/raw")
async def download_raw(
    document_id: UUID,
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
):
    """Stream the bytes through the API. Fallback for signed-URL issues.

    Same authorization as `/download`. Being a fallback does not make it a
    second, weaker door.
    """
    doc = await session.get(Document, document_id)
    if doc is None or not doc.object_key:
        raise HTTPException(status_code=404, detail="document not found")
    if not await _may_read_document(doc, account, session):
        raise HTTPException(status_code=404, detail="document not found")
    data = await get_storage().get(doc.object_key)
    return Response(
        content=data,
        media_type=doc.content_type or "application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{doc.content.get("reference", doc.id)}.pdf"'
            )
        },
    )
