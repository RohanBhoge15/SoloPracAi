"""Registration, login, and profile.

Two flows that must not be confused:

  - A *doctor* registers with an email and a password. That email is the
    identity and is required.
  - A *patient* either registers for themselves (email + password), or
    is created by a doctor as a walk-in with no login at all. When they
    later register with the same email, the linker joins the two.

Every registration normalises the email before it is stored. If it is not
normalised, `Rakesh@x.com` and `rakesk@x.com` become two accounts and the
auto-linking guarantee quietly fails.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import (
    TokenError,
    TokenSigner,
    hash_password,
    normalise_email,
    verify_password,
)
from app.db.identity import Account, Doctor
from app.db.models import Patient
from app.db.session import get_session
from app.identity.linker import IdentityLinker
from app.services.storage import get_storage

logger = logging.getLogger(__name__)
settings = get_settings()
#: No /api/v1 here: main.py applies that prefix to every router, and
#: doubling it produces /api/v1/api/v1/auth.
router = APIRouter(prefix="/auth", tags=["auth"])

_signer = TokenSigner(
    secret=settings.token_secret,
    ttl_seconds=settings.session_ttl_seconds,
)


# ── schemas ────────────────────────────────────────────────────────


class DoctorRegister(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=200)
    display_name: str = Field(min_length=1, max_length=200)
    phone: str | None = None
    specialty: str | None = None
    clinic_name: str | None = None
    nmc_number: str | None = None
    registration_state: str | None = None
    consult_fee: int | None = Field(default=None, ge=0)


class PatientRegister(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=200)
    display_name: str = Field(min_length=1, max_length=200)
    phone: str | None = None
    age: int | None = Field(default=None, ge=0, le=130)


class LoginRequest(BaseModel):
    email: str
    password: str


class AccountOut(BaseModel):
    id: UUID
    email: str
    role: str
    display_name: str
    phone: str | None
    photo_path: str | None
    photo_version: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DoctorOut(BaseModel):
    id: UUID
    account_id: UUID
    specialty: str | None
    clinic_name: str | None
    nmc_number: str | None
    registration_state: str | None
    consult_fee: int | None
    signature_path: str | None

    model_config = {"from_attributes": True}


class SessionOut(BaseModel):
    token: str
    account: AccountOut
    doctor: DoctorOut | None = None
    #: Records this account inherited by email match, on registration.
    linked_records: int = 0
    patients_visible: int = 0


# ── helpers ────────────────────────────────────────────────────────


def _normalise_or_422(raw: str) -> str:
    email = normalise_email(raw)
    if email is None:
        raise HTTPException(
            status_code=422,
            detail=f"'{raw}' is not a valid email address",
        )
    return email


async def _account_by_email(session: AsyncSession, email: str) -> Optional[Account]:
    return (
        await session.execute(
            select(Account).where(Account.email == email)
        )
    ).scalars().first()


async def _session_payload(
    session: AsyncSession, account: Account, linked: int = 0
) -> SessionOut:
    doctor = None
    if account.role == "doctor":
        doctor = (
            await session.execute(
                select(Doctor).where(Doctor.account_id == account.id)
            )
        ).scalars().first()

    visible = (
        await session.execute(
            select(func.count(Patient.id)).where(Patient.user_id == account.id)
        )
    ).scalar_one()

    return SessionOut(
        token=_signer.sign({"sub": str(account.id), "role": account.role}),
        account=AccountOut.model_validate(account),
        doctor=DoctorOut.model_validate(doctor) if doctor else None,
        linked_records=linked,
        patients_visible=int(visible or 0),
    )


# ── routes ────────────────────────────────────────────────────────


@router.post("/register/doctor", response_model=SessionOut, status_code=201)
async def register_doctor(body: DoctorRegister, session: AsyncSession = Depends(get_session)):
    """Doctor sign-up. Email is required - it is the identity.

    A doctor has to be reachable: prescriptions are watermarked with their
    name, and a patient must be able to see who issued one.
    """
    email = _normalise_or_422(body.email)

    if await _account_by_email(session, email):
        raise HTTPException(status_code=409, detail="an account already uses that email")

    account = Account(
        email=email,
        password_hash=hash_password(body.password),
        role="doctor",
        display_name=body.display_name.strip(),
        phone=body.phone,
    )
    session.add(account)
    await session.flush()

    doctor = Doctor(
        account_id=account.id,
        specialty=body.specialty,
        clinic_name=body.clinic_name,
        nmc_number=body.nmc_number,
        registration_state=body.registration_state,
        consult_fee=body.consult_fee,
    )
    session.add(doctor)
    await session.flush()
    await session.refresh(account)

    logger.info("doctor registered: %s", email)
    return await _session_payload(session, account)


@router.post("/register/patient", response_model=SessionOut, status_code=201)
async def register_patient(
    body: PatientRegister, session: AsyncSession = Depends(get_session)
):
    """Patient self sign-up.

    After the account exists, every unlinked record a doctor previously
    created with this email is claimed automatically. That is the whole
    point of requiring email on doctor-created patients.
    """
    email = _normalise_or_422(body.email)

    if await _account_by_email(session, email):
        raise HTTPException(status_code=409, detail="an account already uses that email")

    account = Account(
        email=email,
        password_hash=hash_password(body.password),
        role="patient",
        display_name=body.display_name.strip(),
        phone=body.phone,
    )
    session.add(account)
    await session.flush()

    linker = IdentityLinker(session)
    result = await linker.link_for_account(account.id, email)
    await session.flush()
    await session.refresh(account)

    if result.linked:
        logger.info(
            "patient %s inherited %d doctor record(s) by email match",
            email,
            len(result.linked),
        )

    # Documents that were issued to this person *before* they had an account
    # were never announced - there was nobody to announce them to. The
    # common order for a new patient is exactly that: the doctor writes the
    # prescription at the consultation, and the patient creates their portal
    # afterwards. Without this, the first thing they see is documents in
    # their inbox with no bell and no unread marker anywhere, which reads
    # as nothing having happened.
    #
    # Announced once, at the moment the link is made. Deliberately not a
    # per-document replay on every sign-in: that would re-ring the bell for
    # documents already seen. The unread state lives in Redis and is
    # per-recipient, so this is naturally once-per-recipient.
    if result.linked:
        await _announce_backdated_documents(session, result.linked, account.id)

    return await _session_payload(session, account, linked=len(result.linked))


async def _announce_backdated_documents(
    session: AsyncSession, patient_ids: List[UUID], account_id: UUID
) -> int:
    """Tell a newly-linked patient about documents waiting for them.

    A document issued while the patient had no account was kept, and appears
    in their inbox, but no notification was published - `notify_document_issued`
    skips a row with no `user_id`. This closes that gap at the point the
    account appears, which is the only moment the information exists.

    Notifications are addressed by ACCOUNT id, never by patient row id.
    `link_for_account` hands back the row ids it linked, so passing those
    straight through as the recipient produces a notification filed under an
    identifier no session ever carries - published successfully, readable by
    nobody. The bell stays at zero and there is nothing in the log to show
    for it.

    Three rules, all about not being a nuisance:

      - one summary per account, not one per document. Three results waiting
        should read as "3 documents are waiting", not three separate things
        each demanding attention.
      - only documents actually issued to the patient, so nothing private to
        the clinic is announced.
      - documents are grouped per account, so a person linked to several
        doctors gets one summary covering all of them.

    Never raises. A failure here must not cost someone their sign-in.
    """
    if not patient_ids:
        return 0

    from app.db.documents import Document
    from app.db.models import Patient
    from app.services.notifications import get_bus

    try:
        rows = (
            await session.execute(
                select(Document).where(
                    Document.patient_id.in_(patient_ids),
                    Document.issued_to_patient.is_(True),
                    Document.superseded_by_id.is_(None),
                )
            )
        ).scalars().all()

        if not rows:
            return 0

        # Narrow to the rows actually owned by this account. The caller
        # passes ids from the linker, but re-checking means a wrong id
        # produces a quiet no-op rather than a stranger's summary.
        owned = {
            row_id
            for row_id in (
                await session.execute(
                    select(Patient.id).where(
                        Patient.id.in_(patient_ids),
                        Patient.user_id == account_id,
                    )
                )
            ).scalars().all()
        }
        rows = [r for r in rows if r.patient_id in owned]
        if not rows:
            return 0

        oldest = min(r.created_at for r in rows)
        newest = max(r.created_at for r in rows)
        kinds = sorted({r.kind.replace("_", " ") for r in rows})

        when = (
            oldest.date().isoformat()
            if oldest.date() == newest.date()
            else f"{oldest.date().isoformat()} to {newest.date().isoformat()}"
        )

        await get_bus().publish(
            recipient_account_id=str(account_id),
            kind="documents_waiting",
            title=f"{len(rows)} document{'s' if len(rows) != 1 else ''} waiting for you",
            body=f"{', '.join(kinds)} issued {when}",
            payload={
                "count": len(rows),
                "kinds": kinds,
                "document_ids": [str(r.id) for r in rows],
                "backfilled": True,
            },
        )
        logger.info(
            "announced %d backdated document(s) to account %s",
            len(rows),
            account_id,
        )
        return len(rows)
    except Exception:  # noqa: BLE001
        logger.warning("backdated document announcement failed", exc_info=True)
        return 0


@router.post("/login", response_model=SessionOut)
async def login(body: LoginRequest, session: AsyncSession = Depends(get_session)):
    email = normalise_email(body.email)
    if not email:
        raise HTTPException(status_code=422, detail="invalid email")

    account = await _account_by_email(session, email)
    # One message for both cases: distinguishing them confirms whether an
    # address is registered.
    if account is None or not verify_password(body.password, account.password_hash):
        raise HTTPException(status_code=401, detail="invalid email or password")

    if not account.is_active:
        raise HTTPException(status_code=403, detail="account is disabled")

    account.last_login_at = datetime.now(timezone.utc)

    # Heal any records that should have linked but did not, e.g. a record
    # added while the patient already had an account.
    linker = IdentityLinker(session)
    healed = await linker.link_for_account(account.id, email)
    await session.flush()

    # Same reasoning as on registration: documents written while the record
    # was unlinked should be announced the moment it becomes linked, however
    # that happened. `healed.linked` is empty on an ordinary repeat sign-in,
    # so this does not re-announce anything.
    if healed.linked:
        await _announce_backdated_documents(session, healed.linked, account.id)

    return await _session_payload(session, account, linked=len(healed.linked))


@router.get("/me", response_model=SessionOut)
async def me(request: Request, session: AsyncSession = Depends(get_session)):
    account = await _account_from_request(request, session)
    return await _session_payload(session, account)


# ── current-account resolution ─────────────────────────────────────


def bearer_from(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401, detail="missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    return header[7:].strip()


async def _account_from_token(
    token: str, session: AsyncSession
) -> Account:
    try:
        claims = _signer.verify(token)
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    try:
        account_id = UUID(claims["sub"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="malformed token") from exc

    account = await session.get(Account, account_id)
    if account is None or not account.is_active:
        raise HTTPException(status_code=401, detail="account not found or disabled")
    return account


async def _account_from_request(
    request: Request, session: AsyncSession
) -> Account:
    return await _account_from_token(bearer_from(request), session)


async def current_account(
    request: Request, session: AsyncSession = Depends(get_session)
) -> Account:
    """FastAPI dependency: the signed-in account."""
    return await _account_from_request(request, session)


async def current_doctor(
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
) -> Doctor:
    if account.role != "doctor":
        raise HTTPException(status_code=403, detail="doctor account required")
    doctor = (
        await session.execute(select(Doctor).where(Doctor.account_id == account.id))
    ).scalars().first()
    if doctor is None:
        raise HTTPException(status_code=403, detail="no doctor profile for this account")
    return doctor


async def current_patient(
    account: Account = Depends(current_account),
    session: AsyncSession = Depends(get_session),
) -> Patient:
    """The *primary* patient row for a signed-in patient.

    A patient with several doctors has several rows; the newest is the
    one their own portal opens on.
    """
    if account.role != "patient":
        raise HTTPException(status_code=403, detail="patient account required")
    rows = (
        await session.execute(
            select(Patient)
            .where(Patient.user_id == account.id)
            .order_by(Patient.created_at.desc())
        )
    ).scalars().all()
    if not rows:
        raise HTTPException(
            status_code=404, detail="no clinical record linked to this account yet"
        )
    return rows[0]


# ── profile ────────────────────────────────────────────────────────


class ProfileUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    phone: str | None = None
    photo_path: str | None = None
    photo_version: int | None = None


class DoctorProfileUpdate(ProfileUpdate):
    specialty: str | None = None
    clinic_name: str | None = None
    nmc_number: str | None = None
    registration_state: str | None = None
    consult_fee: int | None = Field(default=None, ge=0)
    signature_path: str | None = None


@router.patch("/me/profile", response_model=AccountOut)
async def update_profile(
    body: ProfileUpdate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    account = await _account_from_request(request, session)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(account, field, value)
    await session.flush()
    await session.refresh(account)
    return AccountOut.model_validate(account)


#: Formats a browser will actually produce from a file input or a phone
#: camera. Anything else is refused rather than sniffed, because the stored
#: object is served back with this content type and a mislabelled one would
#: render as arbitrary markup.
PHOTO_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

#: A profile picture is displayed at most a few hundred pixels across.
#: Capping at 4 MB stops a phone's 12 MP original from becoming a
#: multi-megabyte object that every inbox request would re-fetch.
MAX_PHOTO_BYTES = 4 * 1024 * 1024


@router.post("/me/photo")
async def upload_photo(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Store a profile picture and return the account.

    The version is bumped on every upload so a cached URL stops being
    served. Without that, a doctor who changes their picture keeps seeing
    the old one until the CDN cache expires, which looks like the upload
    having failed.
    """
    account = await _account_from_request(request, session)

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in PHOTO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported image type {content_type or 'unknown'}; "
                f"expected one of {', '.join(sorted(PHOTO_TYPES))}"
            ),
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_PHOTO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image is {len(raw) // 1024} KB; the limit is "
            f"{MAX_PHOTO_BYTES // 1024} KB",
        )

    suffix = PHOTO_TYPES[content_type]
    filename = f"{account.id}{suffix}"
    stored = await get_storage().put(
        raw, f"avatars/{account.id}", filename, content_type=content_type
    )
    account.photo_path = stored.key
    account.photo_version = (account.photo_version or 1) + 1

    # Propagate to every patient row this account owns.
    #
    # A patient who has seen three doctors has three `Patient` rows, all
    # carrying `user_id` pointing here. A doctor reads the *patient* row,
    # not the account, so writing only the account left the picture
    # invisible to every clinician the patient has ever seen - the upload
    # silently did nothing from the only side that matters.
    #
    # One upload, one identity, all the doctors. This is the same linkage
    # the email match established; the photo just has to follow it.
    owned = (
        await session.execute(select(Patient).where(Patient.user_id == account.id))
    ).scalars().all()
    for row in owned:
        row.photo_path = stored.key

    await session.flush()
    await session.refresh(account)
    return AccountOut.model_validate(account)


@router.patch("/me/doctor", response_model=DoctorOut)
async def update_doctor_profile(
    body: DoctorProfileUpdate,
    doctor: Doctor = Depends(current_doctor),
    session: AsyncSession = Depends(get_session),
):
    fields = body.model_dump(exclude_none=True)
    for key in ("specialty", "clinic_name", "nmc_number", "registration_state",
                "consult_fee", "signature_path"):
        if key in fields:
            setattr(doctor, key, fields.pop(key))
    for key, value in fields.items():
        setattr(doctor.account, key, value)
    await session.flush()
    await session.refresh(doctor)
    return DoctorOut.model_validate(doctor)
