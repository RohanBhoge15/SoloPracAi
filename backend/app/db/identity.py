"""Identity schema: accounts, doctors, and patient linking.

The design problem this solves:

    A doctor registers a walk-in patient today, with an email. The patient
    creates their own account next year, with the same email. They must
    become one identity, across every doctor who has seen them - without
    any doctor having to do anything.

Email is the join key, so it has to be:
      - required for a doctor account
      - optional for a doctor-created patient (walk-ins are real)
      - unique across accounts
      - normalised on write, or `A@x.com` and `a@x.com` become two people

`patients.user_id` is the link. NULL means "a doctor knows this person
but they have never logged in."
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

#: Shared declarative base. Must come from models.py, not be redefined:
## two Base objects means two metadata registries and create_all would
#: only ever see half the schema.
from app.db.models import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Account(Base):
    """A login. Either a doctor or a patient; `role` says which.

    One table rather than two because the token, the password, and the
    session logic are identical, and the failure mode of two tables is
    duplicated auth logic.
    """

    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20))  # doctor | patient

    display_name: Mapped[str] = mapped_column(String(200))
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)

    #: Relative path in object storage, e.g. "avatars/<uuid>.png".
    photo_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    photo_version: Mapped[int] = mapped_column(Integer, default=1)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )

    doctor: Mapped["Doctor | None"] = relationship(
        back_populates="account", cascade="all, delete-orphan", uselist=False
    )
    patient: Mapped["Patient | None"] = relationship(
        back_populates="account", cascade="all, delete-orphan", uselist=False
    )


class Doctor(Base):
    """Clinical profile. One per doctor account."""

    __tablename__ = "doctors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        unique=True,
    )

    specialty: Mapped[str | None] = mapped_column(String(120), nullable=True)
    clinic_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    nmc_number: Mapped[str | None] = mapped_column(String(40), nullable=True)
    registration_state: Mapped[str | None] = mapped_column(String(60), nullable=True)

    #: Consultancy fee in whole rupees. Kept out of the account table
    #: because it is clinical metadata, not identity.
    consult_fee: Mapped[int | None] = mapped_column(Integer, nullable=True)

    signature_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )

    account: Mapped["Account"] = relationship(back_populates="doctor")

    @property
    def display_name(self) -> str:
        return self.account.display_name

    @property
    def email(self) -> str:
        return self.account.email


class IdentityLinkLog(Base):
    """Audit of every automatic link.

    Linking by email is convenient and slightly risky: a shared family
    address, or a typo. Recording what was linked and why means a wrong
    link is traceable and reversible rather than silent.
    """

    __tablename__ = "identity_link_log"
    __table_args__ = (Index("ix_link_log_account", "account_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="CASCADE")
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE")
    )
    email: Mapped[str] = mapped_column(String(320))
    method: Mapped[str] = mapped_column(String(40), default="email_exact")
    #: Confidence in the match, 0-1. Exact normalised email is 1.0.
    confidence: Mapped[float] = mapped_column(default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )


# ── derived / helper SQL ────────────────────────────────────────────

ACCOUNT_SEARCH_SQL = """
-- A patient's inbox: everything ever addressed to one account, across
-- every doctor, newest first.
CREATE OR REPLACE FUNCTION patient_inbox(p_account_id UUID)
RETURNS SETOF documents
LANGUAGE sql
STABLE
AS $$
    SELECT d.*
    FROM documents d
    JOIN patients p ON p.id = d.patient_id
    WHERE p.user_id = p_account_id
      AND d.kind = 'prescription'
    ORDER BY d.created_at DESC;
$$;
"""
