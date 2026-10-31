"""Database schema — three-valued event store.

The `events` table plus the `event_validity` view are the whole model.
Validity is *derived*; it is never stored as a mutable column.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


event_type_enum = SAEnum(
    "lab", "diagnosis", "medication", "vital", "note", "imaging",
    name="event_type", native_enum=True,
)
source_type_enum = SAEnum(
    "manual", "ocr", "ai_suggestion", "correction",
    name="source_type", native_enum=True,
)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Patient(Base):
    """A person under a doctor's care.

    Two independent axes, easy to conflate:

      - `doctor_id` scopes the *record*. Each doctor keeps their own copy.
      - `user_id` is the *identity*. One person, one account, many
        doctors.

    A walk-in the doctor created has `user_id = NULL`. When that person
    later registers with the same email, the linker fills it in and the
    rows become one identity without any record being copied.
    """

    __tablename__ = "patients"
    # The email index and the (doctor, account) unique index are created
    # after the class body in migrate_identity.py, because a functional
    # index needs the column to exist first.
    __table_args__ = (
        Index(
            "ix_patients_doctor_account",
            "doctor_id",
            "user_id",
            unique=True,
            postgresql_where="user_id IS NOT NULL",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )

    #: The account this patient belongs to, once they have registered.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="SET NULL"),
        nullable=True,
    )
    #: The doctor who owns this copy of the record.
    doctor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("doctors.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    display_name: Mapped[str] = mapped_column(String(200))
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    #: The join key for identity linking. Optional - walk-ins are real.
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    abha_number: Mapped[str | None] = mapped_column(String(14), nullable=True)
    photo_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: doctor_created | patient_self | linked_clone
    origin: Mapped[str] = mapped_column(String(24), default="doctor_created")
    #: True once an exact email match has confirmed the identity.
    identity_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )

    events: Mapped[list["Event"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", lazy="selectin"
    )

    #: The login this patient belongs to, if they have one. Optional and
    #: deliberately not cascade: deleting an account must never destroy
    #: clinical records, which are medico-legal.
    account: Mapped["object | None"] = relationship(
        "Account", foreign_keys=[user_id], viewonly=True, lazy="selectin"
    )

    @property
    def is_registered(self) -> bool:
        return self.user_id is not None


class Event(Base):
    """An immutable clinical fact. Append-only: never updated, never deleted.

    A correction does not modify its target. It appends a new row whose
    `supersedes_id` points at the old one. That is what makes the record
    time-travellable and the validity derivable.
    """

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_patient_valid_time", "patient_id", "valid_time"),
        Index("ix_events_supersedes", "supersedes_id"),
        Index("ix_events_payload_gin", "payload", postgresql_using="gin"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)"),
        CheckConstraint(
            "(source_type = 'correction') = (supersedes_id IS NOT NULL)",
            name="ck_correction_requires_target",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )

    event_type: Mapped[str] = mapped_column(event_type_enum)
    source_type: Mapped[str] = mapped_column(
        source_type_enum, default="manual", server_default="manual"
    )

    #: When the fact was clinically true.
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    #: When we wrote it. Distinct from valid_time, and that distinction is
    #: the whole point of a bitemporal record.
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )

    payload: Mapped[dict] = mapped_column(JSONB, default=dict)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.id", ondelete="CASCADE"), nullable=True
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    content_hash: Mapped[str] = mapped_column(String(64))

    patient: Mapped["Patient"] = relationship(back_populates="events")

    # ── conversion to and from the pure core type ────────────────────

    def to_core(self):
        from app.core.validity import Event as CoreEvent, EventType, SourceType

        return CoreEvent(
            event_id=str(self.id),
            patient_id=str(self.patient_id),
            event_type=EventType(self.event_type),
            valid_time=self.valid_time,
            payload=self.payload or {},
            source_type=SourceType(self.source_type),
            confidence=self.confidence,
            supersedes_id=str(self.supersedes_id) if self.supersedes_id else None,
            recorded_at=self.recorded_at,
            content_hash=self.content_hash,
        )

    @classmethod
    def from_core(cls, core) -> "Event":
        """Build a row from a core event.

        A new record has no id yet, so the core type uses "" as a
        placeholder. That empty string must not reach the UUID column -
        let the default mint one instead.
        """
        return cls(
            id=core.event_id or _uuid(),
            patient_id=core.patient_id,
            event_type=str(core.event_type),
            source_type=str(core.source_type),
            valid_time=core.valid_time,
            recorded_at=core.recorded_at,
            payload=core.payload,
            confidence=core.confidence,
            supersedes_id=core.supersedes_id or None,
            content_hash=core.content_hash or "",
        )


# ── derived validity ─────────────────────────────────────────────────
# Mirrors SupersessionGraph.state() in SQL so the timeline query is a single
# round trip. The view is the single source of truth in production; the
# Python graph remains the reference implementation and the test oracle.

VALIDITY_VIEW_SQL = """
CREATE OR REPLACE VIEW event_validity AS
SELECT
    e.*,
    EXISTS (
        SELECT 1 FROM events c
        WHERE c.supersedes_id = e.id
          AND c.source_type = 'correction'
    ) AS is_superseded
FROM events e;
"""


AS_OF_VIEW_SQL = """
-- Cannot be a VIEW: the as-of instant is a parameter, and CREATE VIEW
-- does not accept bind parameters. Exposed as a function instead.
--
-- Resolves supersession relative to the *requested* instant, not to now.
-- A record corrected in March is CURRENT as of February and SUPERSEDED now.
-- This is the property a recency-decay baseline cannot replicate.
CREATE OR REPLACE FUNCTION events_as_of(
    p_patient_id UUID,
    p_instant TIMESTAMPTZ
)
RETURNS SETOF events
LANGUAGE sql
STABLE
AS $$
    SELECT e.*
    FROM events e
    WHERE e.patient_id = p_patient_id
      AND e.valid_time <= p_instant
      AND NOT EXISTS (
          SELECT 1 FROM events c
          WHERE c.supersedes_id = e.id
            AND c.source_type = 'correction'
            AND c.valid_time <= p_instant
      )
    ORDER BY e.valid_time DESC;
$$;
"""

SUPERSESSION_FK_SQL = (
    # asyncpg refuses multiple commands in a single prepared statement, so
    # these are executed one at a time.
    "ALTER TABLE events DROP CONSTRAINT IF EXISTS events_supersedes_id_fkey",
    # A correction points at the record it replaces. CASCADE rather than
    # RESTRICT: deleting a patient cascades to their events, and RESTRICT on
    # a self-reference makes that bulk delete fail on the first correction
    # encountered. There is no single-event delete endpoint, so the only path
    # that fires this is a patient deletion, where losing the chain is
    # correct.
    "ALTER TABLE events ADD CONSTRAINT events_supersedes_id_fkey "
    "FOREIGN KEY (supersedes_id) REFERENCES events(id) ON DELETE CASCADE",
)
