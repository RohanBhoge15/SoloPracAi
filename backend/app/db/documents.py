"""Documents: everything a doctor produces or a patient uploads.

Six kinds, one table, because they share a lifecycle: generated, stored
in object storage, optionally issued to the patient, and retained as an
immutable version of the record.

  prescription  a medication order with items, duration, notes
  certificate   a medical certificate or fitness note
  invoice       a bill for consultation or services
  weekly_report an LLM-drafted clinical summary
  scan          an uploaded image (lab report, X-ray, letter)
  reference     anything else the doctor attaches

Two rules that matter more than the schema:

  1. A generated document is written to the event log as a `document`
     event. The PDF is a rendering of a versioned fact, not the fact
     itself - so a corrected prescription is a correction, and the old
     PDF is retained rather than silently replaced.
  2. Nothing is ever overwritten. `superseded_by` marks a document that a
     later one replaced; the bytes stay in object storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_patient_created", "patient_id", "created_at"),
        Index("ix_documents_doctor", "doctor_id"),
        Index("ix_documents_kind", "kind"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE")
    )
    doctor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctors.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    kind: Mapped[str] = mapped_column(String(24))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Human-written description of a study. For an indexed image this is
    #: the text embedded alongside the pixels, which is what makes
    #: cross-modal search work.
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Object storage key. NULL for a document whose bytes are the OCR
    #: text rather than a file (an inline scan record).
    object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: The structured source of the rendered PDF: prescription items,
    #: invoice lines, certificate body, and so on. Kept as JSON so the
    #: document is useful even if the PDF is lost.
    content: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    #: The clinical payload, e.g. {"rows": [...]} for a scan.
    extracted: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # ── capture / OCR fields ─────────────────────────────────────────
    # A `scan` document is an upload rather than a generated PDF, and it
    # carries the abstention decision with it: if confidence fell below
    # threshold, `committed` is False and no clinical event was created.
    filename: Mapped[str | None] = mapped_column(String(300), nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    committed: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Set when the doctor sends it to the patient. Until then the
    #: document is private to the clinic.
    issued_to_patient: Mapped[bool] = mapped_column(Boolean, default=False)
    issued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    read_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    #: A document this one replaced. Set rather than mutating the old row.
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    #: The event this document was derived from, so a report can cite the
    #: records it summarises.
    source_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    #: Monotonic per document, bumped on regeneration. Paired with the
    #: object key it guarantees a cache-busting URL changes.
    version: Mapped[int] = mapped_column(Integer, default=1)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )

    patient: Mapped["object | None"] = relationship("Patient", lazy="selectin")
    doctor: Mapped["object | None"] = relationship("Doctor", lazy="selectin")

    @property
    def is_current(self) -> bool:
        return self.superseded_by_id is None


# ── typed views over `content` ──────────────────────────────────────
# The JSON stays flexible so a new document kind needs no migration, but
# the shapes below are what the templates and the API rely on.


class PrescriptionItem(dict):
    """One line of a prescription."""

    @classmethod
    def make(
        cls,
        drug: str,
        dose: str = "",
        frequency: str = "",
        duration: str = "",
        route: str = "",
        instruction: str = "",
    ) -> "PrescriptionItem":
        return cls(
            drug=drug.strip(),
            dose=dose.strip(),
            frequency=frequency.strip(),
            duration=duration.strip(),
            route=route.strip(),
            instruction=instruction.strip(),
        )

    @property
    def display(self) -> str:
        parts = [self.get("drug", "")]
        for key in ("dose", "frequency", "route", "duration"):
            value = self.get(key)
            if value:
                parts.append(str(value))
        return " - ".join(p for p in parts if p)


class InvoiceLine(dict):
    @classmethod
    def make(
        cls, description: str, quantity: int = 1, unit_price: int = 0
    ) -> "InvoiceLine":
        return cls(
            description=description.strip(),
            quantity=max(1, int(quantity)),
            unit_price=max(0, int(unit_price)),
            amount=max(0, int(quantity)) * max(0, int(unit_price)),
        )
