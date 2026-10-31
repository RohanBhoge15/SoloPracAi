"""Pydantic schemas for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── patients ─────────────────────────────────────────────────────────


class PatientCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    age: Optional[int] = Field(default=None, ge=0, le=130)
    phone: Optional[str] = None
    #: Optional for a walk-in. When present it becomes the join key for
    #: automatic identity linking if the patient later registers.
    email: Optional[str] = None
    address: Optional[str] = None
    abha_number: Optional[str] = None


class PatientOut(BaseModel):
    id: UUID
    display_name: str
    age: Optional[int]
    phone: Optional[str]
    email: Optional[str]
    created_at: datetime
    user_id: Optional[UUID] = None
    doctor_id: Optional[UUID] = None
    origin: Optional[str] = None
    identity_verified: bool = False
    #: The patient's picture, mirrored onto this row when they upload one.
    #: A doctor reads the patient row, not the account, so without this a
    #: patient's photo is invisible to every clinician who has seen them.
    photo_path: Optional[str] = None

    model_config = {"from_attributes": True}


# ── events ───────────────────────────────────────────────────────────


class EventCreate(BaseModel):
    event_type: str = Field(pattern="^(lab|diagnosis|medication|vital|note|imaging)$")
    valid_time: datetime
    payload: Dict[str, Any] = Field(default_factory=dict)
    source_type: str = Field(default="manual", pattern="^(manual|ocr|ai_suggestion|correction)$")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    supersedes_id: Optional[UUID] = None
    document_id: Optional[UUID] = None


class EventOut(BaseModel):
    id: UUID
    event_type: str
    source_type: str
    valid_time: datetime
    recorded_at: datetime
    payload: Dict[str, Any]
    confidence: Optional[float]
    supersedes_id: Optional[UUID]
    content_hash: str
    is_superseded: bool = False
    validity: str = "current"


class SupersededPair(BaseModel):
    superseded: EventOut
    correction: EventOut


class TimelineOut(BaseModel):
    patient_id: UUID
    current: List[EventOut]
    history: List[EventOut]
    pairs: List[SupersededPair]
    stats: Dict[str, int]


class AsOfOut(BaseModel):
    patient_id: UUID
    instant: datetime
    events: List[EventOut]


# ── retrieval and chat ───────────────────────────────────────────────


class CitationOut(BaseModel):
    event_id: str
    text: str
    valid_time: datetime
    #: Resolved against the queried instant, not the present. A Literal so a
    #: third value fails loudly at the boundary instead of silently rendering
    #: an unstyled chip in the frontend.
    validity: Literal["current", "superseded"]
    event_type: str


class SuppressedOut(BaseModel):
    text: str
    reason: str


class ChatRequest(BaseModel):
    patient_id: UUID
    query: str = Field(min_length=1, max_length=2000)
    as_of: Optional[datetime] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[CitationOut]
    suppressed: List[SuppressedOut] = Field(default_factory=list)
    dropped: List[SuppressedOut] = Field(default_factory=list)
    intent: str
    intent_confidence: float
    degraded: bool = False
    latency_ms: float
    model: str
    has_stale_assertion: bool = False


# ── capture ──────────────────────────────────────────────────────────


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    ocr_confidence: Optional[float]
    committed: bool
    review_reason: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class CaptureResult(BaseModel):
    document: DocumentOut
    extracted: Dict[str, Any]
    confidence: float
    committed: bool
    reason: Optional[str] = None
    coverage_note: str = Field(
        default="Below-threshold extractions are not written to the record."
    )
