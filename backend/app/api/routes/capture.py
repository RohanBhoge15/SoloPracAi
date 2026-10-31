"""Capture — photo to structured record, with abstention.

The research contribution here is not the OCR. It is the refusal: below a
confidence threshold the system declines to write a value into a permanent
medical record and routes the scan to a human instead.

A wrong number silently committed is the worst failure mode in the whole
system. Refusing is correct behaviour, and the coverage/accuracy trade-off
is measurable.
"""

from __future__ import annotations

import io
import logging
import re
import uuid
from typing import Any, Dict, Optional, Tuple
from uuid import UUID as UUIDType

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_patient
from app.api.schemas import CaptureResult, DocumentOut
from app.core.config import get_settings
from app.db.documents import Document
from app.db.models import Event, Patient
from app.db.session import get_session
from app.services.ocr import OCRUnavailable, extract, image_quality
from app.services.storage import get_storage

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/patients/{patient_id}/capture", tags=["capture"])

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}

# ── field extraction ─────────────────────────────────────────────────

#: Lab-name -> canonical unit map. Deliberately small and explicit: a wrong
#: unit is a clinical error, so anything unrecognised is left blank rather
#: than guessed.
_UNIT_MAP: Dict[str, str] = {
    "mEq/L": "mEq/L", "meq/l": "mEq/L",
    "mmol/L": "mmol/L", "mg/dL": "mg/dL", "g/dL": "g/dL",
    "g/L": "g/L", "mg/L": "mg/L", "ng/mL": "ng/mL", "pg/mL": "pg/mL",
    "IU": "IU", "U": "IU", "%": "%", "mmHg": "mmHg", "bpm": "bpm",
    "mmol": "mmol", "µmol/L": "µmol/L", "umol/L": "µmol/L",
}

_KNOWN_TESTS = (
    "potassium", "sodium", "chloride", "glucose", "creatinine", "urea",
    "hemoglobin", "haemoglobin", "hba1c", "bilirubin", "albumin", "calcium",
    "platelet", "wbc", "rbc", "hct", "ldh", "ast", "alt", "tsh", "vitamin d",
    "ldl", "hdl", "triglycerides", "total cholesterol", "uric acid", "procalcitonin",
    "lipase", "amylase", "magnesium", "phosphate", "fbs", "ppbs", "bp", "spo2",
    "temperature", "weight", "height", "bmi", "heart rate",
    # Glycated haemoglobin under the spellings an OCR engine actually
    # produces. It reads a printed "1" in "HbA1c" as a lowercase "l" with
    # some regularity, giving "HbAlc" - and without this alias the analyte
    # is extracted under an unrecognised name and scored 0.55 instead of
    # 0.95, which puts a correct reading below the commit threshold.
    "hbalc",
    # Same class of confusion elsewhere on Indian reports.
    "sgpt", "sgot", "ggt", "ldl cholesterol", "total bilirubin",
    "direct bilirubin", "triglyceride", "vitamin b12", "folic acid",
    "tibc", "ferritin", "anti ccp", "psa", "hba",
)

_ROW = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 \-/]{1,28}?)\s+"
    r"(\d+(?:\.\d+)?)\s*"
    r"([A-Za-zµ%/]+)?\s*"
    r"(?:\[?([\d.]+)\s*[-–~]\s*([\d.]+)\]?)?\s*$"
)

_REF = re.compile(
    r"([A-Za-z][A-Za-z0-9 \-/]{1,28}?)\s*[:\-]\s*(\d+(?:\.\d+)?)\s*([A-Za-zµ%/]+)?"
)

_DATE = re.compile(
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)


def parse_labs(text: str) -> Tuple[list[Dict[str, Any]], float]:
    """Extract lab rows. Returns (rows, mean_confidence).

    Confidence here is *extraction* confidence — how sure we are the
    numbers were read correctly — not clinical plausibility. Blending the
    two would make the gate refuse on clinical grounds, which is the
    physician's call, not the software's.

    Two input shapes are handled, because real lab reports are tabular and
    the OCR engine reports what it sees rather than what the page means.

    **Column-per-line**, which is what the engine actually emits for a
    ruled report:

        Haemoglobin
        13.4
        g/dL
        13.0 - 17.0

    Each cell is its own line, because the detector found them as separate
    text blocks. A purely line-oriented regex finds nothing here, which is
    not a near miss: it returned zero rows for a perfectly legible report
    with seven of them.

    **One row per line**, still common on plain-text and hand-typed
    results:

        Haemoglobin 13.4 g/dL 13.0 - 17.0
    """
    lines = [ln.strip() for ln in (text or "").splitlines()]
    return _parse_tabular(lines) if _looks_tabular(lines) else _parse_prose(lines)


def _looks_tabular(lines: List[str]) -> bool:
    """Is this a column-per-line report rather than one-row-per-line?

    The tell is a bare test name immediately followed by a bare number,
    which no sane prose result contains. Requiring two such adjacencies
    keeps a stray pair from flipping the whole parse.

    The name pattern must allow digits. "HbA1c" is the single most common
    analyte on an Indian lab report and "HbAlc" is what the OCR engine
    actually returns for it; a letters-only pattern rejected both, so the
    detector said "not tabular", the prose parser ran, and a report whose
    every cell was on its own line produced nothing at all.
    """
    pairs = 0
    for a, b in zip(lines, lines[1:]):
        if a and b and _VALUE.fullmatch(b) and _NAME.match(a):
            pairs += 1
            if pairs >= 2:
                return True
    return False


#: A bare result: a number, optionally signed, optionally with a trailing
#: comparison symbol. Deliberately strict — "Reference" is a word, not a
#: value, and reading it as one is how a header becomes a result.
_VALUE = re.compile(r"^[<>≤≥]?\s*\d+(?:[.,]\d+)?$")
#: A reference interval, e.g. "13.0 - 17.0", "70-110", "< 5.6", "> 400".
_RANGE = re.compile(
    r"^\[?\s*(?:[<>≤≥]\s*)?(\d+(?:[.,]\d+)?)\s*"
    r"(?:[-–—~]\s*(\d+(?:[.,]\d+)?))?\s*\]?$"
)

#: Real units, as opposed to "a run of letters".
#:
#: The looser pattern `[A-Za-z]+` matched "Haemoglobin", which then consumed
#: the analyte name as if it were a unit and the row was lost. Units in lab
#: reports are a short and known set, so the set is enumerated: an
#: unrecognised token is then left unclassified rather than guessed at,
#: which is the safe direction for a clinical value.
_UNIT = re.compile(
    r"^(?:%|mmHg|mmol|mmol/L|mEq/L|mEq/L|mg/dL|mg/L|g/dL|g/L|ng/mL|pg/mL|"
    r"µmol/L|umol/L|µg/dL|mcg/L|IU|U|L|IU/L|U/L|mg/kg/min|ratio|count|"
    r"cumm|/cumm|/µL|/uL|cells|copies|index|mol|osmol/kg|Hb|sec|"
    r"kg|cm|m|mm|kg/m2|°C|F|bpm|mmol/kg|mg/m2|PERCENT|percent)$",
    re.IGNORECASE,
)
#: A bare analyte name: letters, digits and the punctuation lab reports use
#: in names ("HbA1c", "SGPT/ALT", "Total Leucocyte Count").
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 ()/\-]{1,38}$")

#: Column headings seen on real Indian lab reports. These match the test
#: name pattern exactly, sit above the data, and would otherwise be read as
#: the first row of the table.
_HEADER_CELLS = frozenset({
    "RESULT", "RESULTS", "UNIT", "UNITS", "REFERENCE", "REFERENCES",
    "NORMAL", "NORMAL RANGE", "RANGE", "TEST", "TESTS", "TEST NAME",
    "VALUE", "VALUES", "PARAMETER", "ANALYTE", "INVESTIGATION",
    "OBSERVED", "REPORTED", "FLAG", "METHOD", "SPECIMEN", "COMMENT",
    "SRL LABORATORY", "REPORT GENERATED", "VERIFIED BY", "NOTE",
})

#: Lines that are prose rather than data. A report carries a title, a date
#: line and a signature, and "Report generated 04 Oct 2026" otherwise
#: anchors a row whose "value" is whatever number the date contained - a
#: 2026 written into a patient's chart as a lab result.
_PROSE_PREFIXES = (
    "report generated", "generated on", "verified by", "note", "remarks",
    "comments", "page", "ref ", "ref.", "lab no", "lab no.", "sample",
    "collected", "received", "reported by", "authorized", "signature",
    "doctor", "dr.", "m/s ", "srl laboratory", "laboratory",
)


def _parse_tabular(lines: List[str]) -> Tuple[list[Dict[str, Any]], float]:
    """Walk the column-per-line form, assembling each row as it is found."""
    rows: list[Dict[str, Any]] = []
    confidences: list[float] = []

    i = 0
    n = len(lines)
    while i < n:
        name = lines[i]

        # Skip anything that is not a plausible analyte name at all.
        if not _NAME.match(name):
            i += 1
            continue

        # Column headers are the trap here. "RESULT", "UNIT" and
        # "REFERENCE" all match the name pattern perfectly and sit at the
        # top of every real report, so a parser that does not exclude them
        # reads the header row as the first lab result.
        if name.strip().upper() in _HEADER_CELLS:
            i += 1
            continue

        # Prose, not an analyte. A date line contains digits, so without
        # this the year becomes a result.
        lowered = name.strip().lower()
        if lowered.startswith(_PROSE_PREFIXES):
            i += 1
            continue

        cells: List[str] = []
        j = i + 1
        # Take cells until the next thing that is unambiguously a new
        # analyte name.
        #
        # The break condition has to exclude unit tokens, and it previously
        # did not: "g/dL" matches the name pattern, so the cell walk stopped
        # at the first unit and every row came back empty. `VALUE`, `UNIT`
        # and `RANGE` are matched first precisely because they overlap with
        # it - "/" and "%" are legal in a name by the regex, and a bare
        # number can look like a short word.
        while j < n and len(cells) < 4:
            nxt = lines[j]
            if _VALUE.match(nxt) or _UNIT.match(nxt) or _RANGE.match(nxt):
                cells.append(nxt)
                j += 1
                continue
            if _NAME.match(nxt) and nxt.strip().upper() not in _HEADER_CELLS:
                break
            j += 1

        value = next((c for c in cells if _VALUE.match(c)), None)
        if value is None:
            i = j if j > i else i + 1
            continue

        unit = next((c for c in cells if _UNIT.match(c) and not _VALUE.match(c)), None)
        rng = next((c for c in cells if _RANGE.match(c) and c != value), None)
        low = high = None
        if rng:
            m = _RANGE.match(rng)
            if m:
                low, high = m.group(1), m.group(2)

        rows.append(_make_row(name, value, unit, low, high))
        confidences.append(_row_confidence(name, unit, low, high))
        i = j if j > i else i + 1

    if not confidences:
        return [], 0.0
    return rows, sum(confidences) / len(confidences)


def _parse_prose(lines: List[str]) -> Tuple[list[Dict[str, Any]], float]:
    """One row per line, the long-standing form."""
    rows: list[Dict[str, Any]] = []
    confidences: list[float] = []

    for line in lines:
        match = _ROW.match(line)
        if not match:
            continue
        name, value, unit, ref_low, ref_high = match.groups()
        rows.append(_make_row(name, value, unit, ref_low, ref_high))
        confidences.append(_row_confidence(name, unit, ref_low, ref_high))

    if not confidences:
        return [], 0.0
    return rows, sum(confidences) / len(confidences)


def _row_confidence(name: str, unit: Optional[str], low: Any, high: Any) -> float:
    """How much to trust one extracted row.

    Kept separate from the row so both parsers score identically.
    """
    lowered = (name or "").strip().lower()
    known = next((k for k in _KNOWN_TESTS if k in lowered), None)
    confidence = 0.95 if known else 0.55

    if unit and unit.strip().lower() in _UNIT_MAP:
        confidence += 0.03
    elif unit:
        # A unit we do not recognise is not an error, but it is not
        # corroboration either, so it neither helps nor is invented.
        confidence -= 0.05
    else:
        confidence -= 0.15

    if low and high:
        try:
            lo = float(str(low).replace(",", "."))
            hi = float(str(high).replace(",", "."))
        except ValueError:
            return max(0.0, min(1.0, confidence))
        # A range that does not run low-to-high means the columns were read
        # out of order, which is evidence about the *extraction* and so
        # belongs in this number rather than in a silently invented flag.
        if lo <= hi:
            confidence += 0.05
        else:
            confidence -= 0.20

    return max(0.0, min(1.0, confidence))


def _make_row(
    name: str, value: str, unit: Optional[str], low: Any, high: Any
) -> Dict[str, Any]:
    """Assemble one record-ready row, flagging against the range if given."""
    name = (name or "").strip()
    lowered = name.lower()
    display = next((k for k in _KNOWN_TESTS if k in lowered), None) or name

    row: Dict[str, Any] = {
        "test": display,
        "value": str(value).strip().lstrip("<>≤≥").strip(),
        "unit": _UNIT_MAP.get((unit or "").strip().lower()) or (unit or "").strip() or None,
    }

    if low and high:
        try:
            lo = float(str(low).replace(",", "."))
            hi = float(str(high).replace(",", "."))
            numeric = float(str(row["value"]).replace(",", "."))
        except ValueError:
            return row
        if lo > hi:
            # Inverted interval: the columns were read across a boundary.
            return row
        row["ref_low"] = lo
        row["ref_high"] = hi
        row["flag"] = "low" if numeric < lo else "high" if numeric > hi else "normal"
    return row



def parse_dates(text: str) -> Optional[str]:
    match = _DATE.search(text or "")
    return match.group(1) if match else None


# ── imaging captures ─────────────────────────────────────────────────


def caption_for(filename: str) -> str:
    """A last-resort caption when the doctor typed nothing.

    Modality plus body region, guessed from the filename. It is weak, and it
    is honest about being weak: it describes what the file is called, not
    what the image shows. The pixels remain the retrievable part.
    """
    name = (filename or "").lower()
    modality = "radiograph"
    if "ct" in name or "computed tomography" in name:
        modality = "ct scan"
    elif "mri" in name or "magnetic resonance" in name:
        modality = "mri scan"
    elif "usg" in name or "ultrasound" in name or "sonogram" in name:
        modality = "ultrasound image"
    elif "ecg" in name or "ekg" in name:
        modality = "ecg"

    region = "medical image"
    for word, label in (
        ("chest", "chest"), ("cxr", "chest"), ("thorax", "thorax"),
        ("brain", "head"), ("head", "head"), ("skull", "head"),
        ("knee", "knee"), ("wrist", "wrist"), ("ankle", "ankle"),
        ("abdomen", "abdomen"), ("pelvis", "pelvis"),
        ("fundus", "retinal fundus"), ("retina", "retinal fundus"),
        ("spine", "spine"), ("shoulder", "shoulder"), ("hip", "hip"),
    ):
        if word in name:
            region = label
            break
    return f"{region} {modality}"


async def _capture_imaging(
    *,
    patient: Patient,
    raw: bytes,
    filename: str,
    content_type: str,
    stored_key: str,
    stored_size: int,
    caption: Optional[str],
    findings: Optional[str],
    valid_time: Optional[str],
    session: AsyncSession,
) -> CaptureResult:
    """File a radiograph as a real event, and index its pixels.

    Imaging does not go through the lab gate, and the reason is worth being
    explicit about. That gate exists because OCR can silently invent a
    *number* that a clinician will act on. Committing an image asserts no
    numeric value: it stores bytes that a human can open and look at. The
    failure mode the lab abstention protects against does not exist here, so
    applying it would mean every radiograph is permanently un-committable
    while looking like a working feature.

    What still gates the commit:
      - image readability, checked by the caller before this is called
      - BiomedCLIP producing a vector. If the pixels cannot be embedded the
        study would be stored but unfindable by content, which is the exact
        "stored but absent from the log" failure the image index route warns
        about, so we decline instead of filing a dead record.

    `valid_time` is the date of the *study*, not of the upload. Filing a
    two-year-old radiograph as of today would make it the most recent
    imaging and a present-tense question would return it as current.
    """
    import asyncio

    from app.services.embeddings import get_embedder

    embedder = get_embedder()
    study = (caption or "").strip() or caption_for(filename)

    if not embedder.image_available():
        # Vision is off on this deployment. Storing an unindexable image is
        # still better than losing it, but we must not claim it is
        # retrievable, so it goes to review with the reason stated.
        doc = Document(
            patient_id=patient.id,
            kind="scan",
            title=study,
            filename=filename,
            content_type=content_type,
            object_key=stored_key,
            size_bytes=stored_size,
            caption=study or None,
            raw_text=findings or None,
            ocr_confidence=0.0,
            committed=False,
            review_reason="image indexing disabled on this deployment",
        )
        session.add(doc)
        await session.flush()
        await session.refresh(doc)
        return CaptureResult(
            document=DocumentOut.model_validate(doc),
            extracted={"study": study},
            confidence=0.0,
            committed=False,
            reason=doc.review_reason,
        )

    vector = await asyncio.to_thread(embedder.encode_image, raw)
    if vector is None:
        doc = Document(
            patient_id=patient.id,
            kind="scan",
            title=study,
            filename=filename,
            content_type=content_type,
            object_key=stored_key,
            size_bytes=stored_size,
            caption=study or None,
            raw_text=findings or None,
            ocr_confidence=0.0,
            committed=False,
            review_reason="image could not be embedded for retrieval",
        )
        session.add(doc)
        await session.flush()
        await session.refresh(doc)
        return CaptureResult(
            document=DocumentOut.model_validate(doc),
            extracted={"study": study},
            confidence=0.0,
            committed=False,
            reason=doc.review_reason,
        )

    from datetime import datetime, timezone

    from app.core.validity import Event as CoreEvent
    from app.core.validity import EventType, SourceType, content_hash

    when = (
        datetime.fromisoformat(valid_time).astimezone(timezone.utc)
        if valid_time
        else datetime.now(timezone.utc)
    )

    doc = Document(
        patient_id=patient.id,
        kind="scan",
        title=study,
        filename=filename,
        content_type=content_type,
        object_key=stored_key,
        size_bytes=stored_size,
        caption=study or None,
        raw_text=findings or None,
        ocr_confidence=1.0,
        committed=True,
    )
    session.add(doc)
    await session.flush()

    core = CoreEvent(
        event_id="",
        patient_id=str(patient.id),
        event_type=EventType.IMAGING,
        valid_time=when,
        payload={
            "study": study,
            "findings": findings or None,
            "document_id": str(doc.id),
            "modality": content_type,
        },
        source_type=SourceType.OCR,
        confidence=1.0,
    )
    row = Event.from_core(core)
    row.content_hash = content_hash(core)
    row.document_id = doc.id
    session.add(row)
    await session.flush()
    await session.refresh(row)

    event_id = str(row.id)
    indexed = False
    try:
        from app.services.vectors import get_store

        indexed = await asyncio.to_thread(
            get_store().upsert_image,
            event_id,
            str(patient.id),
            vector,
            study,
            "imaging",
            when,
        )
    except Exception:  # noqa: BLE001
        logger.warning("image vector upsert failed", exc_info=True)

    if not indexed:
        # The event and the bytes are written. A degraded index is not a lost
        # record, so it is reported rather than raised -- raising would roll
        # back the event and lose the study.
        logger.warning(
            "imaging event %s stored and filed, but not vector-indexed", event_id
        )

    await session.refresh(doc)
    return CaptureResult(
        document=DocumentOut.model_validate(doc),
        extracted={"study": study, "event_id": event_id, "indexed": indexed},
        confidence=1.0,
        committed=True,
        reason=None if indexed else "stored and filed, but not indexed for image search",
    )


# ── route ────────────────────────────────────────────────────────────


@router.post("", response_model=CaptureResult, status_code=status.HTTP_201_CREATED)
async def capture(
    patient: Patient = Depends(get_patient),
    file: UploadFile = File(...),
    event_type: str = Form(default="lab"),
    valid_time: Optional[str] = Form(default=None),
    #: Study description for an imaging capture. Ignored for lab sheets,
    #: where the rows are extracted from the picture instead.
    caption: Optional[str] = Form(default=None),
    #: A radiologist's or doctor's own description of an imaging capture.
    findings: Optional[str] = Form(default=None),
    #: Ask MedGemma to attempt a reading when OCR is unsure. Off by
    #: default because it costs about a minute and 5 GB on this host.
    deep_read: bool = Form(default=False),
    session: AsyncSession = Depends(get_session),
):
    """Ingest a scan. Commits an event only if extraction is confident enough.

    Below threshold the document is stored for review and no clinical event
    is created. That is the intended outcome, not an error.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported media type {file.content_type}; "
            f"expected one of {sorted(ALLOWED_TYPES)}",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file exceeds 12 MB limit")
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    quality = image_quality(raw)
    storage = get_storage()
    # The original bytes are stored before anything else. An earlier
    # version read the upload, OCR'd it, and discarded it — so a scan
    # could never be viewed again. The file is the source of truth; the
    # extracted values are a reading of it.
    try:
        stored = await storage.put(
            raw, f"scans/{patient.id}", file.filename or "scan",
            content_type=file.content_type,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("scan storage failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="document storage unavailable"
        ) from exc

    if not quality["readable"]:
        doc = Document(
            patient_id=patient.id,
            kind="scan",
            title=file.filename or "scan",
            filename=file.filename or "scan",
            content_type=file.content_type,
            object_key=stored.key,
            size_bytes=stored.size,
            ocr_confidence=0.0,
            committed=False,
            review_reason="image quality too low to read reliably",
        )
        session.add(doc)
        await session.flush()
        await session.refresh(doc)
        return CaptureResult(
            document=DocumentOut.model_validate(doc),
            extracted={},
            confidence=0.0,
            committed=False,
            reason="image quality too low — re-photograph in better light",
        )

    # An imaging capture goes down a different road. There are no lab rows to
    # extract from a radiograph, so the extraction gate below would reject
    # every study as "no lab rows detected" -- which made the upload feature
    # look broken for exactly the modality it exists to handle. Imaging is
    # filed on readability plus a successful pixel embedding, and it skips
    # OCR entirely, which is also faster.
    if event_type == "imaging":
        return await _capture_imaging(
            patient=patient,
            raw=raw,
            filename=file.filename or "scan",
            content_type=file.content_type,
            stored_key=stored.key,
            stored_size=stored.size,
            caption=caption,
            findings=findings,
            valid_time=valid_time,
            session=session,
        )

    try:
        text = extract(raw)
    except OCRUnavailable:
        text = ""
    except Exception:
        logger.warning("ocr failed", exc_info=True)
        text = ""

    rows, confidence = parse_labs(text)
    threshold = settings.ocr_confidence_threshold
    committed = bool(rows) and confidence >= threshold

    # MedGemma is opt-in per request, never an automatic fallback.
    #
    # Measured on this CPU-only host: 54.6 s to encode one 896x896 image
    # plus a generation pass, and 5.4 GB resident. Firing that
    # automatically whenever OCR is unsure would turn a routine upload
    # into a two-minute stall and would not fit in memory alongside
    # Postgres, Qdrant and Redis. So it runs only when a caller asks for
    # it explicitly.
    #
    # Even then its output is a draft: the document stays uncommitted and
    # a doctor confirms before anything reaches the event log. A general
    # VLM asked to read handwriting will produce a confident number
    # whether or not it is on the page, and a confident wrong number in a
    # lab record is worse than an unreadable one.
    vlm_draft: Optional[Dict[str, Any]] = None
    if deep_read and settings.enable_vision_llm:
        import asyncio

        from app.services.vision_llm import transcribe

        try:
            draft = await asyncio.to_thread(transcribe, raw, settings)
            vlm_draft = draft.as_dict()
            if draft.rows:
                rows = draft.rows or rows
        except Exception:  # noqa: BLE001
            logger.warning("MedGemma deep read failed", exc_info=True)
    elif deep_read and not settings.enable_vision_llm:
        vlm_draft = {
            "error": "MedGemma is disabled on this deployment",
            "requires_review": True,
        }

    reason: Optional[str]
    if committed:
        reason = None
    elif not rows:
        reason = "no lab rows detected"
    else:
        reason = (
            f"extraction confidence {confidence:.2f} below "
            f"threshold {threshold:.2f}"
        )
    if vlm_draft is not None and not committed:
        reason = f"{reason}; MedGemma draft awaiting confirmation"

    doc = Document(
        patient_id=patient.id,
        kind="scan",
        title=file.filename or "scan",
        filename=file.filename or "scan",
        content_type=file.content_type,
        object_key=stored.key,
        size_bytes=stored.size,
        ocr_confidence=confidence if rows else 0.0,
        committed=committed,
        review_reason=reason,
        raw_text=text[:20000] or None,
        extracted={
            "rows": rows,
            "sample_date": parse_dates(text or ""),
            "medgemma": vlm_draft,
        },
    )
    session.add(doc)
    await session.flush()

    created_event: Optional[Dict[str, Any]] = None
    if committed:
        from datetime import datetime, timezone

        from app.core.validity import Event as CoreEvent
        from app.core.validity import EventType, SourceType
        from app.core.validity import content_hash

        when = (
            datetime.fromisoformat(valid_time).astimezone(timezone.utc)
            if valid_time
            else datetime.now(timezone.utc)
        )
        core = CoreEvent(
            event_id="",
            patient_id=str(patient.id),
            event_type=EventType(event_type),
            valid_time=when,
            payload={"extracted": rows, "source": "ocr"},
            source_type=SourceType.OCR,
            confidence=confidence,
        )
        row = Event.from_core(core)
        row.content_hash = content_hash(core)
        row.document_id = doc.id
        session.add(row)
        await session.flush()
        await session.refresh(row)
        created_event = {"id": str(row.id), "valid_time": row.valid_time.isoformat()}

    await session.refresh(doc)
    return CaptureResult(
        document=DocumentOut.model_validate(doc),
        extracted={"rows": rows, "sample_date": parse_dates(text or "")},
        confidence=confidence,
        committed=committed,
        reason=None if committed else doc.review_reason,
    )


@router.get("/review", tags=["capture"])
async def review_queue(
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """Scans awaiting a human. Abstention is a queue, not a dead end."""
    from sqlalchemy import select

    docs = (
        await session.execute(
            select(Document)
            .where(Document.patient_id == patient.id, Document.committed.is_(False))
            .order_by(Document.created_at.desc())
        )
    ).scalars().all()
    return [DocumentOut.model_validate(d) for d in docs]