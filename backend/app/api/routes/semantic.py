"""Semantic routes: indexing, search, and the multimodal capture path.

Indexing is a *derived* store. If it is lost, `POST /reindex` rebuilds
it from Postgres, because the event log is the source of truth and the
vectors are a cache. That is the whole reason the two are kept apart.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_patient, load_core_events
from app.core.validity import Event as CoreEvent
from app.core.validity import EventType, SourceType, SupersessionGraph, Validity
from app.core.validity import content_hash
from app.db.documents import Document
from app.db.identity import Account
from app.db.models import Event, Patient
from app.db.session import get_session
from app.api.routes.auth import current_account
from app.services.embeddings import get_embedder
from app.services.hybrid import index_events, semantic_for
from app.services.retrieval import (
    TemporalIntent,
    ValidityAwareRetriever,
    parse_temporal_intent,
)
from app.services.storage import get_storage
from app.services.vectors import get_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/semantic", tags=["semantic"])


class IndexResponse(BaseModel):
    patient_id: str
    events: int
    vectors_written: int
    backend: str
    #: Imaging events re-embedded. Reported separately because image vectors
    #: are optional -- vision can be off, the stored bytes can be gone, and a
    #: reindex that quietly skipped them would leave image search returning
    #: nothing while claiming success.
    images_written: int = 0
    #: Why an imaging event could not be re-indexed, if any. Empty on success.
    images_skipped: List[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    event_id: str
    score: float
    validity: str
    valid_time: Optional[str]
    event_type: Optional[str]


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]
    backend: str
    degraded: bool = False


@router.get("/status")
async def status(account: Account = Depends(current_account)):
    """What this deployment can actually do, not what it intends to.

    Requires a token. No patient data is in here, but it does disclose
    model names, the models directory path, the vector-store collection and
    how many points it holds, which is enough to fingerprint a deployment
    and plan an attack against it. A doctor account is enough; there is no
    reason for this to be public.
    """
    store = get_store()
    return {
        "embedder": get_embedder().status(),
        "vector_store": {
            "healthy": store.healthy(),
            "collection": store.collection,
            "points": store.count(),
        },
    }


@router.post("/reindex", response_model=IndexResponse)
async def reindex(
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """Rebuild a patient's vectors from the event log.

    Deterministic point ids mean re-running is an update, not a
    duplication, so this is safe to call repeatedly.

    Image vectors are rebuilt too, and this used to be a real gap. Text and
    image vectors shared one Qdrant point id, and `upsert` replaces a point's
    whole vector set, so reindexing an imaging record overwrote its
    `biomedclip` vector with its `medcpt` text vector. Image search then
    returned nothing for a study that was still stored, still in the event
    log, and still listed in the UI. The point ids are now separate, and this
    rebuilds the image side so a vector lost to the old behaviour comes back
    without needing the study re-uploaded.
    """
    patient_id = patient.id

    events = await load_core_events(patient_id, session)
    if not events:
        return IndexResponse(
            patient_id=str(patient_id), events=0, vectors_written=0,
            backend="none",
        )

    graph = SupersessionGraph(events)
    # Encoding a full record is seconds of CPU. On the event loop that
    # would stall every other request for the duration.
    written = await asyncio.to_thread(index_events, events, graph)

    images_written, images_skipped = await _reindex_images(events, session)

    return IndexResponse(
        patient_id=str(patient_id),
        events=len(events),
        vectors_written=written,
        backend="qdrant" if written else "lexical-only",
        images_written=images_written,
        images_skipped=images_skipped,
    )


async def _reindex_images(
    events: List[CoreEvent], session: AsyncSession
) -> tuple:
    """Re-embed this patient's imaging studies. Returns (written, skipped).

    Every imaging event is rebuilt from the bytes in object storage, not from
    a caption, because the caption is the text side and this is the pixel
    side. An event whose bytes are gone is reported rather than passed over:
    a silently skipped study is indistinguishable from a working index.
    """
    imaging = [e for e in events if e.event_type == EventType.IMAGING]
    if not imaging:
        return 0, []

    embedder = get_embedder()
    if not embedder.image_available():
        return 0, [
            f"{len(imaging)} imaging event(s) not indexed: image embeddings "
            f"are disabled on this deployment"
        ]

    doc_ids = [
        (e.payload or {}).get("document_id")
        for e in imaging
        if (e.payload or {}).get("document_id")
    ]
    keys: Dict[str, str] = {}
    if doc_ids:
        rows = (
            await session.execute(
                select(Document.id, Document.object_key).where(
                    Document.id.in_([UUID(d) for d in doc_ids])
                )
            )
        ).all()
        keys = {str(r[0]): r[1] for r in rows if r[1]}

    storage = get_storage()
    store = get_store()
    written = 0
    skipped: List[str] = []
    for ev in imaging:
        doc_id = (ev.payload or {}).get("document_id")
        key = keys.get(str(doc_id)) if doc_id else None
        label = str((ev.payload or {}).get("study") or ev.event_id)

        if not key:
            skipped.append(f"{label}: no stored image on this record")
            continue
        try:
            raw = await storage.get(key)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{label}: stored image unreadable ({exc})")
            continue

        vector = await asyncio.to_thread(embedder.encode_image, raw)
        if vector is None:
            skipped.append(f"{label}: image could not be embedded")
            continue

        caption = str(
            (ev.payload or {}).get("study")
            or (ev.payload or {}).get("findings")
            or label
        )
        ok = await asyncio.to_thread(
            store.upsert_image,
            ev.event_id,
            ev.patient_id,
            vector,
            caption,
            "imaging",
            ev.valid_time,
        )
        if ok:
            written += 1
        else:
            skipped.append(f"{label}: vector store refused the write")

    return written, skipped


@router.post("/search", response_model=SearchResponse)
async def search(
    patient_id: UUID,
    query: str = Query(min_length=1, max_length=500),
    as_of: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50),
    include_superseded: bool = True,
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """Semantic search over one patient's record.

    Superseded records are included by default and labelled. Excluding
    them by default would hide the very corrections this system exists to
    surface; the caller decides.

    `get_patient` is the authorisation: the id in the query string is
    checked against the caller before anything is read or ranked. This route
    previously took a bare `patient_id` and resolved it with no token
    required at all, so an unauthenticated request naming any patient id
    returned that patient's whole record - ranked, scored, and looking
    exactly like a legitimate search result. It was the least detectable
    version of the leak this project has had, because the output is a list
    of well-formed hits rather than an error.
    """
    patient_id = patient.id

    as_of_epoch = None
    intent_now = datetime.now(timezone.utc)
    if as_of:
        try:
            parsed = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            as_of_epoch = int(parsed.timestamp())
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="as_of must be ISO-8601"
            ) from exc

    events = await load_core_events(patient_id, session)
    graph = SupersessionGraph(events)
    retriever = ValidityAwareRetriever(graph=graph)

    if as_of_epoch is not None:
        intent = TemporalIntent(
            kind="as_of",
            instant=datetime.fromtimestamp(as_of_epoch, tz=timezone.utc),
            confidence=1.0,
            cue="explicit",
        )
    else:
        intent = parse_temporal_intent(query, intent_now)

    scorer = semantic_for(
        str(patient_id),
        as_of=as_of_epoch,
        include_superseded=include_superseded,
    )
    # The query encode plus the vector search is a forward pass and a
    # network round-trip; keep both off the event loop.
    results = await asyncio.to_thread(
        retriever.rank,
        query,
        str(patient_id),
        events,
        intent,
        intent_now,
        scorer,
        limit,
    )

    # `include_superseded` is enforced here as well as in the Qdrant
    # filter. The filter alone is not enough: the retriever ranks every
    # candidate in memory and demotes superseded records rather than
    # dropping them, so a caller asking to exclude them would still get
    # them back whenever the index is cold. Enforcing it in both places is
    # what makes the parameter mean the same thing whether or not the
    # vector store is reachable.
    if not include_superseded:
        results = [
            r for r in results if r.validity is not Validity.SUPERSEDED
        ]

    return SearchResponse(
        query=query,
        hits=[
            SearchHit(
                event_id=r.event_id,
                score=round(r.final, 4),
                validity=str(r.validity),
                valid_time=r.event.valid_time.isoformat(),
                event_type=str(r.event.event_type),
            )
            for r in results
        ],
        backend=scorer.backend,
        degraded=scorer.backend != "qdrant",
    )


@router.post("/image")
async def index_image(
    patient_id: UUID,
    file: UploadFile = File(...),
    caption: str = Form(default=""),
    findings: str = Form(default=""),
    valid_time: str = Form(default=""),
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """Embed a radiograph, store the pixels, and file it as a real event.

    Three things happen, and all three are needed for the record to be
    genuinely multimodal:

      - the bytes go to object storage, so they survive and can be opened
      - the pixels are embedded with BiomedCLIP, so a *visual* similarity
        search is possible rather than only a text search over captions
      - an event is appended to the log, so the image participates in
        three-valued resolution like every other record

    The last is the one that is easy to get wrong. An image that is stored
    but absent from the event log is invisible to the retriever, to
    supersession and to "as of" queries, which would make the whole
    validity contribution silently inapplicable to scans.

    `valid_time` is the date of the *study*, not the upload. Defaulting it
    to now would file a two-year-old radiograph as new, and a present-tense
    question would then return it as the most recent imaging.
    """
    embedder = get_embedder()
    if not embedder.image_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "image embeddings are disabled. Set ENABLE_VISION=true and "
                "restart; BiomedCLIP needs ~750 MB of weights and runs on CPU."
            ),
        )

    # `get_patient` authorised the call and already proved this row is the
    # caller's, so there is no second lookup here. Uploading a study to
    # someone else's chart is a write, and a write leak is worse than a read.
    patient_id = patient.id

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    when = datetime.now(timezone.utc)
    if valid_time:
        try:
            when = datetime.fromisoformat(
                valid_time.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="valid_time must be ISO-8601"
            ) from exc

    # Roughly a second of CPU for a ViT forward pass on this host.
    vector = await asyncio.to_thread(embedder.encode_image, raw)
    if vector is None:
        raise HTTPException(status_code=422, detail="could not encode image")

    content_type = file.content_type or "image/jpeg"
    filename = file.filename or "scan"
    stored = await get_storage().put(
        raw, f"imaging/{patient_id}", filename, content_type=content_type
    )

    doc = Document(
        patient_id=patient.id,
        kind="scan",
        title=caption or filename,
        filename=filename,
        content_type=content_type,
        object_key=stored.key,
        size_bytes=stored.size,
        caption=caption or None,
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
            "study": caption or filename,
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
    indexed = await asyncio.to_thread(
        get_store().upsert_image,
        event_id,
        str(patient_id),
        vector,
        caption or findings or caption_for(filename),
        "imaging",
        when,
    )
    if not indexed:
        # The bytes and the event are already written. This is a degraded
        # index, not a lost record, so it is reported rather than raised —
        # raising would roll back the event and lose the study entirely.
        logger.warning(
            "image event %s stored and filed, but not vector-indexed", event_id
        )

    return {
        "document_id": str(doc.id),
        "event_id": event_id,
        "object_key": stored.key,
        "size_bytes": stored.size,
        "vector_dim": int(len(vector)),
        "caption": caption,
        "valid_time": when.isoformat(),
        "indexed": indexed,
    }


def caption_for(filename: str) -> str:
    """A last-resort caption when the doctor typed nothing.

    A filename is weak evidence, so it is used only when there is no
    alternative: an empty caption would leave the study findable solely by
    re-uploading a similar file.
    """
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return stem or "clinical image"


@router.post("/image/{image_path:path}/find-similar")
async def find_similar(
    image_path: str,
    limit: int = Query(default=5, ge=1, le=20),
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """Find this patient's prior studies that look similar.

    Only possible because images and text share one space: the query image
    and the stored study are compared directly, with no caption in the loop
    and no OCR in the loop either.

    Authorised through `get_patient`, which also bounds the search to this
    doctor's own records - and separately rejects an `image_path` belonging
    to someone else, below.
    """
    patient_id = patient.id

    embedder = get_embedder()
    if not embedder.image_available():
        raise HTTPException(
            status_code=503, detail="image embeddings are disabled"
        )

    # `get_patient` proves the *patient* is the caller's. The image path is
    # a separate input and has to be checked against that patient's own
    # documents, or a doctor who guessed another practice's object key could
    # have it rendered and read back as a finding. That is an object-store
    # path traversal in all but name.
    from app.db.documents import Document

    owned = (
        await session.execute(
            select(Document.object_key).where(
                Document.patient_id == patient_id,
                Document.object_key.isnot(None),
            )
        )
    ).scalars().all()

    if not any(k == image_path for k in owned):
        raise HTTPException(status_code=404, detail="image not found")

    raw = await get_storage().get(image_path)
    vector = await asyncio.to_thread(embedder.encode_image, raw)
    if vector is None:
        raise HTTPException(status_code=422, detail="could not encode image")

    hits = await asyncio.to_thread(
        get_store().search_by_image, vector, str(patient_id), limit
    )
    return {"query": image_path, "hits": hits}


@router.post("/image-search")
async def image_text_search(
    description: str = Query(min_length=2, max_length=300),
    limit: int = Query(default=5, ge=1, le=20),
    patient: Patient = Depends(get_patient),
):
    """Find this patient's studies by *describing* them.

    The query is embedded in the same joint space as the pixels, so
    "chest film showing an effusion" reaches a study that was never
    captioned in those words. This is the capability that makes the record
    multimodal rather than merely image-storing.

    Authorised through `get_patient`. This one had no auth at all, so any
    caller could describe studies and get back whatever images were indexed
    for a patient id they named.
    """
    patient_id = patient.id

    embedder = get_embedder()
    if not embedder.image_available():
        raise HTTPException(
            status_code=503, detail="image embeddings are disabled"
        )

    vector = await asyncio.to_thread(
        embedder.encode_image_caption, description
    )
    if vector is None:
        raise HTTPException(status_code=422, detail="could not encode query")

    store = get_store()
    hits = await asyncio.to_thread(
        store.search_by_image, vector, str(patient_id), limit
    )
    return {"query": description, "hits": hits, "degraded": not store.healthy()}
