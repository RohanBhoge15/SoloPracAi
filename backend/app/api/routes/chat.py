"""Chat routes — grounded, cited, validity-aware answers.

Two endpoints:
    POST /chat        a complete answer with citations (non-streaming)
    POST /chat/stream SSE, for bounded perceived latency on a slow tier

Both go through the same pipeline, so the streaming variant cannot drift
from the verified one.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_patient, make_deidentifier
from app.api.schemas import ChatRequest, ChatResponse, CitationOut, SuppressedOut
from app.core.config import get_settings
from app.core.validity import Validity
from app.db.models import Patient
from app.db.session import get_session
from app.services.llm import LLMClient
from app.services.pipeline import AnswerPipeline, build_context

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/patients/{patient_id}/chat", tags=["chat"])


def _pipeline() -> AnswerPipeline:
    """The answering pipeline, wired to the real retriever.

    `semantic_fn` is deliberately left as None. `AnswerPipeline._semantic_for`
    only falls back to the vector scorer - `semantic_for(patient_id)`,
    Qdrant, MedCPT - when no function is injected, so passing
    `lexical_similarity` here was not a conservative default: it replaced
    embedding retrieval with keyword overlap for every chat request.

    The effect was invisible in the answers but obvious in the latency.
    Lexical matching cannot bridge a paraphrase, so a question like "are the
    sugars controlled?" retrieved nothing and the pipeline reported "No
    record directly states whether sugars are controlled" - while
    POST /semantic/search answered the same question from fasting glucose
    and HbA1c in 91 ms. The record was retrievable; chat just was not
    looking for it.

    Cost of the fix: one query encode plus one Qdrant round-trip per
    question, measured at roughly 90 ms, and it happens inside
    `asyncio.to_thread` alongside the ranking rather than on the event loop.
    """
    return AnswerPipeline(
        llm=LLMClient() if settings.llm_configured else None,
        semantic_fn=None,
        k=settings.retrieval_k,
        alpha=settings.retrieval_alpha,
    )


def _citation_validity(event, context, intent) -> str:
    """Validity of a cited record **as of the instant that was asked about**.

    The badge has to agree with the ranking that produced the answer. The
    retriever resolves validity against the query instant (`state_as_of`), so
    asking the present-tense question here instead would label the correct
    answer to an as-of query as retracted: a record corrected in March is
    superseded now, and was current in February. The chip would render amber
    on precisely the citation the doctor asked for.
    """
    instant = getattr(intent, "instant", None)
    if instant is not None and intent.kind in ("as_of", "historical"):
        state = context.graph.state_as_of(event, instant)
    else:
        state = context.graph.state(event)
    return "superseded" if state is Validity.SUPERSEDED else "current"


def _citations(result, context) -> list[CitationOut]:
    from app.services.collapse import _identity

    by_id = {e.event_id: e for e in context.events}
    out: list[CitationOut] = []
    # One chip per entity, not one per version. Seven identical Metformin
    # rows are seven real events but one fact; the chip opens the record
    # and the drawer walks the chain, so nothing is unreachable. First
    # cited wins, preserving ranking order. Events with no identity
    # (empty notes and the like) keep per-event behaviour.
    seen: set[str] = set()
    for citation in result.answer.citations:
        event = by_id.get(citation["event_id"])
        if event is None:
            continue
        try:
            key = _identity(event)
        except Exception:
            key = None
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        payload = ", ".join(
            f"{k}: {v}"
            for k, v in (event.payload or {}).items()
            if not isinstance(v, (dict, list))
        )
        out.append(
            CitationOut(
                event_id=event.event_id,
                text=payload,
                valid_time=event.valid_time,
                validity=_citation_validity(event, context, result.intent),
                event_type=str(event.event_type),
            )
        )
    return out


def _reason_text(reasons: list[str]) -> str:
    return "; ".join(reasons) if reasons else "failed verification"


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    from app.api.deps import load_core_events

    events = await load_core_events(patient.id, session)
    context = build_context(
        str(patient.id),
        events,
        make_deidentifier(patient),
        display_name=patient.display_name,
        age=patient.age,
    )

    pipeline = _pipeline()
    result = await pipeline.answer(context, body.query, as_of=body.as_of)

    return ChatResponse(
        answer=result.answer.text,
        citations=_citations(result, context),
        suppressed=[
            SuppressedOut(text=s["text"], reason=_reason_text(s["reasons"]))
            for s in result.answer.suppressed
        ],
        dropped=[
            SuppressedOut(text=d["text"], reason=_reason_text(d["reasons"]))
            for d in result.answer.dropped
        ],
        intent=result.intent.kind,
        intent_confidence=result.intent.confidence,
        degraded=bool(result.llm and result.llm.degraded),
        latency_ms=round(result.latency_ms, 1),
        model=result.llm.model if result.llm else "extractive",
        has_stale_assertion=result.has_stale_assertion,
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    patient: Patient = Depends(get_patient),
    session: AsyncSession = Depends(get_session),
):
    """SSE. Emits the answer, then citations and any suppression notices.

    The answer is verified *before* the first token is sent, so a stream
    can never carry a claim the linker would have rejected.
    """
    from app.api.deps import load_core_events

    events = await load_core_events(patient.id, session)
    context = build_context(
        str(patient.id),
        events,
        make_deidentifier(patient),
        display_name=patient.display_name,
        age=patient.age,
    )
    pipeline = _pipeline()
    result = await pipeline.answer(context, body.query, as_of=body.as_of)

    async def generate() -> AsyncGenerator[str, None]:
        meta = {
            "intent": result.intent.kind,
            "intent_confidence": result.intent.confidence,
            "latency_ms": round(result.latency_ms, 1),
            "degraded": bool(result.llm and result.llm.degraded),
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
        for chunk in _chunks(result.answer.text):
            yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
        yield f"event: citations\ndata: {json.dumps([c.model_dump(mode='json') for c in _citations(result, context)])}\n\n"
        yield f"event: suppressed\ndata: {json.dumps(result.answer.suppressed)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _chunks(text: str) -> list[str]:
    if not text:
        return []
    return [text[i : i + 40] for i in range(0, len(text), 40)]
