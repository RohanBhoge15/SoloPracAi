"""FastAPI application."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    auth,
    capture,
    chat,
    dashboard,
    documents,
    drugs,
    events,
    notifications,
    patients,
    semantic,
    timeline,
)
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SoloPrac starting in %s mode", settings.environment)
    if settings.llm_configured:
        logger.info("LLM enabled: %s", settings.llm_model)
    else:
        logger.warning(
            "LLM not configured — running model-free. "
            "Answers will be cited extracts rather than summaries."
        )

    # Warm the query encoder in the background. MedCPT takes ~15 s to load
    # on a CPU-only host; doing that lazily would make the first question
    # after a restart look like a hang. Boot stays responsive either way,
    # and if the warm-up fails the request path falls back to lexical.
    warmup: dict[str, object] = {}
    task: asyncio.Task | None = None
    if settings.preload_embeddings:
        task = asyncio.create_task(_warm_embeddings(warmup))

    app.state.embedder_warmup = warmup
    try:
        yield
    finally:
        if task is not None and not task.done():
            task.cancel()
    logger.info("SoloPrac shutting down")


async def _warm_embeddings(sink: dict[str, object]) -> None:
    from app.services.embeddings import get_embedder

    try:
        result = await asyncio.to_thread(get_embedder().preload)
        sink.update(result)
        if result.get("ready"):
            logger.info("embeddings warm in %.1fs", result.get("seconds", -1))
        else:
            logger.warning(
                "embeddings did not load; retrieval falls back to lexical "
                "similarity. See GET /api/v1/semantic/status."
            )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("embedding warm-up failed", exc_info=True)


app = FastAPI(
    title="SoloPrac",
    description=(
        "Validity-aware clinical record assistant. Answers are grounded, "
        "cited, and aware of which records have been superseded."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    auth.router,
    patients.router,
    events.router,
    timeline.router,
    chat.router,
    capture.router,
    documents.router,
    drugs.router,
    notifications.router,
    semantic.router,
    dashboard.router,
):
    app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["ops"])
async def health():
    # The chain, not one model's name: `llm_model` is the last-resort
    # link's id, so reporting only it would name whoever happens to be
    # third while the primary does the work. Names and model ids are not
    # secrets; keys never leave the server.
    chain = settings.llm_provider_chain
    return {
        "status": "ok",
        "service": "soloprac",
        "version": "0.1.0",
        "llm_configured": settings.llm_configured,
        "llm_model": f"{chain[0][0]}:{chain[0][2]}" if chain else settings.llm_model,
        "llm_chain": [f"{name}:{model}" for name, _url, model in chain],
        "embeddings_warm": bool(
            getattr(app.state, "embedder_warmup", {}).get("ready")
        ),
    }


@app.get("/api/v1/config", tags=["ops"])
async def public_config():
    """Non-secret configuration the frontend needs at boot."""
    chain = settings.llm_provider_chain
    return {
        "app_name": settings.app_name,
        "llm_enabled": settings.llm_configured,
        "llm_model": f"{chain[0][0]}:{chain[0][2]}" if chain else settings.llm_model,
        "retrieval_k": settings.retrieval_k,
        "ocr_confidence_threshold": settings.ocr_confidence_threshold,
    }


@app.exception_handler(Exception)
async def unhandled(request, exc):
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "path": request.url.path},
    )
