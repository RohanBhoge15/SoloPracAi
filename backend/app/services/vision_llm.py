"""MedGemma as an opt-in reading aid for scans OCR could not read.

**Measured cost on this host** (CPU only, no GPU, 15.7 GB shared with
Postgres, Qdrant, Redis and SeaweedFS):

    model load          6.3 s
    image encode       54.6 s   (896x896 -> 256 tokens)
    image decode        5.5 s
    plus one generation pass on 256 image tokens
    resident memory    5.4 GB

That is why `ENABLE_VISION_LLM` defaults to false and why capture only
runs this when a caller passes `deep_read=true`. Firing it automatically
whenever OCR abstains would turn a routine upload into a two-minute
stall and would not fit in memory alongside everything else.

**What this is not.** It is not a transcription pipeline. MedGemma 4B is a
general vision-language model, and asked to read a handwritten lab sheet it
will produce a plausible number with complete confidence whether or not
that number is on the page. Wiring that straight into the event log would
put invented values into a clinical record, more confidently than the OCR
it replaced.

So the chain stays three-legged:

    OCR below threshold  ->  document held back (abstention)
                        ->  MedGemma drafts a reading, if explicitly asked
                        ->  a doctor confirms before anything is recorded

The output lands on the document with `committed=False`, exactly where an
unreadable scan already lands. It never creates an event. This is the same
selective-prediction behaviour the OCR gate already implements; MedGemma
sits behind the same gate rather than in front of it.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: MedGemma's context window. 2048 is plenty for a transcription and keeps
#: the KV cache small on a memory-constrained host.
CONTEXT = 2048

SYSTEM = (
    "You transcribe medical document images for a clinician. "
    "You are reading, not interpreting. "
    "Rules you must follow:\n"
    "1. Copy values exactly as they appear. Do not round, convert units, "
    "or infer a value from context.\n"
    "2. If a value is illegible, cut off, or absent, write UNCLEAR. "
    "Never guess a number, a date, or a unit.\n"
    "3. Do not add findings, interpretations, or advice. Only transcribe.\n"
    "4. If the image is not a medical document at all, return "
    '{"readable": false}.\n'
    "Reply with JSON only, no prose and no code fence."
)

USER = (
    "Transcribe this document. Return JSON with these keys:\n"
    '  "readable": boolean - true if this is a medical document at all\n'
    '  "document_type": string, e.g. "lab report", "prescription", '
    '"discharge summary"\n'
    '  "patient_name": string or "UNCLEAR"\n'
    '  "sample_date": string as YYYY-MM-DD, or "UNCLEAR"\n'
    '  "lines": array of objects, each with '
    '{"test": string, "value": string, "unit": string, "ref": string} - '
    "one per result row, in the order printed\n"
    '  "free_text": string - any other visible printed text, verbatim\n'
    "Transcribe only what you can actually read."
)


@dataclass
class Transcription:
    """A draft for a human to accept or reject. Never authoritative."""

    readable: bool
    document_type: str = ""
    patient_name: str = ""
    sample_date: str = ""
    lines: List[Dict[str, str]] = field(default_factory=list)
    free_text: str = ""
    #: Model output exactly as returned, kept so a doctor can see what the
    #: model actually claimed rather than a tidy summary of it.
    raw: str = ""
    seconds: float = 0.0
    error: Optional[str] = None

    @property
    def rows(self) -> List[Dict[str, Any]]:
        """Rows with a readable value, in a shape the capture parser uses."""
        out: List[Dict[str, Any]] = []
        for line in self.lines:
            value = (line.get("value") or "").strip()
            if not value or value.upper() == "UNCLEAR":
                continue
            out.append(
                {
                    "test": (line.get("test") or "").strip() or "Unlabelled",
                    "value": value,
                    "unit": (line.get("unit") or "").strip(),
                    "reference": (line.get("ref") or "").strip(),
                    "source": "medgemma",
                }
            )
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "readable": self.readable,
            "document_type": self.document_type,
            "patient_name": self.patient_name,
            "sample_date": self.sample_date,
            "lines": self.lines,
            "free_text": self.free_text,
            "rows": self.rows,
            "raw": self.raw,
            "seconds": round(self.seconds, 1),
            "error": self.error,
            #: Always false. Nothing from this model is ever committed to a
            #: record without a person confirming it.
            "requires_review": True,
        }


class _MedGemma:
    """Holds the model. Loading costs seconds and gigabytes, so it is lazy
    and loaded at most once per process."""

    def __init__(self) -> None:
        self._llm = None
        self._lock = threading.Lock()
        self._failed = False
        self.load_seconds: float = 0.0

    def available(self, settings: Settings) -> bool:
        if not settings.enable_vision_llm:
            return False
        if self._failed or self._llm is not None:
            return not self._failed
        # `medgemma_gguf` names the *directory*; the two .gguf files live
        # inside it. Checking is_file() on the directory is never true,
        # which is how this reported "not present" while listing the files
        # right beside it.
        root = Path(settings.medgemma_gguf)
        if not (root / settings.medgemma_file).is_file():
            logger.info(
                "MedGemma weights not present: expected %s",
                root / settings.medgemma_file,
            )
            self._failed = True
            return False
        return self._load(settings) is not None

    def _load(self, settings: Settings):
        if self._llm is not None:
            return self._llm
        if self._failed:
            return None
        with self._lock:
            if self._llm is not None:
                return self._llm
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import Llava15ChatHandler

                started = time.time()
                model_path = str(Path(settings.medgemma_gguf) / settings.medgemma_file)
                mmproj = (
                    str(Path(settings.medgemma_gguf) / settings.medgemma_mmproj)
                    if settings.medgemma_mmproj
                    else None
                )
                self._llm = Llama(
                    model_path=model_path,
                    # MedGemma is a Gemma-3 derivative, so it ships a
                    # separate multimodal projector alongside the weights.
                    chat_handler=Llava15ChatHandler(clip_model_path=mmproj),
                    n_ctx=settings.medgemma_context,
                    n_threads=settings.medgemma_threads,
                    # Q4_K_M is already quantised; re-quantising would only
                    # cost quality and startup time.
                    n_gpu_layers=0,
                    verbose=False,
                )
                self.load_seconds = round(time.time() - started, 1)
                logger.info("MedGemma loaded in %.1fs", self.load_seconds)
                return self._llm
            except Exception as exc:  # noqa: BLE001
                logger.warning("MedGemma failed to load: %s", exc)
                self._failed = True
                return None

    def transcribe(self, image_bytes: bytes, settings: Settings) -> Transcription:
        if not self.available(settings):
            return Transcription(
                readable=False, error="MedGemma is not available"
            )
        import base64
        import io

        started = time.time()
        try:
            from PIL import Image

            # Normalised to RGB JPEG first. The projector expects a
            # decoded image, and a PNG screenshot with alpha is a common
            # upload.
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
            encoded = base64.b64encode(buffer.getvalue()).decode()

            llm = self._llm
            assert llm is not None
            # The image must be a typed content part. Writing the
            # "<__media__>" marker by hand makes the handler skip the
            # bitmap, and tokenisation then fails with "number of media
            # markers (1) exceeds number of bitmaps (0)".
            raw = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded}"
                                },
                            },
                        ],
                    },
                ],
                # Greedy. A transcription that varies run to run cannot be
                # reviewed, and this is a reading task, not a creative one.
                temperature=0.0,
                max_tokens=settings.medgemma_max_tokens,
            )
            text = (raw["choices"][0]["message"]["content"] or "").strip()
            elapsed = time.time() - started
            return _parse(text, elapsed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MedGemma transcription failed: %s", exc)
            return Transcription(
                readable=False,
                error=f"{type(exc).__name__}: {exc}",
                seconds=time.time() - started,
            )


def _parse(text: str, elapsed: float) -> Transcription:
    """Tolerant JSON extraction.

    Models wrap JSON in prose and fences even when told not to, so this
    pulls out the outermost object rather than trusting the whole reply.
    A reply that cannot be read at all returns `readable: false`, which
    routes to human review - the safe direction.
    """
    if not text:
        return Transcription(readable=False, error="empty response", seconds=elapsed)

    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return Transcription(
            readable=False, raw=text, error="no JSON in response", seconds=elapsed
        )

    try:
        data = json.loads(candidate[start : end + 1])
    except ValueError as exc:
        return Transcription(
            readable=False, raw=text, error=f"invalid JSON: {exc}", seconds=elapsed
        )

    lines = []
    for item in data.get("lines") or []:
        if isinstance(item, dict):
            lines.append({str(k): str(v) for k, v in item.items()})

    return Transcription(
        readable=bool(data.get("readable", True)),
        document_type=str(data.get("document_type") or ""),
        patient_name=str(data.get("patient_name") or ""),
        sample_date=str(data.get("sample_date") or ""),
        lines=lines,
        free_text=str(data.get("free_text") or ""),
        raw=text,
        seconds=elapsed,
    )


_model = _MedGemma()


def get_medgemma() -> _MedGemma:
    return _model


def transcribe(image_bytes: bytes, settings: Optional[Settings] = None) -> Transcription:
    return _model.transcribe(image_bytes, settings or get_settings())


def status(settings: Optional[Settings] = None) -> Dict[str, Any]:
    settings = settings or get_settings()
    root = Path(settings.medgemma_gguf)
    return {
        "enabled": settings.enable_vision_llm,
        "available": _model.available(settings),
        "model_file_present": (root / settings.medgemma_file).is_file(),
        "projector_present": (root / settings.medgemma_mmproj).is_file(),
        "load_seconds": _model.load_seconds,
        "context": settings.medgemma_context,
        "threads": settings.medgemma_threads,
        "note": (
            "Output is a draft for human review and is never committed to a "
            "record automatically."
        ),
    }
