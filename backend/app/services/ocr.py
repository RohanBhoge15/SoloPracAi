"""OCR and image-quality assessment.

RapidOCR is the primary path (CPU-only, Apache-2.0). It is optional at
import time so the system runs without it — the capture route degrades to
a human review rather than failing the request.
"""

from __future__ import annotations

import io
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_engine = None
_engine_failed = False


class OCRUnavailable(RuntimeError):
    pass


def _get_engine():
    """Lazily construct the OCR engine; cache the failure."""
    global _engine, _engine_failed
    if _engine is not None:
        return _engine
    if _engine_failed:
        raise OCRUnavailable("rapidocr unavailable")
    try:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
        return _engine
    except Exception as exc:  # pragma: no cover - depends on optional dep
        _engine_failed = True
        logger.warning("rapidocr unavailable: %s", exc)
        raise OCRUnavailable(str(exc)) from exc


def image_quality(data: bytes) -> Dict[str, object]:
    """Cheap pre-OCR gate.

    Blur, contrast, and resolution are checked before spending time on
    recognition. A blurry photo yields low confidence from OCR regardless
    of engine, so refusing early is both faster and more honest.
    """
    try:
        from PIL import Image, ImageFilter, ImageStat
    except ImportError:  # pragma: no cover
        return {"readable": True, "reason": "pillow not installed"}

    try:
        image = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        return {"readable": False, "reason": "unreadable image"}

    width, height = image.size
    if width < 400 or height < 400:
        return {
            "readable": False,
            "reason": f"resolution too low ({width}x{height}); need at least 400x400",
        }

    stat = ImageStat.Stat(image)
    contrast = stat.stddev[0]
    if contrast < 12:
        return {
            "readable": False,
            "reason": f"contrast too low ({contrast:.1f}); re-photograph in better light",
        }

    # Variance of Laplacian, via edge energy on a blurred copy.
    edges = image.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).stddev[0]
    if sharpness < 8:
        return {
            "readable": False,
            "reason": f"image too blurry (edge energy {sharpness:.1f}); steady the camera",
        }

    return {
        "readable": True,
        "reason": "ok",
        "width": width,
        "height": height,
        "contrast": round(float(contrast), 1),
        "sharpness": round(float(sharpness), 1),
    }


def extract(data: bytes) -> str:
    """Run OCR and return concatenated text lines."""
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    engine = _get_engine()
    result, _elapsed = engine(np.array(image))
    if not result:
        return ""
    lines = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
            lines.append(item[1])
    return "\n".join(lines)
