"""PDF generation with WeasyPrint.

Why WeasyPrint rather than Playwright or ReportLab:

  - WeasyPrint implements CSS paged media. `@page`, running headers,
    orphans/widows and page breaks all behave properly, which is what
    makes a rendered document read like a document.
  - Playwright would need a Chromium in the image (roughly 400 MB) to
    render a template that is itself HTML. For a prescription that is a
    lot of container for one job.
  - ReportLab draws PDFs imperatively. Good output is possible but
    verbose, and a designer cannot restyle it without code changes.

Watermarking uses a fixed-position element, which the page layout engine
repeats on every page. A watermark placed in normal flow appears once and
then scrolls away, which defeats the only purpose it has.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_weasyprint = None
_weasyprint_failed = False


class PDFError(RuntimeError):
    pass


def _weasy():
    """Import WeasyPrint lazily and cache the failure.

    WeasyPrint needs Pango at the native level, which is the most common
    way a container build fails. Failing soft keeps the API up and turns a
    missing system library into a clear message rather than a 500.
    """
    global _weasyprint, _weasyprint_failed
    if _weasyprint is not None:
        return _weasyprint
    if _weasyprint_failed:
        raise PDFError("weasyprint unavailable (missing pango/cairo?)")

    try:
        from weasyprint import HTML

        _weasyprint = HTML
        return HTML
    except Exception as exc:  # pragma: no cover - environment dependent
        _weasyprint_failed = True
        raise PDFError(
            f"weasyprint could not be imported: {exc}. "
            "Install libpango-1.0-0 and libcairo2 in the image."
        ) from exc


def render_html_to_pdf(html: str, base_url: Optional[str] = None) -> bytes:
    """Render an HTML document to PDF bytes."""
    HTML = _weasy()
    try:
        document = HTML(string=html, base_url=base_url).render()
        buffer = io.BytesIO()
        document.write_pdf(buffer)
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise PDFError(f"pdf render failed: {exc}") from exc


def pdf_available() -> bool:
    try:
        _weasy()
        return True
    except PDFError:
        return False
