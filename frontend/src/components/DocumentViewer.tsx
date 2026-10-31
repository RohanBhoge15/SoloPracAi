/**
 * Document and image viewer.
 *
 * A record's value is often in the file, not the fields. A radiograph's
 * pixels, a scanned lab sheet's handwriting, a three-page weekly report --
 * none of that is in the structured payload, so a citation that opens only
 * the parsed fields shows the doctor a fraction of what they filed.
 *
 * Two decisions worth stating.
 *
 * The bytes are fetched with the Authorization header and shown through an
 * object URL, not through a presigned link. These documents carry the
 * patient's name, age and phone on the page, and a URL that grants read
 * access can be forwarded to anyone. `/raw` also sends
 * `Content-Disposition: attachment`, so pointing an `<iframe>` at it would
 * download the file instead of displaying it.
 *
 * PDFs are rendered page by page with pdf.js rather than handed to the
 * browser's built-in viewer. The browser viewer cannot be styled, cannot be
 * zoomed from our own controls, and behaves differently across machines --
 * which is a poor thing to demo on someone else's laptop. Page-by-page also
 * means a long report scrolls predictably instead of jumping.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import "./viewer.css";

type Loaded = {
  url: string;
  contentType: string;
  size: number;
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentViewer({
  documentId,
  title,
  onClose,
}: {
  documentId: string;
  title: string;
  onClose: () => void;
}) {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pages, setPages] = useState<HTMLCanvasElement[]>([]);
  const [pageCount, setPageCount] = useState(0);
  const [renderPage, setRenderPage] = useState(0);
  const [scale, setScale] = useState(1);
  const [zoomPct, setZoomPct] = useState(100);
  const bodyRef = useRef<HTMLDivElement>(null);
  // The object URL is revoked on close. Leaving it live pins the whole file
  // in memory for the life of the tab, which matters for a 12 MB scan
  // repeated across a list of citations.
  const urlRef = useRef<string | null>(null);

  const release = useCallback(() => {
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const res = await api.documentBlob(documentId);
        if (cancelled) {
          URL.revokeObjectURL(res.url);
          return;
        }
        urlRef.current = res.url;
        setLoaded({ url: res.url, contentType: res.contentType, size: res.blob.size });
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof ApiError ? e.message : "Could not open the document.");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [documentId]);

  // Release the blob only once the component is gone, not on every re-render.
  useEffect(() => () => release(), [release]);

  // Escape closes, and the page behind must not scroll underneath.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // Consume the event here rather than merely closing.
      //
      // The record drawer behind this modal has its own Escape listener, on
      // `window`. A keydown bubbles document -> window, so that listener
      // fires in the *same* native dispatch, before React has re-rendered and
      // removed it. Suppressing the drawer while a viewer is open is not
      // enough: by the time that suppression would take effect the event has
      // already reached window. So the topmost layer stops propagation, which
      // is what "Escape closes one thing" actually requires.
      e.stopPropagation();
      onClose();
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const isPdf = (loaded?.contentType ?? "").includes("pdf");
  const isImage = (loaded?.contentType ?? "").startsWith("image/");

  // Render every page once, then let CSS scale them. Re-rasterising on every
  // zoom step is what makes a pdf viewer feel slow, and a page at 2x of a
  // clinical document is still legible from a 2x bitmap.
  useEffect(() => {
    if (!isPdf || !loaded) return;
    let cancelled = false;
    let task: { destroy?: () => void } | null = null;

    (async () => {
      try {
        const pdfjs = await import("pdfjs-dist");
        const workerSrc = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
        pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;

        const doc = await pdfjs.getDocument({ url: loaded.url }).promise;
        if (cancelled) return;
        setPageCount(doc.numPages);

        const rendered: HTMLCanvasElement[] = [];
        for (let n = 1; n <= doc.numPages; n++) {
          const page = await doc.getPage(n);
          if (cancelled) return;
          // 1.4x of CSS pixels keeps text crisp on a laptop without
          // rendering a 4K canvas per page.
          const viewport = page.getViewport({ scale: 1.4 });
          const canvas = document.createElement("canvas");
          canvas.width = Math.ceil(viewport.width);
          canvas.height = Math.ceil(viewport.height);
          const ctx = canvas.getContext("2d");
          if (!ctx) continue;
          await page.render({ canvasContext: ctx, viewport }).promise;
          if (cancelled) return;
          rendered.push(canvas);
        }
        task = doc;
        setPages(rendered);
      } catch (e) {
        if (!cancelled) {
          setError(
            e instanceof ApiError
              ? e.message
              : "This PDF could not be rendered in the browser.",
          );
        }
      }
    })();

    return () => {
      cancelled = true;
      task?.destroy?.();
    };
  }, [isPdf, loaded]);

  // Track which page is on screen so the control can say "page 2 of 6".
  useEffect(() => {
    if (!pages.length || !bodyRef.current) return;
    const onScroll = () => {
      const host = bodyRef.current;
      if (!host) return;
      const mid = host.scrollTop + host.clientHeight / 2;
      let best = 0;
      pages.forEach((c, i) => {
        const top = c.parentElement?.offsetTop ?? 0;
        if (top <= mid) best = i;
      });
      setRenderPage(best + 1);
    };
    const host = bodyRef.current;
    host.addEventListener("scroll", onScroll, { passive: true });
    return () => host.removeEventListener("scroll", onScroll);
  }, [pages]);

  const zoom = (next: number) => {
    const clamped = Math.min(2.5, Math.max(0.6, next));
    setScale(clamped);
    setZoomPct(Math.round(clamped * 100));
  };

  return (
    <div
      className="viewer-backdrop"
      onClick={(e) => {
        // Only a click on the backdrop itself closes. A click that started
        // inside the panel and drifted out must not dismiss it.
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="viewer"
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="viewer-head">
          <div className="viewer-titles">
            <div className="viewer-title">{title}</div>
            <div className="viewer-meta">
              {loaded
                ? `${loaded.contentType.split("/")[1]?.toUpperCase() ?? loaded.contentType} · ${formatSize(loaded.size)}${
                    isPdf && pageCount ? ` · ${pageCount} page${pageCount === 1 ? "" : "s"}` : ""
                  }`
                : error
                  ? "unavailable"
                  : "loading…"}
            </div>
          </div>

          <div className="viewer-tools">
            {(isPdf || isImage) && (
              <>
                <button className="viewer-btn" onClick={() => zoom(scale - 0.2)} aria-label="Zoom out">
                  −
                </button>
                <span className="viewer-zoom">{zoomPct}%</span>
                <button className="viewer-btn" onClick={() => zoom(scale + 0.2)} aria-label="Zoom in">
                  +
                </button>
              </>
            )}
            <button className="viewer-btn" onClick={onClose} aria-label="Close document">
              ✕
            </button>
          </div>
        </div>

        <div className="viewer-body" ref={bodyRef}>
          {error && (
            <div className="viewer-msg viewer-msg--err">
              <b>Cannot display this document</b>
              {error}
            </div>
          )}

          {!error && !loaded && (
            <div className="viewer-msg">
              <span className="spinner" /> Opening…
            </div>
          )}

          {loaded && isImage && (
            <div className="viewer-scroll">
              {/* Zoom starts at "fit the whole study in view", not "fill the
                  width". A radiograph that has to be scrolled to see is no use
                  for judging it; the panel then scrolls once the zoom
                  exceeds 100%, which is when detail is actually wanted. */}
              <img
                className="viewer-img"
                src={loaded.url}
                alt={title}
                style={{
                  maxWidth: `${scale * 100}%`,
                  maxHeight: `${scale * 100}%`,
                }}
              />
            </div>
          )}

          {loaded && isPdf && !pages.length && !error && (
            <div className="viewer-msg">
              <span className="spinner" /> Rendering {pageCount ? `${pageCount} pages` : "document"}…
            </div>
          )}

          {loaded && isPdf && pages.length > 0 && (
            <div className="viewer-pages">
              {pages.map((canvas, i) => (
                <div className="viewer-page" key={i}>
                  <div className="viewer-page-no">{i + 1}</div>
                  <canvas
                    ref={(el) => {
                      if (el && canvas) {
                        // Copy the rendered bitmap across. Re-rendering each
                        // canvas on every React pass would throw away the
                        // page and flash.
                        if (el.width !== canvas.width || el.height !== canvas.height) {
                          el.width = canvas.width;
                          el.height = canvas.height;
                        }
                        const ctx = el.getContext("2d");
                        ctx?.drawImage(canvas, 0, 0);
                      }
                    }}
                    style={{ width: `${canvas.width * scale * 0.714}px` }}
                  />
                </div>
              ))}
            </div>
          )}

          {loaded && !isImage && !isPdf && (
            <div className="viewer-msg">
              <b>No inline preview for this file type</b>
              {loaded.contentType} is stored, but SoloPrac does not render it in the
              browser.
              <a className="viewer-dl" href={loaded.url} download>
                Download it instead
              </a>
            </div>
          )}
        </div>

        {isPdf && pages.length > 1 && (
          <div className="viewer-foot">
            Page {renderPage} of {pageCount}
          </div>
        )}
      </div>
    </div>
  );
}