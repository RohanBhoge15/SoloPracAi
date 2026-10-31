/**
 * The shared "we just issued a document" flow, used by the certificate,
 * invoice and weekly report sheets so they cannot drift apart.
 *
 * The doctor's PDF is the point of all three, so handing it over is part of
 * issuing it rather than a separate action. The complication is that a new
 * tab can only be opened from a real user gesture, and by the time the
 * signed URL has come back from the API the original click is over. So:
 *
 *   - desktop: open a tab and close the sheet, as before
 *   - phone:   keep the sheet open on a success screen with a "View PDF"
 *              button, because the tap on that button *is* a user gesture
 *
 * PrescriptionComposer predates this and keeps its own copy, because it has
 * to render its own confirmation and auto-close after 1.4s.
 */

import { useCallback, useState } from "react";
import { api, ApiError, type DocumentOut } from "@/lib/api";
import { openPdfInNewTab } from "@/lib/pdf";

/** What `content.reference` looks like on a generated document. */
function referenceOf(doc: DocumentOut, fallback: string): string {
  const raw = (doc.content as Record<string, string> | null)?.reference;
  return raw ?? `${fallback} ${doc.id.slice(0, 8)}`;
}

export function useIssuedDocument(fallback: string) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reference, setReference] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  /**
   * Issue the document, then surface the PDF.
   *
   * `onOpened(reference)` runs only when a tab was opened for the doctor,
   * which is the desktop path where the sheet should close itself. It
   * receives the reference because every caller already reports one.
   */
  const issue = useCallback(
    async (
      create: () => Promise<DocumentOut>,
      onOpened?: (reference: string) => void,
    ) => {
      setBusy(true);
      setError(null);
      try {
        const doc = await create();
        const ref = referenceOf(doc, fallback);
        setReference(ref);

        // The document already exists. A failure to fetch or open its
        // preview must not be reported as a failure to issue it.
        try {
          const { url } = await api.downloadDocument(doc.id);
          if (openPdfInNewTab(url)) {
            onOpened?.(ref);
          } else {
            // No tab could be opened; hold the URL for the button.
            setPdfUrl(url);
          }
        } catch {
          setError("Issued, but the PDF preview could not be loaded.");
        }
      } catch (err) {
        setError(
          err instanceof ApiError ? err.message : "Could not issue it.",
        );
      } finally {
        setBusy(false);
      }
    },
    [fallback],
  );

  return { busy, error, reference, pdfUrl, issue };
}

/**
 * The confirmation shown when no tab could be opened for the doctor.
 *
 * Kept deliberately plain: a reference they can read aloud, the file, and
 * one way out. The whole point is that a phone user is not stranded on a
 * blank sheet with no idea what happened.
 */
export function IssuedSuccess({
  label,
  reference,
  pdfUrl,
  onDone,
}: {
  label: string;
  reference: string;
  pdfUrl: string | null;
  onDone: () => void;
}) {
  return (
    <div className="issued">
      <div className="done-mark" aria-hidden="true">
        <svg
          width="30"
          height="30"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m5 13 4 4L19 7" />
        </svg>
      </div>
      <h3 className="issued-title">{label}</h3>
      <p className="mono issued-ref">{reference}</p>
      <div className="issued-actions">
        {pdfUrl && (
          <a
            className="btn btn-primary"
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            View PDF
          </a>
        )}
        <button className="btn btn-ghost" onClick={onDone}>
          Done
        </button>
      </div>
    </div>
  );
}
