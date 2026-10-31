import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import AppShell from "@/components/AppShell";
import DocumentViewer from "@/components/DocumentViewer";
import { api, ApiError, type InboxItem, type PatientRecordRow } from "@/lib/api";
import { isMobile, openPdfInNewTab } from "@/lib/pdf";
import { useSession } from "@/lib/session";
import "@/pages/portal.css";

const NAV = [
  { href: "/portal", label: "My documents" },
  { href: "/settings", label: "Settings" },
];

const KIND_LABEL: Record<string, string> = {
  prescription: "Prescription",
  certificate: "Certificate",
  invoice: "Invoice",
  weekly_report: "Health report",
  scan: "Scan",
};

const KIND_ICON: Record<string, string> = {
  prescription: "Rx",
  certificate: "Cert",
  invoice: "INR",
  weekly_report: "Report",
  scan: "Scan",
};

function when(iso: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const days = Math.floor((Date.now() - then) / 86_400_000);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  if (days < 60) return `${Math.floor(days / 7)} weeks ago`;
  return new Date(iso).toLocaleDateString();
}

function size(bytes: number | null): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function initials(name: string | null): string {
  if (!name) return "D";
  return (
    name
      .split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((p) => p[0]?.toUpperCase() ?? "")
      .join("") || "D"
  );
}

/**
 * A doctor's picture in "My doctors", or their initials.
 *
 * This panel was always initials, because the records payload carried no
 * account id, so the picture could not be requested even in principle.
 * The id is now sent along with a `doctor_has_photo` hint, and the
 * presigned URL is fetched per row.
 *
 * The initials fallback is not decoration: most doctors never set a
 * picture, and the row must still render a sensible identity rather than
 * a broken image.
 */
function DoctorAvatar({ row }: { row: PatientRecordRow }) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    if (!row.doctor_has_photo || !row.doctor_account_id) {
      setSrc(null);
      return;
    }
    void api.avatarUrl(row.doctor_account_id).then((url) => {
      if (live) setSrc(url);
    });
    return () => {
      live = false;
    };
  }, [row.doctor_account_id, row.doctor_has_photo]);

  if (!src) {
    return <span className="davatar">{initials(row.doctor_name)}</span>;
  }
  return (
    <img
      className="davatar davatar--img"
      src={src}
      alt=""
      onError={() => setSrc(null)}
    />
  );
}

export default function PatientPortal() {
  const session = useSession();
  const [inbox, setInbox] = useState<InboxItem[] | null>(null);
  const [records, setRecords] = useState<PatientRecordRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [opening, setOpening] = useState<string | null>(null);
  // In-app viewer for phones and the PWA. A phone cannot open a tab
  // programmatically (no user gesture survives the API round trip), and
  // same-tab navigation strands the patient on the PDF with only Back to
  // return. The modal renders with pdf.js, so it works where an iframe
  // would show a blank page, and it carries its own close button.
  const [viewing, setViewing] = useState<InboxItem | null>(null);

  const load = useCallback(async () => {
    try {
      const [box, recs] = await Promise.all([api.inbox(), api.myRecords()]);
      setInbox(box);
      setRecords(recs);
      setError(null);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Could not load your documents.",
      );
      // A distinct empty state from "no documents yet": the first is a
      // failure the user can retry, the second is the real answer.
      setInbox([]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  /* Reload when the app comes back to the foreground.

     A patient is the audience most likely to leave the app: they open a
     prescription in another app to read it, or switch to a messaging app
     because that is how the clinic actually tells them it is ready. On a
     phone that suspends background timers, so the fetch-on-mount above is
     the only load that ever happens, and a prescription issued while they
     were away is invisible until they pull-to-refresh or reload the page.

     Cheap on purpose: two GETs, on a transition that happens a handful of
     times an hour. */
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible") void load();
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("pageshow", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("pageshow", onVisible);
    };
  }, [load]);

  async function open(item: InboxItem) {
    setOpening(item.document_id);
    try {
      const { url, filename } = await api.openDocument(item.document_id);
      // Desktop: a new tab, via the shared helper. Deliberately NOT a raw
      // `window.open(url, "_blank", "noopener,noreferrer")` here: with
      // `noopener` in the feature string Chrome returns null even when the
      // tab opened, so the old `if (!tab)` fallback fired every time and
      // the portal tab navigated to the PDF as well - the double-open this
      // replaces. See lib/pdf for the full account.
      if (!isMobile() && openPdfInNewTab(url)) {
        // Tab is open; nothing further to do.
      } else {
        // Phone, PWA, or blocked popup: the document opens in place, in
        // the scrollable viewer with its own close button. The portal
        // keeps its place underneath.
        setViewing(item);
      }
      setInbox((prev) =>
        prev?.map((d) =>
          d.document_id === item.document_id ? { ...d, read: true } : d,
        ) ?? null,
      );
      void filename;
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Could not open that document.",
      );
    } finally {
      setOpening(null);
    }
  }

  const first = session?.account.display_name.split(" ")[0] ?? "there";
  const unread = inbox?.filter((d) => !d.read).length ?? 0;

  return (
    <AppShell items={NAV}>
      <div className="dash-head">
        <div>
          <h1 className="dash-title">Hello, {first}</h1>
          <p className="dash-sub">
            {inbox === null
              ? "Loading your documents..."
              : inbox.length === 0
                ? "Nothing has been sent to you yet."
                : `${inbox.length} document${inbox.length === 1 ? "" : "s"}${
                    unread ? `, ${unread} unread` : ""
                  }`}
          </p>
        </div>
      </div>

      {error && (
        <p className="dash-error">
          {error}{" "}
          <button type="button" className="linkish" onClick={() => void load()}>
            Try again
          </button>
        </p>
      )}

      <div className="portal-grid">
        <section className="portal-main">
          <h2 className="dash-h2">Documents</h2>

          {inbox === null && (
            <div className="dash-empty card">
              <span className="spinner" />
            </div>
          )}

          {inbox !== null && inbox.length === 0 && (
            <div className="dash-empty card">
              <div className="empty-mark" aria-hidden="true">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <path d="M14 2v6h6" />
                </svg>
              </div>
              <p>
                When your doctor issues a prescription or report, it will
                appear here.
              </p>
            </div>
          )}

          <ul className="doclist">
            <AnimatePresence initial={false}>
              {inbox?.map((item, i) => (
                <motion.li
                  key={item.document_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: Math.min(i * 0.04, 0.3) }}
                >
                  <button
                    type="button"
                    className={`doccard card card-hover ${item.read ? "" : "unread"}`}
                    onClick={() => void open(item)}
                    disabled={opening === item.document_id}
                  >
                    <span className={`dockind k-${item.kind}`}>
                      {KIND_ICON[item.kind] ?? "Doc"}
                    </span>
                    <span className="docbody">
                      <span className="doctitle">
                        {KIND_LABEL[item.kind] ?? item.title}
                        {!item.read && <span className="doc-new" />}
                      </span>
                      <span className="docmeta">
                        {item.doctor_name ?? "Your doctor"}
                        {item.doctor_specialty ? `, ${item.doctor_specialty}` : ""}
                        {item.issued_at ? ` - ${when(item.issued_at)}` : ""}
                      </span>
                      <span className="docref mono">{item.reference}</span>
                    </span>
                    <span className="doctail">
                      {item.superseded && (
                        <span className="badge badge-superseded">
                          Replaced
                        </span>
                      )}
                      {size(item.size_bytes) && (
                        <span className="docsize">{size(item.size_bytes)}</span>
                      )}
                      {opening === item.document_id ? (
                        <span className="spinner" />
                      ) : (
                        <span className="docopen">Open</span>
                      )}
                    </span>
                  </button>
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        </section>

        <aside className="portal-side">
          <h2 className="dash-h2">My doctors</h2>
          <p className="side-note">
            One account, every clinic. Your records from each are kept
            together here.
          </p>

          {records.length === 0 && (
            <div className="dash-empty card">
              <p>No clinical records linked yet.</p>
            </div>
          )}

          <ul className="doctorlist">
            {records.map((r) => (
              <li key={r.patient_id} className="doccard card">
                <DoctorAvatar row={r} />
                <span className="docbody">
                  <span className="doctitle">
                    {r.doctor_name ?? "Clinician"}
                  </span>
                  <span className="docmeta">
                    {r.specialty ?? "Records kept for you"}
                  </span>
                  <span className="docmeta">
                    {/* "for you" is load-bearing. The count is scoped to this
                        account's own patient row, but it sits under the
                        doctor's name, so "3 records" reads as the doctor's
                        activity across their whole practice. It is not. */}
                    {r.event_count} record{r.event_count === 1 ? "" : "s"} for
                    you
                    {r.since ? ` since ${when(r.since)}` : ""}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </aside>
      </div>

      {viewing && (
        <DocumentViewer
          documentId={viewing.document_id}
          title={KIND_LABEL[viewing.kind] ?? viewing.title ?? "Document"}
          onClose={() => setViewing(null)}
        />
      )}
    </AppShell>
  );
}
