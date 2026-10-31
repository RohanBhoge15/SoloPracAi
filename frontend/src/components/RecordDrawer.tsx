import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import type { EventOut, Timeline } from "@/lib/api";
import DocumentViewer from "./DocumentViewer";
import "./drawer.css";

/* ── formatting ──────────────────────────────────────────────── */

const full = (iso: string) =>
  new Date(iso).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

const day = (iso: string) =>
  new Date(iso).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });

/** Keys rendered in the grid, and how. */
const HIDDEN = new Set(["ref_low", "ref_high"]);

function isProse(key: string, value: unknown) {
  return (
    (key === "note" || key === "findings" || key === "reason" ||
      key === "amendment") &&
    typeof value === "string"
  );
}

function title(ev: EventOut): string {
  const p = ev.payload as Record<string, unknown>;
  switch (ev.event_type) {
    case "lab":
      return String(p.test ?? "Lab result");
    case "medication":
      return String(p.drug ?? "Medication");
    case "diagnosis":
      return String(p.diagnosis ?? "Diagnosis");
    case "vital":
      return String(p.measurement ?? "Vital sign");
    case "imaging":
      return String(p.study ?? "Imaging");
    default:
      return ev.event_type.charAt(0).toUpperCase() + ev.event_type.slice(1);
  }
}

/** The single most important number, promoted to a headline. */
function headline(ev: EventOut): { value: string; unit: string; flag?: string } | null {
  const p = ev.payload as Record<string, unknown>;
  if (ev.event_type === "lab" && p.value !== undefined) {
    return {
      value: String(p.value),
      unit: String(p.unit ?? ""),
      flag: p.flag ? String(p.flag) : undefined,
    };
  }
  if (ev.event_type === "medication" && p.dose !== undefined) {
    return { value: String(p.dose), unit: String(p.frequency ?? "") };
  }
  if (ev.event_type === "vital" && p.value !== undefined) {
    return { value: String(p.value), unit: String(p.unit ?? "") };
  }
  return null;
}

function oneLine(ev: EventOut): string {
  const p = ev.payload as Record<string, unknown>;
  switch (ev.event_type) {
    case "lab":
      return `${p.test}: ${p.value} ${p.unit ?? ""}`.trim();
    case "medication": {
      const stop = p.status === "stopped" ? " (stopped)" : "";
      return `${p.drug} ${p.dose ?? ""} ${p.frequency ?? ""}${stop}`.trim();
    }
    case "diagnosis":
      return `${p.diagnosis}${p.icd10 ? ` (${p.icd10})` : ""}`;
    case "vital":
      return `${p.measurement}: ${p.value} ${p.unit ?? ""}`.trim();
    case "imaging":
      return `${p.study}: ${p.impression ?? p.findings ?? ""}`;
    default:
      return String(p.note ?? "").slice(0, 90);
  }
}

/* ── drawer ──────────────────────────────────────────────────── */

interface Props {
  event: EventOut | null;
  timeline: Timeline | null;
  onClose: () => void;
  /** Navigate to a sibling version in the same correction chain. */
  onJump: (id: string) => void;
}

export default function RecordDrawer({ event, timeline, onClose, onJump }: Props) {
  const open = event !== null;
  const [viewerDoc, setViewerDoc] = useState<{ id: string; title: string } | null>(
    null,
  );

  // Close the viewer first when the drawer itself goes, so the modal is never
  // left floating over nothing.
  useEffect(() => {
    if (!open) setViewerDoc(null);
  }, [open]);

  // Escape to close. Registered only while open so the landing page's
  // shortcuts are unaffected.
  //
  // Not while the document viewer is up. Both layers listen for Escape --
  // the viewer on `document`, this on `window` -- so one keypress would
  // otherwise close the viewer *and* the drawer behind it, losing the record
  // the viewer was opened from. Escape should dismiss the topmost layer only.
  useEffect(() => {
    if (!open || viewerDoc) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose, viewerDoc]);

  // Every version of the same entity, oldest first, so the clinician can
  // walk the whole history of one measurement.
  const lineage = useMemo(() => {
    if (!event || !timeline) return [];
    const all = timeline.history;
    const sameEntity = (e: EventOut) => {
      const a = e.payload as Record<string, unknown>;
      const b = event.payload as Record<string, unknown>;
      if (e.event_type !== event.event_type) return false;
      if (event.event_type === "lab") return a.test === b.test;
      if (event.event_type === "medication") return a.drug === b.drug;
      if (event.event_type === "diagnosis") return true;
      if (event.event_type === "vital") return a.measurement === b.measurement;
      return false;
    };
    return all
      .filter(sameEntity)
      .sort(
        (a, b) =>
          new Date(a.valid_time).getTime() - new Date(b.valid_time).getTime(),
      );
  }, [event, timeline]);

  if (!event) {
    return (
      <AnimatePresence>
        {null}
      </AnimatePresence>
    );
  }

  const p = event.payload as Record<string, unknown>;
  const main = headline(event);
  const isStale = event.validity === "superseded";
  const otherVersions = lineage.filter((e) => e.id !== event.id);

  // The file behind this record, when there is one.
  //
  // Not every record has an attachment, and saying so is better than showing
  // an empty "no document" box on the majority that are just structured
  // values. A prescription, a scanned report and a radiograph all do.
  const attachment = (() => {
    const id = p.document_id;
    if (typeof id !== "string" || !id) return null;
    const title =
      (typeof p.study === "string" && p.study) ||
      (typeof p.title === "string" && p.title) ||
      (typeof p.document_kind === "string" &&
        `${p.document_kind} for this record`) ||
      "Attached document";
    return { id, title };
  })();

  // Portaled to <body> so the drawer escapes the page's stacking context.
  //
  // The same bug was already found and fixed for the authoring sheet, and
  // this drawer was missed. Rendered inline, the drawer's DOM ancestors are
  // `.drawer > .topmain > .shell`, and `.topmain` carries `z-index: 1`, which
  // creates a stacking context. So the drawer's own `z-index: 101` is
  // resolved *inside* that context and can never beat the sticky `.topbar` at
  // `z-index: 50` — no matter how large the number.
  //
  // The visible consequence was that the whole drawer rendered behind the
  // navbar, and because the close button sits in the drawer's top-right
  // corner, directly under the navbar's height, it could not be clicked at
  // all: `elementFromPoint` over it returned the topbar. Escape and the
  // footer Close button still worked, so the drawer looked fine in a
  // screenshot and was unusable with a mouse.
  return createPortal(
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="scrim"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          />

          <motion.aside
            className="drawer"
            role="dialog"
            aria-modal="true"
            aria-label={`Record: ${title(event)}`}
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 32, stiffness: 320 }}
          >
            {/* ── head ── */}
            <div className="drawer-head">
              <div className="drawer-top">
                <span className="ev-type">{event.event_type}</span>
                <button className="drawer-close" onClick={onClose} aria-label="Close">
                  ✕
                </button>
              </div>
              <div className="drawer-title">{title(event)}</div>
              <div className="drawer-badges">
                <span
                  className={`badge badge-${event.validity}`}
                >
                  <span className="dot" />
                  {event.validity}
                </span>
                {event.source_type === "correction" && (
                  <span className="badge badge-accent">correction</span>
                )}
                {event.source_type === "ocr" && (
                  <span className="badge badge-accent">
                    from scan · {Math.round((event.confidence ?? 0) * 100)}%
                  </span>
                )}
                {typeof p.flag === "string" && p.flag !== "normal" && (
                  <span className="badge badge-danger">{p.flag}</span>
                )}
              </div>
            </div>

            {/* ── body ── */}
            <div className="drawer-body">
              {/* the file itself, when the record has one */}
              {attachment && (
                <>
                  <div className="section-label">
                    <span style={{ color: "var(--accent)" }}>▤</span>
                    Attached document
                  </div>
                  <button
                    className="attach"
                    onClick={() =>
                      setViewerDoc({ id: attachment.id, title: attachment.title })
                    }
                  >
                    <span className="attach-ico" aria-hidden>
                      ⬗
                    </span>
                    <span className="attach-text">
                      <span className="attach-title">{attachment.title}</span>
                      <span className="attach-hint">
                        Opens in a viewer — scrollable for multi-page reports
                      </span>
                    </span>
                    <span className="attach-go" aria-hidden>
                      →
                    </span>
                  </button>
                </>
              )}

              {/* headline value */}
              {main && (
                <div
                  className={`headline headline--${isStale ? "superseded" : "current"}`}
                >
                  <div className="headline-label">
                    {isStale ? "Superseded value" : "Current value"}
                  </div>
                  <div className="headline-value">
                    {main.value}
                    {main.unit && <span className="headline-unit">{main.unit}</span>}
                    {main.flag && main.flag !== "normal" && (
                      <span className={`flag-pill flag-pill--${main.flag}`}>
                        {main.flag}
                      </span>
                    )}
                  </div>
                  <div className="headline-note">
                    {isStale
                      ? p.amendment
                        ? `Replaced: ${p.amendment}`
                        : "A later record corrected this value. Shown for audit only."
                      : "This is the version of the record the system treats as true now."}
                  </div>
                </div>
              )}

              {/* full payload */}
              <div className="section-label">
                <span style={{ color: "var(--accent)" }}>◈</span>
                Record contents
              </div>
              <div className="kv">
                {Object.entries(p)
                  .filter(([k]) => !HIDDEN.has(k))
                  .map(([k, v]) => (
                    <div key={k} style={{ display: "contents" }}>
                      <div className="kv-key">{k.replace(/_/g, " ")}</div>
                      <div
                        className={`kv-val${isProse(k, v) ? " kv-val--prose" : ""}`}
                      >
                        {typeof v === "boolean"
                          ? v
                            ? "yes"
                            : "no"
                          : String(v ?? "—")}
                      </div>
                    </div>
                  ))}
                <div style={{ display: "contents" }}>
                  <div className="kv-key">event id</div>
                  <div className="kv-val">{event.id}</div>
                </div>
              </div>

              {/* bitemporal split - the medico-legal property */}
              <div className="section-label">
                <span style={{ color: "var(--current)" }}>◷</span>
                Two clocks
              </div>
              <div className="time-split">
                <div className="time-card time-card--valid">
                  <div className="time-card-label">Clinically true</div>
                  <div className="time-card-value">{day(event.valid_time)}</div>
                </div>
                <div className="time-card time-card--recorded">
                  <div className="time-card-label">Recorded here</div>
                  <div className="time-card-value">{day(event.recorded_at)}</div>
                </div>
              </div>
              <div className="time-explainer">
                These are tracked independently. A lab result is entered with the
                date it describes, not the date it arrived — so a report filed
                three days late still sorts to the right point in the record, and
                &ldquo;what did we know then&rdquo; stays answerable.
              </div>

              {/* version lineage */}
              {otherVersions.length > 0 && (
                <>
                  <div className="section-label">
                    <span style={{ color: "var(--superseded)" }}>⟲</span>
                    All {lineage.length} versions of this record
                  </div>
                  <div className="lineage">
                    {lineage.map((e, i) => (
                      <div key={e.id}>
                        {i > 0 && (
                          <div className="lineage-arrow">
                            ↓{" "}
                            {e.source_type === "correction"
                              ? "corrected"
                              : "changed"}
                          </div>
                        )}
                        <button
                          className={`lineage-node${e.id === event.id ? " lineage-node--active" : ""}`}
                          onClick={() => e.id !== event.id && onJump(e.id)}
                          disabled={e.id === event.id}
                          style={
                            e.id === event.id
                              ? { cursor: "default" }
                              : undefined
                          }
                        >
                          <span className="lineage-rail">
                            <span
                              className={`lineage-dot lineage-dot--${e.validity}`}
                            />
                            {i < lineage.length - 1 && (
                              <span className="lineage-stem" />
                            )}
                          </span>
                          <span className="lineage-text">
                            <span className="lineage-title">{oneLine(e)}</span>
                            <span className="lineage-sub">
                              {full(e.valid_time)} · {e.validity}
                            </span>
                          </span>
                        </button>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* tamper evidence */}
              <div className="section-label">
                <span style={{ color: "var(--text-3)" }}>⛉</span>
                Tamper evidence
              </div>
              <div className="hash">
                content-hash {event.content_hash}
                <br />
                {event.supersedes_id
                  ? `supersedes ${event.supersedes_id}`
                  : "no predecessor — this is a root record"}
              </div>
            </div>

            {/* ── foot ── */}
            <div className="drawer-foot">
              {otherVersions.length > 0 && (
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => {
                    const idx = lineage.findIndex((e) => e.id === event.id);
                    const target = lineage[(idx + 1) % lineage.length];
                    if (target.id !== event.id) onJump(target.id);
                  }}
                >
                  View{" "}
                  {lineage.find((e) => e.id !== event.id)?.validity === "superseded"
                    ? "previous version"
                    : "other version"}
                </button>
              )}
              <button
                className="btn btn-primary btn-sm"
                style={{ marginLeft: "auto" }}
                onClick={onClose}
              >
                Close
              </button>
            </div>
          </motion.aside>

          {/* The file itself, over the drawer. Rendered outside the motion
              element so the drawer's exit animation cannot take the viewer
              with it mid-read. */}
          {viewerDoc && (
            <DocumentViewer
              documentId={viewerDoc.id}
              title={viewerDoc.title}
              onClose={() => setViewerDoc(null)}
            />
          )}
        </>
      )}
    </AnimatePresence>,
    document.body,
  );
}
