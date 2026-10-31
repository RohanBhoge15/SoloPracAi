import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import AppShell from "@/components/AppShell";
import CertificateSheet from "@/components/CertificateSheet";
import InvoiceSheet from "@/components/InvoiceSheet";
import PrescriptionComposer from "@/components/PrescriptionComposer";
import RecordDrawer from "@/components/RecordDrawer";
import WeeklyReportSheet from "@/components/WeeklyReportSheet";
import {
  CaptureSheet,
  NewPatientSheet,
  NewRecordSheet,
} from "@/components/Authoring";
import {
  api,
  ApiError,
  type ChatResponse,
  type EventOut,
  type Patient,
  type Timeline,
} from "@/lib/api";
import { useWhistle } from "@/lib/stt/useWhistle";
import "./console.css";

/* ── helpers ────────────────────────────────────────────────── */

/**
 * Render **bold** without pulling in a markdown library.
 *
 * The extractive path emits markdown emphasis for values, and a full
 * parser would be ~20 kB to format six characters. Anything unexpected
 * is left as literal text rather than silently dropped.
 */
function RichText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <>
      {parts.map((part, i) =>
        part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
          <strong key={i}>{part.slice(2, -2)}</strong>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

const fmtDate = (iso: string) =>
  new Date(iso).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });

const shortDate = (iso: string) =>
  new Date(iso).toLocaleDateString("en-GB", { day: "2-digit", month: "short" });

/** Render a payload as one readable line, whatever the event type. */
function summarise(ev: EventOut): string {
  const p = ev.payload as Record<string, unknown>;
  switch (ev.event_type) {
    case "lab": {
      const flag = p.flag && p.flag !== "normal" ? ` (${p.flag})` : "";
      return `${p.test}: ${p.value} ${p.unit ?? ""}${flag}`.trim();
    }
    case "medication": {
      const stop = p.status === "stopped" ? " — STOPPED" : "";
      return `${p.drug} ${p.dose ?? ""} ${p.frequency ?? ""}${stop}`.trim();
    }
    case "diagnosis":
      return `${p.diagnosis}${p.icd10 ? ` (${p.icd10})` : ""}`;
    case "vital":
      return `${p.measurement}: ${p.value} ${p.unit ?? ""}`.trim();
    case "imaging":
      return `${p.study}: ${p.impression ?? p.findings ?? ""}`;
    default:
      return describeLoosePayload(p);
  }
}

/**
 * A readable one-liner for a record type this list does not render natively.
 *
 * The previous fallback was `String(p.note ?? JSON.stringify(p))`, which
 * dumped the whole payload into the record list. Every document-backed note
 * therefore displayed as
 *   {"title":"Sickness Certificate for Ranveer Thakur","document_id":"8f3…
 * which is unreadable, pushes the patient's name onto the screen twice, and
 * buries the one field a clinician wanted.
 *
 * So: prefer the human title, then the note text, then a count of whatever
 * the payload holds. Never the raw JSON.
 */
function describeLoosePayload(p: Record<string, unknown>): string {
  const title = typeof p.title === "string" ? p.title.trim() : "";
  const note = typeof p.note === "string" ? p.note.trim() : "";

  if (Array.isArray(p.items) && p.items.length > 0) {
    const drugs = p.items
      .map((it) => (it && typeof it === "object" ? (it as Record<string, unknown>).drug : null))
      .filter((d): d is string => typeof d === "string" && d.length > 0);
    const body =
      drugs.length > 0
        ? drugs.join(", ")
        : `${p.items.length} item${p.items.length === 1 ? "" : "s"}`;
    // The titles read "Prescription for <name>", so the drug list is the part
    // that carries information the title does not.
    return `${title ? `${title} — ` : ""}${body}`.slice(0, 130);
  }

  if (title && note) return `${title} — ${note}`.slice(0, 130);
  if (title) return title.slice(0, 130);
  if (note) return note.slice(0, 130);

  // Last resort: name the keys rather than printing their values. An
  // unfamiliar payload should look unfamiliar, not leak its contents.
  const keys = Object.keys(p).filter((k) => k !== "document_id");
  return keys.length > 0 ? `${keys.slice(0, 3).join(", ")} recorded` : "record";
}

/** Locate a record in the loaded timeline, so a citation can be opened. */
function findEvent(timeline: Timeline | null, id: string): EventOut | null {
  if (!timeline) return null;
  return (
    timeline.history.find((e) => e.id === id) ??
    timeline.current.find((e) => e.id === id) ??
    null
  );
}

function ValidityBadge({ validity }: { validity: "current" | "superseded" }) {
  return (
    <span
      className={`badge badge-${validity}`}
      style={{ fontSize: 9, padding: "2px 6px" }}
    >
      {validity}
    </span>
  );
}

/* ── image search ──────────────────────────────────────────────────────
 *
 * The other retrieval surface. Chat searches record *text*, so it finds a
 * study by the words typed beside it. This searches the pixels: the
 * description and the stored image are embedded into one space by the same
 * training run, so "the film with a right lower zone opacity" reaches a
 * radiograph whose caption never contained those words.
 *
 * That distinction is the whole point, so the UI says which one ran. A score
 * here is a pixel-similarity score, not a clinical confidence, and it is
 * labelled as such rather than dressed up as certainty.
 *
 * It is only offered when the chart actually holds imaging. A search box
 * that can only ever return nothing is worse than no search box.
 */

type ImageHit = { event_id: string; caption: string; score: number };

/**
 * Below this, the nearest neighbour is not worth showing as a match.
 *
 * Cosine always returns the k closest points, so a search never comes back
 * empty: "a photograph of a dog in long grass" will still return the
 * chest film, at a score around 0.18. Reporting that as a match would be
 * the system inventing an answer, which is the one thing it must not do.
 *
 * Measured on this deployment with real studies on the chart:
 *   on-target descriptions          0.43
 *   wrong modality (ultrasound)     0.18
 *   unrelated photograph            0.13 - 0.18
 * The gap is wide, so 0.30 sits in it rather than on a boundary. It is
 * stated in the UI because a hidden threshold that silently hides results
 * is indistinguishable from a broken search.
 */
const IMAGE_MATCH_FLOOR = 0.3;

function ImageSearch({
  patientId,
  timeline,
  onSelect,
}: {
  patientId: string;
  timeline: Timeline;
  onSelect: (ev: EventOut) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<ImageHit[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imaging = timeline.history.filter((e) => e.event_type === "imaging");
  if (imaging.length === 0) return null;

  const run = async () => {
    const q = query.trim();
    if (q.length < 2) return;
    setBusy(true);
    setError(null);
    setHits(null);
    try {
      const res = await api.imageSearch(patientId, q, 5);
      setHits(res.hits ?? []);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Image search failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="imgsearch">
      <button
        type="button"
        className={`imgsearch__toggle${open ? " imgsearch__toggle--on" : ""}`}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span aria-hidden>◐</span> Find a study by what it shows
        <span className="imgsearch__n">{imaging.length}</span>
      </button>

      {open && (
        <div className="imgsearch__panel">
          <form
            className="imgsearch__form"
            onSubmit={(e) => {
              e.preventDefault();
              void run();
            }}
          >
            <input
              className="input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. chest film with an opacity in the lower zone"
              aria-label="Describe what the study shows"
            />
            <button className="btn btn-primary btn-sm" type="submit" disabled={busy || query.trim().length < 2}>
              {busy ? "Matching…" : "Match"}
            </button>
          </form>

          <p className="imgsearch__note">
            Matched against the pixels, not the words typed beside the study.
          </p>

          {error && <div className="imgsearch__err">{error}</div>}

          {hits && hits.length === 0 && !busy && (
            <div className="imgsearch__none">
              No imaging study is stored on this chart yet.
            </div>
          )}

          {hits && hits.length > 0 && hits[0].score < IMAGE_MATCH_FLOOR && (
            <div className="imgsearch__none">
              Nothing on this chart resembles that closely. The nearest study
              scored {(hits[0].score).toFixed(2)}, below the{" "}
              {IMAGE_MATCH_FLOOR.toFixed(2)} needed to count as a match — so it
              is not offered as one.
            </div>
          )}

          {hits && hits.length > 0 && hits[0].score >= IMAGE_MATCH_FLOOR && (
            <ul className="imgsearch__hits">
              {hits.map((h, i) => {
                const ev = findEvent(timeline, h.event_id);
                return (
                  <li key={h.event_id}>
                    <button
                      type="button"
                      className="imgsearch__hit"
                      onClick={() => ev && onSelect(ev)}
                      disabled={!ev}
                    >
                      <span className="imgsearch__rank">{i + 1}</span>
                      <span className="imgsearch__cap">{h.caption}</span>
                      <span className="imgsearch__score" title="Cosine similarity between the description and the stored pixels">
                        {h.score.toFixed(3)}
                      </span>
                      {ev && <ValidityBadge validity={ev.validity} />}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

/* ── timeline pane ──────────────────────────────────────────── */

function TimelinePane({
  patient,
  timeline,
  onSelect,
  onAddRecord,
  onCorrect,
  onUpload,
}: {
  patient: Patient;
  timeline: Timeline | null;
  onSelect: (ev: EventOut) => void;
  onAddRecord: () => void;
  onCorrect: (ev: EventOut) => void;
  onUpload: () => void;
}) {
  if (!timeline) {
    return (
      <div className="pane">
        <div className="pane-head">
          <div className="patient-row">
            <div className="avatar">
              {patient.display_name.charAt(0)}
            </div>
            <div>
              <div className="patient-name">{patient.display_name}</div>
              <div className="patient-meta">
                {patient.age ? `${patient.age} yrs` : "age unknown"}
              </div>
            </div>
          </div>
        </div>
        <div className="pane-body">
          <div className="empty" style={{ margin: "40px auto" }}>
            <div className="spinner" style={{ margin: "0 auto 14px" }} />
            <p>Loading record…</p>
          </div>
        </div>
      </div>
    );
  }

  // Records that are in no correction pair, shown flat.
  const paired = new Set<string>();
  timeline.pairs.forEach((p) => {
    paired.add(p.superseded.id);
    paired.add(p.correction.id);
  });
  const loose = timeline.current.filter((e) => !paired.has(e.id));

  return (
    <div className="pane">
      <div className="pane-head">
        <div className="patient-row">
          <div className="avatar">{patient.display_name.charAt(0)}</div>
          <div>
            <div className="patient-name">{patient.display_name}</div>
            <div className="patient-meta">
              {patient.age ? `${patient.age} yrs` : "age unknown"} ·{" "}
              {timeline.stats.total} events
            </div>
          </div>
        </div>

        <div className="stats-row">
          <div className="stat stat--current">
            <b>{timeline.stats.current}</b> current
          </div>
          <div className="stat stat--superseded">
            <b>{timeline.stats.superseded}</b> superseded
          </div>
          <div className="stat">
            <b>{timeline.stats.corrections}</b> corrections
          </div>
        </div>

        <ImageSearch
          patientId={patient.id}
          timeline={timeline}
          onSelect={onSelect}
        />
      </div>

      <div className="pane-body">
        {timeline.stats.total === 0 ? (
          <div className="empty" style={{ margin: "30px auto" }}>
            <div className="empty-icon">◷</div>
            <h3>Empty record</h3>
            <p>
              Nothing recorded for {patient.display_name.split(" ")[0]} yet. Add
              a lab result, or photograph a report and let SoloPrac read it.
            </p>
            <div className="empty-cta">
              <button className="btn btn-primary btn-sm" onClick={onAddRecord}>
                Add the first record
              </button>
            </div>
          </div>
        ) : (
          <>
        {timeline.pairs.length > 0 && (
          <>
            <div className="group-label">
              <span style={{ color: "var(--superseded)" }}>⊘</span>
              Corrected records
            </div>
            {timeline.pairs.map((pair) => (
              <motion.div
                key={pair.correction.id}
                className="pair"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              >
                <div
                  className="pair-row pair-row--stale"
                  onClick={() => onSelect(pair.superseded)}
                  style={{ cursor: "pointer" }}
                  title="Open the superseded record"
                >
                  <span className="pair-body" style={{ textDecoration: "line-through" }}>
                    {summarise(pair.superseded)}
                  </span>
                  <span className="pair-date">
                    {shortDate(pair.superseded.valid_time)}
                  </span>
                  <ValidityBadge validity="superseded" />
                </div>
                <div className="pair-link">↓ corrected by</div>
                <div
                  className="pair-row pair-row--current"
                  onClick={() => onSelect(pair.correction)}
                  style={{ cursor: "pointer" }}
                  title="Open the current record"
                >
                  <span className="pair-body" style={{ fontWeight: 600 }}>
                    {summarise(pair.correction)}
                  </span>
                  <span className="pair-date">
                    {shortDate(pair.correction.valid_time)}
                  </span>
                  <ValidityBadge validity="current" />
                  <button
                    className="btn btn-ghost btn-sm"
                    style={{ padding: "3px 9px", fontSize: 11 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onCorrect(pair.correction);
                    }}
                    title="Correct this record again"
                  >
                    correct
                  </button>
                </div>
              </motion.div>
            ))}
          </>
        )}

        {loose.length > 0 && (
          <>
            <div className="group-label">
              <span style={{ color: "var(--current)" }}>✓</span>
              Current record
            </div>
            {loose.map((ev, i) => (
              <motion.div
                key={ev.id}
                className="ev"
                onClick={() => onSelect(ev)}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: Math.min(i * 0.03, 0.3), duration: 0.35 }}
                style={{ cursor: "pointer" }}
              >
                <span className="ev-type">{ev.event_type}</span>
                <span className="ev-summary">{summarise(ev)}</span>
                <span className="pair-date">{shortDate(ev.valid_time)}</span>
                <button
                  className="btn btn-ghost btn-sm"
                  style={{ padding: "3px 9px", fontSize: 11 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onCorrect(ev);
                  }}
                  title="Correct this record"
                >
                  correct
                </button>
              </motion.div>
            ))}
          </>
        )}
          </>
        )}
      </div>

      {/* Authoring actions. A doctor adds, corrects, and scans from here. */}
      <div className="fab">
        <div className="fab-item">
          <button
            className="fab-btn fab-btn--ghost"
            onClick={onUpload}
            title="Add a document"
            aria-label="Add a document"
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <path d="M14 2v6h6" />
              <path d="M12 18v-6" />
              <path d="m9 14 3-3 3 3" />
            </svg>
          </button>
          <span className="fab-label" aria-hidden="true">
            Document
          </span>
        </div>
        <div className="fab-item">
          <button
            className="fab-btn"
            onClick={onAddRecord}
            title="Add a record"
            aria-label="Add a record"
          >
            +
          </button>
          <span className="fab-label" aria-hidden="true">
            Record
          </span>
        </div>
      </div>
    </div>
  );
}

/* ── chat pane ──────────────────────────────────────────────── */

interface Turn {
  id: number;
  question: string;
  response: ChatResponse | null;
  error?: string;
}

/** Share of the console the timeline keeps. The remainder goes to chat. */
const DEFAULT_SPLIT = 46.5;
/**
 * Floor on each side. Below this the divider is grabbed rather than dragged,
 * which is a genuinely unpleasant experience and a sign the pane has been
 * collapsed by accident.
 */
const MIN_SPLIT = 22;
/** Where the doctor's chosen split is kept between visits. */
const SPLIT_KEY = "solum.console.split";

function ChatPane({
  patientId,
  timeline,
  onInspect,
  splitRef,
  splitPercent,
  onSplit,
  onSplitting,
}: {
  patientId: string | null;
  timeline: Timeline | null;
  onInspect: (ev: EventOut) => void;
  splitRef: React.RefObject<HTMLDivElement>;
  splitPercent: number;
  onSplit: (percent: number) => void;
  onSplitting: (dragging: boolean) => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [asOf, setAsOf] = useState(false);
  const [sttError, setSttError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const nextId = useRef(1);
  // Text present when dictation starts. Live transcript is appended after
  // it; stopping commits the final text. Typing mid-dictation is allowed
  // but the next live update overwrites it, so the button disables itself
  // against nothing - the doctor dictates, then edits.
  const sttBase = useRef("");

  /* Dictation. Offline, on-device, nothing leaves the browser. */
  const stt = useWhistle({
    onLive: (live) => {
      const base = sttBase.current;
      setInput(base + (base && live ? " " : "") + live);
    },
    onDone: (finalText) => {
      const base = sttBase.current;
      setInput(base + (base && finalText ? " " : "") + finalText);
      sttBase.current = "";
      taRef.current?.focus();
    },
    onError: (message) => setSttError(message),
  });

  const toggleMic = useCallback(() => {
    if (stt.state === "listening" || stt.state === "loading") {
      void stt.stop();
    } else {
      setSttError(null);
      sttBase.current = input;
      void stt.start();
    }
  }, [stt, input]);

  /* ── pane divider ──────────────────────────────────────────────────
     The pointer handlers live here rather than on the divider element so
     the drag keeps tracking after the cursor leaves the 8px handle, which
     is what makes it feel like a divider instead of a slider.

     Pointer capture does the rest: the browser routes every subsequent
     move to this element even when the pointer is over something else, so
     there is no window-level listener to leak and no cleanup to get wrong.
     `touch-action: none` in the CSS stops the browser claiming the gesture
     as a scroll, which would otherwise cancel the drag mid-way on a
     trackpad. */
  const clampSplit = useCallback((pct: number) => {
    return Math.min(100 - MIN_SPLIT, Math.max(MIN_SPLIT, pct));
  }, []);

  // Tracked here and mirrored to the parent, because the drag handlers live
  // in this pane while `data-splitting` and the persisted value belong to
  // the console grid.
  const [dragging, setDragging] = useState(false);
  const setSplitting = onSplitting;
  useEffect(() => {
    setSplitting(dragging);
  }, [dragging, setSplitting]);

  const onMove = useCallback(
    (e: React.PointerEvent) => {
      const host = splitRef.current;
      if (!host) return;
      const box = host.getBoundingClientRect();
      // Zero-width guard: a hidden or mid-transition container would
      // otherwise divide by ~0 and fling the divider off screen.
      if (box.width < 1) return;
      onSplit(clampSplit(((e.clientX - box.left) / box.width) * 100));
    },
    [splitRef, onSplit, clampSplit],
  );

  const endDrag = useCallback((e: React.PointerEvent) => {
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setDragging(false);
  }, []);

  const startDrag = useCallback(
    (e: React.PointerEvent) => {
      // Primary button / single touch only, so a right-click or a
      // two-finger gesture does not start a resize.
      if (e.button !== 0) return;
      e.currentTarget.setPointerCapture?.(e.pointerId);
      setDragging(true);
      onMove(e);
    },
    [onMove],
  );

  const onKey = useCallback(
    (e: React.KeyboardEvent) => {
      const step = e.shiftKey ? 10 : 2;
      let next: number | null = null;
      if (e.key === "ArrowLeft") next = splitPercent - step;
      else if (e.key === "ArrowRight") next = splitPercent + step;
      else if (e.key === "Home") next = MIN_SPLIT;
      else if (e.key === "End") next = 100 - MIN_SPLIT;
      else if (e.key === "Enter" || e.key === " ") next = DEFAULT_SPLIT;
      if (next === null) return;
      e.preventDefault();
      onSplit(clampSplit(next));
    },
    [splitPercent, onSplit, clampSplit],
  );

  useEffect(() => {
    if (!dragging) return;
    const stop = () => setDragging(false);
    // A pointerup outside the handle still ends the drag. Pointer capture
    // covers the normal case, but a lost pointerup (alt-tab, OS gesture)
    // would otherwise leave the divider stuck to the cursor.
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
    window.addEventListener("blur", stop);
    return () => {
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
      window.removeEventListener("blur", stop);
    };
  }, [dragging]);

  useEffect(() => {
    logRef.current?.scrollTo({
      top: logRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns]);

  const ask = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || !patientId) return;

      const id = nextId.current++;
      setTurns((t) => [...t, { id, question: q, response: null }]);
      setInput("");
      setBusy(true);

      try {
        // The as-of window is derived from the record's own dates rather
        // than asked for, so the toggle demonstrates the behaviour without
        // the user having to know a date.
        const asOfIso = asOf
          ? new Date(Date.now() - 1000 * 60 * 60 * 24 * 200).toISOString()
          : undefined;
        const res = await api.chat(patientId, q, asOfIso);
        setTurns((t) =>
          t.map((x) => (x.id === id ? { ...x, response: res } : x)),
        );
      } catch (err) {
        setTurns((t) =>
          t.map((x) =>
            x.id === id
              ? {
                  ...x,
                  error:
                    err instanceof ApiError
                      ? err.message
                      : "Something went wrong.",
                }
              : x,
          ),
        );
      } finally {
        setBusy(false);
      }
    },
    [patientId, asOf],
  );

  const SUGGESTIONS = [
    "What is the current potassium level?",
    "What was the potassium as of last year?",
    "Which medications is the patient currently on?",
    "What is the latest HbA1c?",
    "What diagnoses are on the record?",
  ];

  return (
    <div className="pane pane--right">
      {/* The divider. Drag to move it, or double-click to snap back.

          Keyboard reachable because a mouse-only divider is unusable for
          anyone who cannot drag, and the arrow keys move it in 2% steps with
          Home/End at the clamps. The value is exposed through role and
          aria-valuenow so it is not an unlabelled mystery widget. */}
      <div
        className="splitter"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the record and chat panes"
        aria-valuenow={Math.round(splitPercent)}
        aria-valuemin={MIN_SPLIT}
        aria-valuemax={100 - MIN_SPLIT}
        tabIndex={0}
        onDoubleClick={() => onSplit(DEFAULT_SPLIT)}
        onPointerDown={startDrag}
        onPointerMove={dragging ? onMove : undefined}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onLostPointerCapture={endDrag}
        onKeyDown={onKey}
      >
        <span className="splitter-grip" aria-hidden="true" />
      </div>

      <div className="pane-head">
        <div className="patient-row">
          <div className="brand-mark" style={{ width: 26, height: 26, fontSize: 13 }}>
            ?
          </div>
          <div>
            <div className="patient-name">Ask the record</div>
            <div className="patient-meta">
              Answers are grounded, cited, and validity-checked
            </div>
          </div>
        </div>
      </div>

      <div className="chat-log" ref={logRef}>
        {turns.length === 0 ? (
          <div className="empty">
            <div className="empty-icon">◈</div>
            <h3>Ask anything about this patient</h3>
            <p>
              Every answer cites the records it used and marks whether each one
              is current or has been corrected. Nothing identifiable leaves
              this machine.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className="suggestion"
                  onClick={() => ask(s)}
                  disabled={!patientId}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {turns.map((t) => (
              <motion.div
                key={t.id}
                className="msg"
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
              >
                <div className="msg-role">You asked</div>
                <div className="bubble bubble--user">{t.question}</div>

                {t.error ? (
                  <>
                    <div className="msg-role" style={{ color: "var(--danger)" }}>
                      Error
                    </div>
                    <div
                      className="bubble bubble--ai"
                      style={{ color: "var(--danger)" }}
                    >
                      {t.error}
                    </div>
                  </>
                ) : t.response ? (
                  <>
                    <div className="msg-role" style={{ color: "var(--current)" }}>
                      Grounded answer
                    </div>
                    <div className="bubble bubble--ai">
                      <RichText text={t.response.answer} />
                      {!t.response.answer && "No records matched that question."}
                    </div>

                    {/* The "we caught it" panel - the whole product in
                        one component. */}
                    {t.response.suppressed.length > 0 && (
                      <div className="caught">
                        <b>⚠ Blocked — superseded value</b>
                        {t.response.suppressed.map((s, i) => (
                          <div key={i}>
                            &ldquo;{s.text}&rdquo; — {s.reason}
                          </div>
                        ))}
                      </div>
                    )}

                    {t.response.citations.length > 0 && (
                      <div className="cites">
                        {t.response.citations.map((c) => {
                          // Prefer the record's own summary: the API's
                          // `text` field is the raw payload, which reads
                          // as "flag: high, test: HbA1c, unit: %, v...".
                          const match = findEvent(timeline, c.event_id);
                          const label = match
                            ? summarise(match)
                            : c.text;
                          return (
                            <button
                              key={c.event_id}
                              className={`cite cite--${c.validity}`}
                              // A citation whose record is not in the loaded
                              // timeline cannot be opened. It used to render
                              // fully interactive and swallow the click, so it
                              // now says so instead of pretending.
                              disabled={!match}
                              aria-disabled={!match}
                              onClick={() => {
                                if (match) onInspect(match);
                              }}
                              title={
                                match
                                  ? `Open this record · ${fmtDate(match.valid_time)}`
                                  : `${label} · not in the loaded timeline`
                              }
                            >
                              <span className="dot" />
                              <span className="cite-text">{label}</span>
                            </button>
                          );
                        })}
                      </div>
                    )}

                    <div className="answer-meta">
                      <span className="badge badge-neutral">
                        {t.response.intent}
                      </span>
                      <span className="badge badge-neutral">
                        {Math.round(t.response.latency_ms)} ms
                      </span>
                      <span className="badge badge-neutral">
                        {t.response.citations.length} citation
                        {t.response.citations.length === 1 ? "" : "s"}
                      </span>
                      {t.response.degraded && (
                        <span className="badge badge-superseded">
                          model unavailable · extractive
                        </span>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="answer-meta">
                    <span className="spinner" />
                    <span style={{ fontSize: 12, color: "var(--text-3)" }}>
                      retrieving, verifying…
                    </span>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>

      <div className="composer">
        <div className="composer-row">
          <textarea
            ref={taRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(input);
              }
            }}
            placeholder="Ask about this patient's record…"
            rows={1}
            disabled={!patientId}
          />
          <button
            className={`mic${stt.state === "listening" ? " on" : ""}`}
            onClick={toggleMic}
            disabled={!patientId || stt.state === "loading"}
            aria-label={stt.state === "listening" ? "Stop dictation" : "Dictate (offline)"}
            title={
              stt.state === "listening"
                ? "Stop dictation"
                : stt.state === "loading"
                  ? "Loading speech engine…"
                  : "Dictate — runs on this device, nothing is uploaded"
            }
          >
            {stt.state === "loading" ? (
              <span className="spinner" />
            ) : (
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <rect x="9" y="2" width="6" height="12" rx="3" />
                <path d="M5 10a7 7 0 0 0 14 0" />
                <path d="M12 17v4" />
              </svg>
            )}
          </button>
          <button
            className="send"
            onClick={() => ask(input)}
            disabled={busy || !input.trim() || !patientId}
            aria-label="Send"
          >
            {busy ? <span className="spinner" /> : "↑"}
          </button>
        </div>
        <div className="composer-foot">
          <span>
            <span className="kbd">Enter</span> to send ·{" "}
            <span className="kbd">Shift+Enter</span> for a new line
            {stt.state === "listening" && (
              <span className="stt-live"> · listening… tap mic to stop</span>
            )}
            {sttError && <span className="stt-err"> · {sttError}</span>}
          </span>
          <label className="asof-toggle">
            <input
              type="checkbox"
              checked={asOf}
              onChange={(e) => setAsOf(e.target.checked)}
            />
            answer as of ~200 days ago
          </label>
        </div>
      </div>
    </div>
  );
}

/* ── page ───────────────────────────────────────────────────── */

// The same three destinations as the dashboard, and in the same order.
//
// This page used to hand-roll its own `<nav>` instead of using AppShell,
// which is why it had no nav links, no notification bell and no account
// menu: a doctor's only way back to the dashboard from Records was to
// edit the URL. One shell, one nav.
const NAV = [
  { href: "/dashboard", label: "Overview" },
  { href: "/app", label: "Records" },
  { href: "/settings", label: "Settings" },
];

export default function Console({
  initialPatient,
}: {
  /** Deep link from the dashboard: /app?patient=<uuid>. */
  initialPatient?: string | null;
}) {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selected, setSelected] = useState<string | null>(
    initialPatient ?? null,
  );
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  /** The record currently open in the inspector. */
  const [inspect, setInspect] = useState<EventOut | null>(null);

  /* ── the draggable divider ──────────────────────────────────────────
     Read once on mount so a doctor who set a comfortable split keeps it
     across sessions. Storage is wrapped because it throws in private
     browsing and when a quota is exhausted, and a pane divider is not
     worth taking the console down for. */
  const [split, setSplit] = useState(() => {
    try {
      const raw = window.localStorage.getItem(SPLIT_KEY);
      const n = raw === null ? NaN : Number(raw);
      return Number.isFinite(n)
        ? Math.min(100 - MIN_SPLIT, Math.max(MIN_SPLIT, n))
        : DEFAULT_SPLIT;
    } catch {
      return DEFAULT_SPLIT;
    }
  });
  const [splitting, setSplitting] = useState(false);
  const splitRef = useRef<HTMLDivElement>(null);

  // Persist on release, not on every pointer move: a drag emits dozens of
  // updates and localStorage is synchronous.
  useEffect(() => {
    if (splitting) return;
    try {
      window.localStorage.setItem(SPLIT_KEY, String(Math.round(split)));
    } catch {
      /* nothing to do; the split just will not be remembered */
    }
  }, [split, splitting]);

  // Reset when the window is too narrow to have two panes. The CSS already
  // stacks them, and carrying a 46% split into a stacked layout would
  // squash the timeline to a sliver once the window grew again.
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 1080px)");
    const apply = () => {
      if (mq.matches) setSplit(DEFAULT_SPLIT);
    };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  // authoring sheets
  const [newPatient, setNewPatient] = useState(false);
  const [newRecord, setNewRecord] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [correcting, setCorrecting] = useState<EventOut | null>(null);
  const [composing, setComposing] = useState(false);
  const [weekly, setWeekly] = useState(false);
  const [certifying, setCertifying] = useState(false);
  const [invoicing, setInvoicing] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const flash = useCallback((msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 3200);
  }, []);

  /** Reload the timeline after any write, so the view is never stale. */
  const refresh = useCallback(async () => {
    if (!selected) return;
    try {
      setTimeline(await api.timeline(selected));
    } catch {
      /* leave the previous timeline in place rather than blanking the pane */
    }
  }, [selected]);

  const onSaved = useCallback(
    async (msg: string) => {
      await refresh();
      flash(msg);
    },
    [refresh, flash],
  );

  useEffect(() => {
    api
      .listPatients()
      .then((list) => {
        setPatients(list);
        // A successful load clears whatever the last one complained about.
        // This banner is shared state, and an empty practice is not an
        // error worth carrying forward: without this, adding the very first
        // patient left "No patients yet. Add one to get started." sitting
        // above the patient who had just been added and their record.
        setError(null);

        if (list.length === 0) {
          setError("No patients yet. Add one to get started.");
          return;
        }
        // Honour a deep link, but only if it names a patient this
        // doctor can actually see. Trusting the query string blindly
        // would open an empty pane for an id from another practice.
        const wanted =
          initialPatient && list.some((p) => p.id === initialPatient)
            ? initialPatient
            : list[0].id;
        setSelected(wanted);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : "API unreachable"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) {
      setTimeline(null);
      return;
    }
    setTimeline(null);
    api
      .timeline(selected)
      // Clearing first, so a successful load cancels a previous failure -
      // the same shared-state problem as the patient list above.
      .then((tl) => {
        setTimeline(tl);
        setError(null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load"));
  }, [selected]);

  const patient = patients.find((p) => p.id === selected) ?? null;

  return (
    <AppShell items={NAV}>
      <div className="console-bar">
        {patients.length > 0 && (
          <select
            className="patient-picker"
            value={selected ?? ""}
            onChange={(e) => setSelected(e.target.value)}
            aria-label="Choose a patient"
          >
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.display_name}
              </option>
            ))}
          </select>
        )}
        <button
          className="btn btn-ghost btn-sm"
          onClick={() => setNewPatient(true)}
        >
          + Patient
        </button>

        {/* Document actions, scoped to the patient on screen. These were
            reachable only from the dashboard, where the doctor had to pick
            the patient again from a dropdown even though a patient was
            already selected here. */}
        {patient && (
          <div className="doc-actions">
            <span className="doc-actions-label">For {patient.display_name}</span>
            <button
              className="btn btn-primary btn-sm"
              onClick={() => setComposing(true)}
            >
              New prescription
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setWeekly(true)}
              disabled={!timeline || timeline.stats.total === 0}
              title={
                timeline && timeline.stats.total === 0
                  ? "This patient has no records to report on yet"
                  : undefined
              }
            >
              Weekly report
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setCertifying(true)}
            >
              Certificate
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setInvoicing(true)}
            >
              Invoice
            </button>
          </div>
        )}
      </div>

      {error && <div className="banner">⚠ {error}</div>}

        {loading ? (
          <div className="empty" style={{ paddingTop: 120 }}>
            <div className="spinner" style={{ margin: "0 auto 14px" }} />
            <p>Connecting to the SoloPrac API…</p>
          </div>
        ) : patient ? (
          <div
            className="console"
            ref={splitRef}
            style={{ ["--split" as string]: `${split}%` }}
            data-splitting={splitting ? "true" : undefined}
          >
            <TimelinePane
              patient={patient}
              timeline={timeline}
              onSelect={setInspect}
              onAddRecord={() => setNewRecord(true)}
              onCorrect={(ev) => setCorrecting(ev)}
              onUpload={() => setUploading(true)}
            />
            <ChatPane
              patientId={selected}
              timeline={timeline}
              onInspect={setInspect}
              splitRef={splitRef}
              splitPercent={split}
              onSplit={setSplit}
              onSplitting={setSplitting}
            />
          </div>
        ) : (
          <div className="empty" style={{ paddingTop: 120 }}>
            <div className="empty-icon">◈</div>
            <h3>No patients yet</h3>
            <p>Add a patient to start a record.</p>
            <div className="empty-cta">
              <button
                className="btn btn-primary btn-sm"
                onClick={() => setNewPatient(true)}
              >
                Add the first patient
              </button>
            </div>
          </div>
        )}

        {/* ── authoring sheets ── */}
        <AnimatePresence>
          {newPatient && (
            <NewPatientSheet
              onClose={() => setNewPatient(false)}
              onCreated={async (p) => {
                setPatients((list) => [p, ...list]);
                // The practice is no longer empty. Leaving the old message
                // up here is what put "No patients yet" directly above the
                // patient the doctor had just created.
                setError(null);
                setSelected(p.id);
                setTimeline(null);
                flash(`${p.display_name} added.`);
              }}
            />
          )}

          {/* Document generation, scoped to the patient on screen. */}
          {composing && selected && (
            <PrescriptionComposer
              initialPatientId={selected}
              onClose={() => setComposing(false)}
              onSent={() => {
                setComposing(false);
                flash("Prescription issued.");
              }}
            />
          )}

          {weekly && selected && (
            <WeeklyReportSheet
              initialPatientId={selected}
              onClose={() => setWeekly(false)}
              onSent={(ref) => {
                setWeekly(false);
                flash(`Report ${ref} ready.`);
              }}
            />
          )}

          {certifying && selected && (
            <CertificateSheet
              initialPatientId={selected}
              onClose={() => setCertifying(false)}
              onSent={(ref) => {
                setCertifying(false);
                flash(`Certificate ${ref} issued.`);
              }}
            />
          )}

          {invoicing && selected && (
            <InvoiceSheet
              initialPatientId={selected}
              onClose={() => setInvoicing(false)}
              onSent={(ref) => {
                setInvoicing(false);
                flash(`Invoice ${ref} created.`);
              }}
            />
          )}

          {newRecord && selected && (
            <NewRecordSheet
              patientId={selected}
              onClose={() => setNewRecord(false)}
              onSaved={onSaved}
            />
          )}

          {correcting && selected && (
            <NewRecordSheet
              patientId={selected}
              correcting={correcting}
              onClose={() => setCorrecting(null)}
              onSaved={onSaved}
            />
          )}

          {uploading && selected && (
            <CaptureSheet
              patientId={selected}
              onClose={() => setUploading(false)}
              onSaved={onSaved}
            />
          )}

          {toast && (
            <motion.div
              className="toast"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
            >
              <span className="dot" />
              {toast}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Time-travel inspector. Opens from a citation chip or any
            record in the timeline, and can walk the version chain. */}
        <RecordDrawer
          event={inspect}
          timeline={timeline}
          onClose={() => setInspect(null)}
          onJump={(id) => {
            const next = findEvent(timeline, id);
            if (next) setInspect(next);
          }}
        />
    </AppShell>
  );
}
