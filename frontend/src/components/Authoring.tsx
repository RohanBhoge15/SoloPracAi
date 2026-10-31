import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, ApiError, type EventOut, type Patient } from "@/lib/api";
import "./authoring.css";

/** The one-line summary of a record, for the "replacing" preview. */
function summariseFor(ev: EventOut): string {
  const p = ev.payload as Record<string, unknown>;
  switch (ev.event_type) {
    case "lab":
      return `${p.test ?? "Lab"}: ${p.value ?? "?"} ${p.unit ?? ""}`.trim();
    case "medication":
      return `${p.drug ?? "?"} ${p.dose ?? ""}`.trim();
    case "diagnosis":
      return String(p.diagnosis ?? "diagnosis");
    case "vital":
      return `${p.measurement ?? "?"}: ${p.value ?? "?"}`.trim();
    case "imaging":
      return String(p.study ?? "imaging");
    default:
      return String(p.note ?? ev.event_type);
  }
}

/* ── shared shell ────────────────────────────────────────────── */

export function Sheet({
  title,
  subtitle,
  onClose,
  children,
  footer,
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  // Portaled to <body> so the sheet escapes the page's stacking context.
  // Rendered inline, the sticky topbar (z-index 50) sat on top of the
  // scrim (z-index 200) because the page created a stacking context
  // underneath it — the sheet slid in *behind* the navbar.
  return createPortal(
    <>
      <motion.div
        className="sheet-scrim"
        onClick={onClose}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.18 }}
      />
      <motion.aside
        className="sheet"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", damping: 32, stiffness: 320 }}
      >
        <div className="sheet-head">
          <div className="drawer-top">
            <span className="drawer-title">{title}</span>
            <button className="drawer-close" onClick={onClose} aria-label="Close">
              ✕
            </button>
          </div>
          {subtitle && <div className="sheet-sub">{subtitle}</div>}
        </div>
        <div className="sheet-body">{children}</div>
        {footer && <div className="sheet-foot">{footer}</div>}
      </motion.aside>
    </>,
    document.body
  );
}

function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="err">⚠ {error}</div>;
}

/* ── new patient ─────────────────────────────────────────────── */

export function NewPatientSheet({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (p: Patient) => void;
}) {
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [phone, setPhone] = useState("");
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!name.trim()) {
      setError("A name is required.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const created = await api.createPatient({
        display_name: name.trim(),
        age: age ? Number(age) : null,
        phone: phone.trim() || null,
        // Optional, and the field that makes identity linking work. Omitted
        // here is a walk-in with no portal access, which is a valid choice.
        email: email.trim() || null,
      });
      onCreated(created);
      onClose();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not create patient.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet
      title="New patient"
      subtitle="Start an empty record. Everything can be added and corrected later."
      onClose={onClose}
      footer={
        <>
          <button
            className="btn btn-primary btn-sm"
            onClick={submit}
            disabled={busy || !name.trim()}
          >
            {busy ? <span className="spinner" /> : "Create patient"}
          </button>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            Cancel
          </button>
        </>
      }
    >
      <ErrorBox error={error} />
      <div className="field">
        <label className="field-label">Full name *</label>
        <input
          className="input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Rakesh Mehta"
          autoFocus
        />
      </div>
      <div className="field-row">
        <div className="field">
          <label className="field-label">Age</label>
          <input
            className="input"
            type="number"
            min={0}
            max={130}
            value={age}
            onChange={(e) => setAge(e.target.value)}
            placeholder="54"
          />
        </div>
        <div className="field">
          <label className="field-label">Phone</label>
          <input
            className="input"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            placeholder="9876543210"
          />
        </div>
      </div>
      <div className="field">
        <label className="field-label">Email</label>
        <input
          className="input"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="patient@example.com"
        />
        <div className="field-hint">
          If they sign up with this address, they find this record on their
          own — no invite, no code. Leave blank for a walk-in who will not
          use the portal.
        </div>
      </div>
      <div className="callout callout--info">
        <b>What happens next</b>
        Add labs, medications, and notes. When a value is later corrected,
        the original is kept and marked superseded — nothing is ever
        overwritten.
      </div>
    </Sheet>
  );
}

/* ── new record ──────────────────────────────────────────────── */

const TYPES = ["lab", "medication", "diagnosis", "vital", "note", "imaging"] as const;
type Type = (typeof TYPES)[number];

/** Every form is a flat string map, so one shape serves all six types. */
type Fields = Record<string, string>;

function defaults(t: Type): Fields {
  switch (t) {
    case "lab":
      return { test: "", value: "", unit: "", refLow: "", refHigh: "" };
    case "medication":
      return { drug: "", dose: "", frequency: "twice daily", status: "active", reason: "" };
    case "diagnosis":
      return { diagnosis: "", icd10: "" };
    case "vital":
      return { measurement: "", value: "", unit: "" };
    case "imaging":
      return { study: "", impression: "" };
    default:
      return { note: "", author: "" };
  }
}

/** Build the JSON payload the API expects for each record type. */
function buildPayload(t: Type, f: Fields) {
  const num = (v: string) => (v.trim() === "" ? null : Number(v));
  switch (t) {
    case "lab": {
      const value = num(f.value);
      if (value === null || !f.test.trim())
        throw new Error("A test name and a value are required.");
      const low = num(f.refLow);
      const high = num(f.refHigh);
      return {
        test: f.test.trim(),
        value,
        unit: f.unit.trim() || null,
        ...(low !== null && high !== null
          ? { ref_low: low, ref_high: high, flag: value < low ? "low" : value > high ? "high" : "normal" }
          : {}),
      };
    }
    case "medication": {
      if (!f.drug.trim()) throw new Error("A drug name is required.");
      return {
        drug: f.drug.trim(),
        dose: f.dose.trim(),
        frequency: f.frequency.trim(),
        status: f.status,
        ...(f.status === "stopped" && f.reason.trim() ? { reason: f.reason.trim() } : {}),
      };
    }
    case "diagnosis": {
      if (!f.diagnosis.trim()) throw new Error("A diagnosis is required.");
      return { diagnosis: f.diagnosis.trim(), icd10: f.icd10.trim() || null };
    }
    case "vital": {
      if (!f.measurement.trim() || f.value.trim() === "")
        throw new Error("A measurement and a value are required.");
      return { measurement: f.measurement.trim(), value: f.value.trim(), unit: f.unit.trim() || null };
    }
    case "imaging": {
      if (!f.study.trim()) throw new Error("A study name is required.");
      return { study: f.study.trim(), impression: f.impression.trim() };
    }
    default: {
      if (!f.note.trim()) throw new Error("A note is required.");
      return { note: f.note.trim(), author: f.author.trim() || "clinician" };
    }
  }
}

export function NewRecordSheet({
  patientId,
  correcting,
  onClose,
  onSaved,
}: {
  patientId: string;
  /** When set, the new record supersedes this one instead of standing alone. */
  correcting?: EventOut | null;
  onClose: () => void;
  onSaved: (msg: string) => void;
}) {
  const [type, setType] = useState<Type>("lab");
  const [fields, setFields] = useState<Fields>(defaults("lab"));
  const [when, setWhen] = useState(() => new Date().toISOString().slice(0, 10));
  const [amendment, setAmendment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pre-fill the correction form with the record being replaced, so the
  // clinician edits a value rather than retyping it.
  useEffect(() => {
    if (!correcting) return;
    const p = correcting.payload as Record<string, unknown>;
    switch (correcting.event_type) {
      case "lab":
        setType("lab");
        setFields({
          test: String(p.test ?? ""),
          value: p.value === undefined ? "" : String(p.value),
          unit: String(p.unit ?? ""),
          refLow: p.ref_low === undefined ? "" : String(p.ref_low),
          refHigh: p.ref_high === undefined ? "" : String(p.ref_high),
        });
        break;
      case "medication":
        setType("medication");
        setFields({
          drug: String(p.drug ?? ""),
          dose: String(p.dose ?? ""),
          frequency: String(p.frequency ?? ""),
          status: "stopped",
          reason: String(p.reason ?? ""),
        });
        break;
      case "diagnosis":
        setType("diagnosis");
        setFields({ diagnosis: String(p.diagnosis ?? ""), icd10: String(p.icd10 ?? "") });
        break;
      case "vital":
        setType("vital");
        setFields({
          measurement: String(p.measurement ?? ""),
          value: String(p.value ?? ""),
          unit: String(p.unit ?? ""),
        });
        break;
      case "imaging":
        setType("imaging");
        setFields({ study: String(p.study ?? ""), impression: String(p.impression ?? "") });
        break;
      default:
        setType("note");
        setFields({ note: String(p.note ?? ""), author: String(p.author ?? "") });
    }
  }, [correcting]);

  const switchType = (t: Type) => {
    setType(t);
    setFields(defaults(t));
    setError(null);
  };

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const payload = buildPayload(type, fields);
      if (correcting && amendment.trim()) {
        (payload as Record<string, unknown>).amendment = amendment.trim();
      }
      await api.createEvent(patientId, {
        event_type: type,
        valid_time: new Date(when).toISOString(),
        payload,
        source_type: correcting ? "correction" : "manual",
        supersedes_id: correcting?.id ?? null,
      });
      onSaved(
        correcting
          ? `Correction recorded. The previous value is now marked superseded.`
          : "Record added.",
      );
      onClose();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not save the record.");
    } finally {
      setBusy(false);
    }
  };

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) =>
    setFields((f) => ({ ...f, [k]: e.target.value }));

  return (
    <Sheet
      title={correcting ? "Correct a record" : "Add a record"}
      subtitle={
        correcting
          ? "The original is kept. It stays on the timeline, marked superseded, and remains auditable."
          : "Appended to the record. Nothing is ever overwritten."
      }
      onClose={onClose}
      footer={
        <>
          <button className="btn btn-primary btn-sm" onClick={save} disabled={busy}>
            {busy ? <span className="spinner" /> : correcting ? "Save correction" : "Save record"}
          </button>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            Cancel
          </button>
        </>
      }
    >
      <ErrorBox error={error} />

      {correcting && (
        <div className="supersede-target">
          <div className="supersede-label">Replacing</div>
          <div className="supersede-value">{summariseFor(correcting)}</div>
          <div className="supersede-note">
            recorded{" "}
            {new Date(correcting.valid_time).toLocaleDateString("en-GB")} ·{" "}
            {correcting.event_type} · id {correcting.id.slice(0, 8)}
          </div>
        </div>
      )}

      {!correcting && (
        <div className="field">
          <label className="field-label">Record type</label>
          <div className="type-picker">
            {TYPES.map((t) => (
              <button
                key={t}
                className={`type-opt${type === t ? " type-opt--on" : ""}`}
                onClick={() => switchType(t)}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      )}

      {type === "lab" && (
        <>
          <div className="field">
            <label className="field-label">Test name *</label>
            <input className="input" value={fields.test} onChange={set("test")} placeholder="Potassium" autoFocus />
          </div>
          <div className="field-row--3 field field-row">
            <div className="field">
              <label className="field-label">Value *</label>
              <input className="input" value={fields.value} onChange={set("value")} placeholder="4.8" />
            </div>
            <div className="field">
              <label className="field-label">Unit</label>
              <input className="input" value={fields.unit} onChange={set("unit")} placeholder="mEq/L" />
            </div>
            <div className="field">
              <label className="field-label">Ref. low</label>
              <input className="input" value={fields.refLow} onChange={set("refLow")} placeholder="3.5" />
            </div>
          </div>
          <div className="field">
            <label className="field-label">Ref. high</label>
            <input className="input" value={fields.refHigh} onChange={set("refHigh")} placeholder="5.1" />
            <div className="field-hint">
              Both bounds are needed to flag the result as low or high.
            </div>
          </div>
        </>
      )}

      {type === "medication" && (
        <>
          <div className="field">
            <label className="field-label">Drug *</label>
            <input className="input" value={fields.drug} onChange={set("drug")} placeholder="Metformin" autoFocus />
          </div>
          <div className="field-row">
            <div className="field">
              <label className="field-label">Dose</label>
              <input className="input" value={fields.dose} onChange={set("dose")} placeholder="1000 mg" />
            </div>
            <div className="field">
              <label className="field-label">Frequency</label>
              <input className="input" value={fields.frequency} onChange={set("frequency")} placeholder="twice daily" />
            </div>
          </div>
          <div className="field">
            <label className="field-label">Status</label>
            <select className="select" value={fields.status} onChange={set("status")}>
              <option value="active">Active</option>
              <option value="stopped">Stopped</option>
            </select>
          </div>
          {fields.status === "stopped" && (
            <div className="field">
              <label className="field-label">Reason for stopping</label>
              <input className="input" value={fields.reason} onChange={set("reason")} placeholder="eGFR 24, switched to insulin" />
            </div>
          )}
        </>
      )}

      {type === "diagnosis" && (
        <>
          <div className="field">
            <label className="field-label">Diagnosis *</label>
            <input className="input" value={fields.diagnosis} onChange={set("diagnosis")} placeholder="Type 2 Diabetes Mellitus" autoFocus />
          </div>
          <div className="field">
            <label className="field-label">ICD-10</label>
            <input className="input" value={fields.icd10} onChange={set("icd10")} placeholder="E11.9" />
          </div>
        </>
      )}

      {type === "vital" && (
        <div className="field-row--3 field field-row">
          <div className="field">
            <label className="field-label">Measurement *</label>
            <input className="input" value={fields.measurement} onChange={set("measurement")} placeholder="Blood pressure" autoFocus />
          </div>
          <div className="field">
            <label className="field-label">Value *</label>
            <input className="input" value={fields.value} onChange={set("value")} placeholder="128/82" />
          </div>
          <div className="field">
            <label className="field-label">Unit</label>
            <input className="input" value={fields.unit} onChange={set("unit")} placeholder="mmHg" />
          </div>
        </div>
      )}

      {type === "imaging" && (
        <>
          <div className="field">
            <label className="field-label">Study *</label>
            <input className="input" value={fields.study} onChange={set("study")} placeholder="Chest X-ray" autoFocus />
          </div>
          <div className="field">
            <label className="field-label">Impression</label>
            <textarea className="textarea" value={fields.impression} onChange={set("impression")} />
          </div>
        </>
      )}

      {type === "note" && (
        <>
          <div className="field">
            <label className="field-label">Note *</label>
            <textarea className="textarea" value={fields.note} onChange={set("note")} placeholder="Reports occasional dizziness on standing." autoFocus />
          </div>
          <div className="field">
            <label className="field-label">Author</label>
            <input className="input" value={fields.author} onChange={set("author")} placeholder="Dr. Anita Desai" />
          </div>
        </>
      )}

      {correcting && (
        <div className="field">
          <label className="field-label">Why is it being corrected?</label>
          <input
            className="input"
            value={amendment}
            onChange={(e) => setAmendment(e.target.value)}
            placeholder="Specimen re-run; original transcription error"
          />
          <div className="field-hint">
            Shown to the clinician alongside the old value. This is the medico-legal
            explanation for the change.
          </div>
        </div>
      )}

      <div className="field">
        <label className="field-label">Date of the finding</label>
        <input className="input" type="date" value={when} onChange={(e) => setWhen(e.target.value)} />
        <div className="field-hint">
          When it was clinically true, not when it was typed. This is what lets
          &ldquo;what did we know then&rdquo; be answered.
        </div>
      </div>
    </Sheet>
  );
}

/* ── document capture ────────────────────────────────────────── */

export function CaptureSheet({
  patientId,
  onClose,
  onSaved,
}: {
  patientId: string;
  onClose: () => void;
  onSaved: (msg: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.capture>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Which kind of document this is decides which gate the upload meets.
  //
  // A lab report is *read*: values are extracted and committed only above a
  // confidence threshold. An imaging study is *filed*: there are no numbers to
  // misread, so it is committed on readability and its pixels are indexed so
  // the study can be found by what it shows. Sending a radiograph down the lab
  // path returns "no lab rows detected" and files nothing.
  const [kind, setKind] = useState<"lab" | "imaging">("lab");
  const [caption, setCaption] = useState("");
  const [findings, setFindings] = useState("");
  const [studyDate, setStudyDate] = useState("");

  const isImaging = kind === "imaging";

  const upload = async (f: File) => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.capture(patientId, f, {
        eventType: kind,
        caption: isImaging ? caption.trim() : undefined,
        findings: isImaging ? findings.trim() : undefined,
        // A date-only input yields no timezone; the API reads it as UTC.
        validTime: isImaging && studyDate ? `${studyDate}T00:00:00Z` : undefined,
      });
      setResult(res);
      if (res.committed) onSaved(`Record added from ${f.name}.`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  };

  const pct = result ? Math.round(result.confidence * 100) : 0;
  const rows = (result?.extracted?.rows ?? []) as Array<Record<string, unknown>>;
  const indexed = Boolean((result?.extracted as { indexed?: boolean } | undefined)?.indexed);

  return (
    <Sheet
      title="Add a document"
      subtitle={
        isImaging
          ? "Upload a radiograph or scan. The pixels are indexed, so the study can be found later by what it shows."
          : "Photograph or scan a lab report. SoloPrac refuses to commit anything it cannot read reliably."
      }
      onClose={onClose}
      footer={
        <button className="btn btn-ghost btn-sm" onClick={onClose}>
          Close
        </button>
      }
    >
      <ErrorBox error={error} />

      <div className="seg" role="tablist" aria-label="Document kind">
        {(
          [
            ["lab", "Lab report"],
            ["imaging", "Imaging study"],
          ] as ["lab" | "imaging", string][]
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={kind === value}
            className={`seg__btn ${kind === value ? "on" : ""}`}
            onClick={() => {
              setKind(value);
              setResult(null);
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {isImaging && (
        <div className="img-fields">
          <label className="field">
            <span className="field-label">What is the study?</span>
            <input
              className="input"
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              placeholder="Chest X-Ray PA view"
            />
          </label>
          <label className="field">
            <span className="field-label">Findings (optional)</span>
            <textarea
              className="input"
              rows={3}
              value={findings}
              onChange={(e) => setFindings(e.target.value)}
              placeholder="Bilateral lower zone airspace opacities."
            />
          </label>
          <label className="field">
            <span className="field-label">Date of the study</span>
            <input
              className="input"
              type="date"
              value={studyDate}
              onChange={(e) => setStudyDate(e.target.value)}
            />
            <span className="field-hint">
              Leave blank to file it as today. An old film dated today becomes the
              most recent imaging, and a present-tense question returns it as
              current.
            </span>
          </label>
        </div>
      )}

      <div
        className={`drop${over ? " drop--over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) {
            setFile(f);
            upload(f);
          }
        }}
      >
        <div className="drop-icon">⬡</div>
        <div className="drop-title">{file ? file.name : "Drop a scan, or click to choose"}</div>
        <div className="drop-hint">JPEG, PNG, WebP or TIFF · up to 12 MB</div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) {
              setFile(f);
              upload(f);
            }
          }}
        />
      </div>

      {busy && (
        <div className="preview">
          <div className="preview-title">{isImaging ? "Filing" : "Reading"}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="spinner" />
            <span style={{ color: "var(--text-3)" }}>
              {isImaging ? "indexing the image…" : "extracting values…"}
            </span>
          </div>
        </div>
      )}

      {result && (
        <>
          {/* The confidence meter is an OCR number. For an image there is no
              extraction and nothing to be unsure about, so showing "100%"
              against a "75% threshold" would be theatre. What actually
              matters is whether the pixels were indexed, because an
              unindexed study is stored but unfindable by content. */}
          {isImaging ? (
            <div className="preview">
              <div className="preview-title">Study filed</div>
              <div style={{ marginTop: 4 }}>
                {result.committed ? (
                  <span className="badge badge-current">
                    <span className="dot" /> added to the record
                  </span>
                ) : (
                  <span className="badge badge-superseded">
                    <span className="dot" /> held for review
                  </span>
                )}
              </div>
              <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-3)" }}>
                {result.committed && indexed
                  ? "Its pixels are indexed, so this study can be found by what it shows."
                  : "Stored, but not indexed for image search on this deployment."}
              </div>
            </div>
          ) : (
            <div className="preview">
              <div className="preview-title">Extraction confidence</div>
              <div className={`meter-fill meter-fill--${result.committed ? "ok" : "low"}`} style={{ width: `${pct}%` }} />
              <div className="meter-legend">
                <span>{pct}%</span>
                <span>threshold 75%</span>
              </div>
              <div style={{ marginTop: 10 }}>
                {result.committed ? (
                  <span className="badge badge-current">
                    <span className="dot" /> committed to the record
                  </span>
                ) : (
                  <span className="badge badge-superseded">
                    <span className="dot" /> held for review
                  </span>
                )}
              </div>
            </div>
          )}

          {rows.length > 0 && (
            <div className="preview">
              <div className="preview-title">
                {rows.length} value{rows.length === 1 ? "" : "s"} found
              </div>
              {rows.map((r, i) => (
                <div key={i} style={{ display: "flex", gap: 10, padding: "3px 0" }}>
                  <span style={{ flex: 1, color: "var(--text-2)" }}>{String(r.test)}</span>
                  <span className="mono" style={{ color: "var(--text)" }}>
                    {String(r.value)} {String(r.unit ?? "")}
                  </span>
                  {typeof r.flag === "string" && r.flag !== "normal" && (
                    <span className="badge badge-danger" style={{ fontSize: 9 }}>
                      {r.flag}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}

          {!result.committed && (
            <div className="callout callout--warn" style={{ marginTop: 16 }}>
              <b>Not written to the record</b>
              {result.reason ??
                "Extraction confidence was below the threshold, so nothing was committed. A wrong value in a permanent record is worse than an unread one."}
            </div>
          )}
        </>
      )}

      {!result && !busy && (
        <div className="callout callout--info" style={{ marginTop: 18 }}>
          {isImaging ? (
            <>
              <b>What happens to an image</b>
              The film is filed as a record in its own right and its pixels are
              embedded, so you can ask later for &ldquo;the study showing a
              right lower zone opacity&rdquo; and have it found by what it
              shows rather than by the words typed next to it. Nothing is read
              off the picture, so nothing can be misread.
            </>
          ) : (
            <>
              <b>Why it refuses sometimes</b>
              A blurry or unfamiliar report yields low-confidence numbers. Committing
              those would put a wrong value into a permanent medical record, so the
              scan goes to a human instead.
            </>
          )}
        </div>
      )}
    </Sheet>
  );
}
