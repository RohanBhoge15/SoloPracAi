import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { api, ApiError, type Patient } from "@/lib/api";
import { openPdfInNewTab } from "@/lib/pdf";
import DrugInput from "@/components/DrugInput";
import "@/components/composer.css";

/** One row of the medication table. `id` is local only - the API does not
 *  want it, and keying rows by index makes React reuse the wrong input. */
interface Row {
  id: number;
  drug: string;
  dose: string;
  frequency: string;
  duration: string;
  route: string;
  timing: string;
  instruction: string;
}

const BLANK: Omit<Row, "id"> = {
  drug: "",
  dose: "",
  frequency: "",
  duration: "",
  route: "",
  timing: "",
  instruction: "",
};

let nextId = 1;
const blankRow = (): Row => ({ id: nextId++, ...BLANK });

/* Common frequencies, offered as buttons because typing "twice daily" 30
   times a day is the fastest way to make a doctor stop using the tool.
   The field stays editable for anything not on this list. */
const FREQUENCIES = ["Once daily", "Twice daily", "Three times daily", "SOS"];

export default function PrescriptionComposer({
  onClose,
  onSent,
  initialPatientId,
}: {
  onClose: () => void;
  onSent: () => void;
  /** Pre-selects a patient. Passed when the composer is opened from a
   *  patient's own record, so the doctor does not re-pick the patient
   *  they are already looking at. */
  initialPatientId?: string;
}) {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, setPatientId] = useState(initialPatientId ?? "");
  const [rows, setRows] = useState<Row[]>([blankRow()]);
  const [notes, setNotes] = useState("");
  // What the drugs are for. Without it the patient receives a list of
  // medicines with no indication, and has to work out from the drug names
  // alone what they are being taken for.
  const [diagnosis, setDiagnosis] = useState("");
  const [followUp, setFollowUp] = useState("");
  const [send, setSend] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState<string | null>(null);
  // The PDF's presigned URL, kept so the success screen can offer it.
  // Null on desktop, where the tab already opened itself.
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  const firstField = useRef<HTMLInputElement>(null);
  const drugField = useRef<HTMLInputElement>(null);

  // The drug name is the only field that unlocks the submit button, so
  // the cursor goes there.
  //
  // `focus()` alone was not enough: it moves the caret into the field but
  // does not necessarily scroll its scroll container, so on a short
  // window the field stayed above the fold and invisible while the reason
  // was printed at the bottom. `preventScroll` is deliberately not used
  // here - scrolling the field into view is exactly what is wanted.
  useEffect(() => {
    // Twice on purpose. The sheet animates open, so an early scroll lands
    // on a layout that is about to change; the second call runs after the
    // animation has settled and puts it right.
    const reveal = () => {
      const el = drugField.current;
      if (!el) return;
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      el.focus({ preventScroll: true });
    };
    const first = window.setTimeout(reveal, 140);
    const second = window.setTimeout(reveal, 420);
    return () => {
      window.clearTimeout(first);
      window.clearTimeout(second);
    };
  }, []);

  useEffect(() => {
    let live = true;
    api
      .listPatients()
      .then((list) => {
        if (!live) return;
        setPatients(list);
        // An explicit patient from the caller always wins. The doctor
        // opened this from a specific record, so re-picking that patient
        // in a dropdown is busywork and a chance to pick the wrong one.
        if (initialPatientId) {
          setPatientId(initialPatientId);
        } else if (list.length === 1) {
          // Otherwise preselect the only patient, which is the common case
          // for a single-patient practice.
          setPatientId(list[0].id);
        }
      })
      .catch(() => {
        if (live) setError("Could not load patients.");
      });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    firstField.current?.focus();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    // The page behind must not scroll while a dialog is open.
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [onClose]);

  function update(id: number, patch: Partial<Row>) {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }

  const filled = useMemo(
    () => rows.filter((r) => r.drug.trim().length > 0),
    [rows],
  );

  const canSend = patientId && filled.length > 0 && !busy;

  // Why the button is locked, in the order the doctor would fix it.
  const blockedReason = !patientId
    ? "Choose a patient first"
    : filled.length === 0
      ? "Add at least one medicine to continue"
      : "";

  async function submit() {
    setError(null);
    setBusy(true);
    try {
      const result = await api.createPrescription({
        patient_id: patientId,
        items: filled.map((r) => ({
          drug: r.drug.trim(),
          dose: r.dose.trim(),
          frequency: r.frequency.trim(),
          duration: r.duration.trim(),
          route: r.route.trim(),
          timing: r.timing.trim(),
          instruction: r.instruction.trim(),
        })),
        notes: notes.trim(),
        diagnosis: diagnosis.trim(),
        follow_up_days: followUp ? Number(followUp) : null,
        send_to_patient: send,
      });
      const reference =
        (result.content as Record<string, unknown>)?.reference ??
        result.id.slice(0, 8);
      setSent(String(reference));

      // Hand the doctor the PDF. Desktop gets a new tab; on a phone the
      // tab cannot be opened from here (see lib/pdf) so the success screen
      // shows a button instead. Fetched after issuing rather than before,
      // because the URL is a short-lived presigned one.
      void api
        .downloadDocument(result.id)
        .then(({ url }) => {
          if (openPdfInNewTab(url)) {
            // The tab is open, so the doctor already has the PDF. Close
            // up after a beat to show the reference, the way it always did.
            window.setTimeout(onSent, 1400);
          } else {
            // No tab. Keep the sheet on the success screen so the button
            // is actually reachable. Closing here would be what made this
            // path look broken on a phone: the reference flashed past and
            // the "Issued" screen with the View PDF button was gone before
            // it could be tapped.
            setPdfUrl(url);
          }
        })
        .catch(() => {
          // The prescription is already issued. Failing to open the
          // preview must not turn that into an error state - but do not
          // leave the doctor stranded either, so close as before.
          window.setTimeout(onSent, 1400);
        });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not issue it.");
      setBusy(false);
    }
  }

  // Portaled to <body> so the dialog escapes the page's stacking context.
  // Rendered inline, the sticky topbar (z-index 50) sat on top of the
  // backdrop (z-index 100) because the page created a stacking context
  // underneath it — the dialog opened *behind* the navbar.
  return createPortal(
    <AnimatePresence>
      <motion.div
        className="modal-backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className="modal card"
          initial={{ opacity: 0, y: 24, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 16, scale: 0.98 }}
          transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
          onClick={(e) => e.stopPropagation()}
          role="dialog"
          aria-modal="true"
          aria-label="New prescription"
        >
          <header className="modal-head">
            <div>
              <h2 className="modal-title">New prescription</h2>
              <p className="modal-sub">
                Rendered as a watermarked PDF and sent to the patient's inbox.
              </p>
            </div>
            <button
              type="button"
              className="modal-x"
              onClick={onClose}
              aria-label="Close"
            >
              x
            </button>
          </header>

          {sent ? (
            <div className="modal-done">
              <div className="done-mark" aria-hidden="true">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m5 13 4 4L19 7" />
                </svg>
              </div>
              <h3>Issued</h3>
              <p className="mono">{sent}</p>
              {/* The actions are rendered whenever the sheet is on the
                  success screen, not only when there is a PDF to show.
                  On desktop the PDF is already open in a new tab, so
                  `pdfUrl` stays null - and that used to leave this screen
                  with no button on it at all. Escape and a backdrop click
                  still worked, so it was never a lockout, but a dialog the
                  doctor cannot see a way out of is a bad state to leave
                  someone in. "Done" is always here; "View PDF" appears
                  only when there is actually something to view. */}
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
                <button
                  type="button"
                  className={pdfUrl ? "btn btn-ghost" : "btn btn-primary"}
                  onClick={onClose}
                >
                  Done
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="modal-body">
                <label className="field">
                  <span>Patient</span>
                  <select
                    value={patientId}
                    onChange={(e) => setPatientId(e.target.value)}
                  >
                    <option value="">Choose a patient</option>
                    {patients.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.display_name}
                        {p.age != null ? ` - ${p.age}y` : ""}
                      </option>
                    ))}
                  </select>
                  {patients.length === 0 && (
                    <small>No patients yet. Add one from the Records tab first.</small>
                  )}
                </label>

                <div className="rx-table">
                  <div className="rx-head">
                    <span>Drug</span>
                    <span>Dose</span>
                    <span>Frequency</span>
                    <span>Duration</span>
                    <span aria-hidden="true" />
                  </div>

                  {rows.map((row, index) => (
                    <div className="rx-row" key={row.id}>
                      <DrugInput
                        inputRef={index === 0 ? drugField : firstField}
                        value={row.drug}
                        onChange={(next) => update(row.id, { drug: next })}
                        placeholder="Metformin"
                        ariaLabel={`Drug ${index + 1}`}
                      />
                      <input
                        value={row.dose}
                        onChange={(e) => update(row.id, { dose: e.target.value })}
                        placeholder="500 mg"
                        aria-label={`Dose ${index + 1}`}
                      />
                      <input
                        value={row.frequency}
                        onChange={(e) =>
                          update(row.id, { frequency: e.target.value })
                        }
                        placeholder="Twice daily"
                        list="rx-frequencies"
                        aria-label={`Frequency ${index + 1}`}
                      />
                      <input
                        value={row.duration}
                        onChange={(e) =>
                          update(row.id, { duration: e.target.value })
                        }
                        placeholder="30 days"
                        aria-label={`Duration ${index + 1}`}
                      />
                      <button
                        type="button"
                        className="rx-del"
                        onClick={() =>
                          setRows((prev) =>
                            prev.length === 1
                              ? [blankRow()]
                              : prev.filter((r) => r.id !== row.id),
                          )
                        }
                        aria-label={`Remove drug ${index + 1}`}
                        disabled={rows.length === 1}
                      >
                        x
                      </button>
                    </div>
                  ))}
                  <datalist id="rx-frequencies">
                    {FREQUENCIES.map((f) => (
                      <option key={f} value={f} />
                    ))}
                  </datalist>

                  <div className="rx-foot">
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => setRows((prev) => [...prev, blankRow()])}
                    >
                      + Add another drug
                    </button>
                    {rows.length > 1 && (
                      <span className="rx-hint">
                        {rows.length - filled.length} empty{" "}
                        {rows.length - filled.length === 1 ? "row" : "rows"} will
                        be ignored
                      </span>
                    )}
                  </div>
                </div>

                <label className="field">
                  <span>Diagnosis</span>
                  <input
                    value={diagnosis}
                    onChange={(e) => setDiagnosis(e.target.value)}
                    placeholder="Type 2 Diabetes Mellitus"
                  />
                  <span className="field-hint">
                    Printed at the top of the prescription. This is what tells
                    the patient what the medicines are for.
                  </span>
                </label>

                <label className="field">
                  <span>Notes for the patient</span>
                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Take after food. Report any unusual bruising."
                  />
                </label>

                <div className="field-row">
                  <label className="field">
                    <span>Follow-up in days</span>
                    <input
                      type="number"
                      min={1}
                      max={365}
                      value={followUp}
                      onChange={(e) => setFollowUp(e.target.value)}
                      placeholder="14"
                    />
                  </label>
                  <label className="field checkbox-field">
                    <input
                      type="checkbox"
                      checked={send}
                      onChange={(e) => setSend(e.target.checked)}
                    />
                    <span>Send to the patient's inbox now</span>
                  </label>
                </div>

                {error && (
                  <p className="auth-error" role="alert">
                    {error}
                  </p>
                )}
              </div>

              <footer className="modal-foot">
                {/* A disabled button with no stated reason is a guess. The
                    reason lives next to the button, not in a field the
                    doctor has to find. */}
                <span className="modal-reason" role="status">
                  {!canSend && !busy ? blockedReason : ""}
                </span>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={onClose}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!canSend}
                  onClick={submit}
                >
                  {busy ? <span className="spinner" /> : "Issue prescription"}
                </button>
              </footer>
            </>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );
}
