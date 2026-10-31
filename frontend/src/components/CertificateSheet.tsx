/**
 * Issue a medical certificate: fitness, sickness, or a fitness-to-work
 * note.
 *
 * The body text is the doctor's own words. This sheet deliberately does
 * not generate it - a certificate is a legal attestation, and a language
 * model inventing one would be indefensible in a way that a clinical
 * summary is not. The backend endpoint, the template and the PDF renderer
 * all already existed; only this entry point was missing.
 */

import { useEffect, useRef, useState } from "react";
import { api, type Patient } from "@/lib/api";
import { Sheet } from "@/components/Authoring";
import { IssuedSuccess, useIssuedDocument } from "@/components/IssuedDoc";

/** The three kinds a doctor actually issues. */
const TYPES = [
  { value: "Medical Certificate", hint: "A general fitness note" },
  { value: "Sickness Certificate", hint: "Unfit for work, with dates" },
  { value: "Fitness Certificate", hint: "Fit to resume duty" },
  { value: "Medical Leave Certificate", hint: "Leave for treatment" },
];

export default function CertificateSheet({
  onClose,
  onSent,
  initialPatientId,
}: {
  onClose: () => void;
  onSent: (reference: string) => void;
  initialPatientId?: string;
}) {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, setPatientId] = useState(initialPatientId ?? "");
  const [kind, setKind] = useState(TYPES[0].value);
  const [body, setBody] = useState("");
  const [from, setFrom] = useState("");
  const [until, setUntil] = useState("");
  const [send, setSend] = useState(true);
  const alive = useRef(true);
  // Kept separate from the issue flow's error: failing to load the patient
  // list is a different problem from failing to write a certificate, and
  // they should not overwrite each other's message.
  const [loadError, setLoadError] = useState<string | null>(null);
  const { busy, error, reference, pdfUrl, issue } =
    useIssuedDocument("Certificate");

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    let live = true;
    api
      .listPatients()
      .then((list) => {
        if (!live) return;
        setPatients(list);
        if (initialPatientId) setPatientId(initialPatientId);
        else if (list.length === 1) setPatientId(list[0].id);
      })
      .catch(() => {
        if (live) setLoadError("Could not load patients.");
      });
    return () => {
      live = false;
    };
  }, [initialPatientId]);

  // A date input gives "yyyy-mm-dd"; the endpoint wants a full instant.
  const asInstant = (d: string) => (d ? new Date(`${d}T00:00:00`).toISOString() : null);

  async function submit() {
    if (!patientId || !body.trim()) return;
    await issue(
      () =>
        api.createCertificate({
          patient_id: patientId,
          certificate_type: kind,
          body_text: body.trim(),
          valid_from: asInstant(from),
          valid_until: asInstant(until),
          send_to_patient: send,
        }),
      // Desktop: the PDF already opened in its own tab, so close up.
      onSent,
    );
  }

  const active = TYPES.find((t) => t.value === kind);

  // Issued, but no tab could be opened (phone). The form is gone and the
  // only things left are the file and a way out.
  if (reference) {
    return (
      <Sheet title="Medical certificate" onClose={onClose}>
        <IssuedSuccess
          label="Certificate issued"
          reference={reference}
          pdfUrl={pdfUrl}
          onDone={onClose}
        />
      </Sheet>
    );
  }

  return (
    <Sheet
      title="Medical certificate"
      subtitle="Rendered as a watermarked PDF and sent to the patient's inbox."
      onClose={onClose}
      footer={
        <>
          <span className="modal-reason" role="status">
            {!busy && !patientId
              ? "Choose a patient first"
              : !body.trim()
                ? "Write what the certificate says"
                : ""}
          </span>
          <button
            className="btn btn-primary btn-sm"
            onClick={submit}
            disabled={busy || !patientId || !body.trim()}
          >
            {busy ? <span className="spinner" /> : "Issue certificate"}
          </button>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            Cancel
          </button>
        </>
      }
    >
      {(error ?? loadError) && (
        <div className="callout callout--warn">{error ?? loadError}</div>
      )}

      <div className="field">
        <label className="field-label">Patient</label>
        <select
          className="input"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
        >
          <option value="">Choose a patient…</option>
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              {p.display_name}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label className="field-label">Type</label>
        <div className="window-row">
          {TYPES.map((t) => (
            <button
              key={t.value}
              type="button"
              className={`chip ${kind === t.value ? "chip--on" : ""}`}
              onClick={() => setKind(t.value)}
              title={t.hint}
            >
              {t.value}
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <label className="field-label">What it says *</label>
        <textarea
          className="textarea"
          value={body}
          onChange={(e) => setBody(e.target.value)}
          placeholder="Patient was examined today and is unfit for work for three days."
          rows={4}
        />
        <span className="field-hint">
          Written by you, not generated. A certificate is an attestation,
          so the words on it have to be yours.
        </span>
      </div>

      <div className="field-row">
        <label className="field">
          <span>Valid from</span>
          <input
            className="input"
            type="date"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
          />
        </label>
        <label className="field">
          <span>Valid until</span>
          <input
            className="input"
            type="date"
            value={until}
            onChange={(e) => setUntil(e.target.value)}
          />
        </label>
      </div>

      <label className="field checkbox-field">
        <input
          type="checkbox"
          checked={send}
          onChange={(e) => setSend(e.target.checked)}
        />
        <span>Send to the patient's inbox</span>
      </label>

      {active && (
        <div className="callout callout--info">
          <b>{active.value}</b>
          {active.hint}. The doctor's name, specialty and registration
          number are printed in the signature block, so the patient can
          verify who issued it.
        </div>
      )}
    </Sheet>
  );
}
