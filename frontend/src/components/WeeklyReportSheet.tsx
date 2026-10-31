/**
 * Generate a weekly clinical report, grounded in the patient's own record.
 *
 * The report is not a template with the numbers dropped in. The backend
 * loads the patient's events, builds the supersession graph, and renders
 * the period - current values and the corrections that superseded them -
 * into the LLM prompt. A potassium amended from 2.1 to 4.8 is written as
 * 4.8, with 2.1 retained in the corrections table.
 *
 * This sheet only chooses the window and whether the patient receives it.
 * Everything about the content is decided server-side, from the record.
 */

import { useEffect, useRef, useState } from "react";
import { api, type Patient } from "@/lib/api";
import { Sheet } from "@/components/Authoring";
import { IssuedSuccess, useIssuedDocument } from "@/components/IssuedDoc";

/** The windows a doctor actually asks for. "Whole record" is included
 *  because for a long-standing patient it is a legitimate report, and the
 *  endpoint allows it. */
const WINDOWS = [
  { days: 7, label: "Last 7 days" },
  { days: 14, label: "Last 14 days" },
  { days: 30, label: "Last 30 days" },
  { days: 90, label: "Last 3 months" },
  { days: 3650, label: "Whole record" },
];

export default function WeeklyReportSheet({
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
  const [days, setDays] = useState(7);
  const [send, setSend] = useState(true);
  const mounted = useRef(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const { busy, error, reference, pdfUrl, issue } =
    useIssuedDocument("Report");

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
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

  const patient = patients.find((p) => p.id === patientId) ?? null;
  void patient;

  async function submit() {
    if (!patientId) return;
    await issue(
      () =>
        api.createWeeklyReport({
          patient_id: patientId,
          days,
          send_to_patient: send,
        }),
      onSent,
    );
  }

  if (reference) {
    return (
      <Sheet title="Weekly report" onClose={onClose}>
        <IssuedSuccess
          label="Report ready"
          reference={reference}
          pdfUrl={pdfUrl}
          onDone={onClose}
        />
      </Sheet>
    );
  }

  return (
    <Sheet
      title="Weekly report"
      subtitle="Written from this patient's record, with corrections applied."
      onClose={onClose}
      footer={
        <>
          <button
            className="btn btn-primary btn-sm"
            onClick={submit}
            disabled={busy || !patientId}
          >
            {busy ? <span className="spinner" /> : "Generate report"}
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
        <label className="field-label">Period</label>
        <div className="window-row">
          {WINDOWS.map((w) => (
            <button
              key={w.days}
              type="button"
              className={`chip ${days === w.days ? "chip--on" : ""}`}
              onClick={() => setDays(w.days)}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>

      <label className="field checkbox-field">
        <input
          type="checkbox"
          checked={send}
          onChange={(e) => setSend(e.target.checked)}
        />
        <span>Send to the patient's inbox</span>
      </label>

      <div className="callout callout--info">
        <b>What the report will say</b>
        The current value of every test in the period, plus any value that
        was corrected. A corrected value is written as the corrected
        figure; the original is kept in the corrections table, struck
        through, and never presented as current.
      </div>
    </Sheet>
  );
}
