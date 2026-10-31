/**
 * Issue a consultation invoice.
 *
 * Money is handled as integer paise throughout, because floating point
 * arithmetic on currency is a bug generator: 0.1 + 0.2 is famously not
 * 0.3, and a bill that is a paisa out is a bill a patient disputes. The
 * amount typed here is rupees, converted once on the way out.
 */

import { useEffect, useRef, useState } from "react";
import { api, type Patient } from "@/lib/api";
import { Sheet } from "@/components/Authoring";
import { IssuedSuccess, useIssuedDocument } from "@/components/IssuedDoc";

interface Line {
  id: number;
  description: string;
  quantity: number;
  unitPrice: number;
}

let nextId = 1;
const blankLine = (): Line => ({
  id: nextId++,
  description: "",
  quantity: 1,
  unitPrice: 0,
});

export default function InvoiceSheet({
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
  const [lines, setLines] = useState<Line[]>([blankLine()]);
  const [discount, setDiscount] = useState("0");
  const [tax, setTax] = useState("0");
  const [paid, setPaid] = useState(false);
  const [send, setSend] = useState(true);
  const alive = useRef(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const { busy, error, reference, pdfUrl, issue } = useIssuedDocument("Invoice");

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

  const set = (id: number, patch: Partial<Line>) =>
    setLines((ls) => ls.map((l) => (l.id === id ? { ...l, ...patch } : l)));

  const num = (v: string) => {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0;
  };

  const usable = lines.filter((l) => l.description.trim());
  const subtotal = usable.reduce(
    (sum, l) => sum + num(String(l.quantity)) * num(String(l.unitPrice)),
    0,
  );
  const afterDiscount = Math.max(0, subtotal - num(discount));
  const taxAmount = Math.round((afterDiscount * (Number(tax) || 0)) / 100);
  const total = afterDiscount + taxAmount;
  const rupees = (paise: number) => (paise / 100).toFixed(2);

  async function submit() {
    if (!patientId || !usable.length) return;
    await issue(
      () =>
        api.createInvoice({
          patient_id: patientId,
          lines: usable.map((l) => ({
            description: l.description.trim(),
            quantity: num(String(l.quantity)),
            unit_price: num(String(l.unitPrice)),
          })),
          discount: num(discount),
          tax_percent: Number(tax) || 0,
          paid,
          send_to_patient: send,
        }),
      onSent,
    );
  }

  if (reference) {
    return (
      <Sheet title="Invoice" onClose={onClose}>
        <IssuedSuccess
          label="Invoice created"
          reference={reference}
          pdfUrl={pdfUrl}
          onDone={onClose}
        />
      </Sheet>
    );
  }

  return (
    <Sheet
      title="Invoice"
      subtitle="Consultation bill, rendered as a watermarked PDF."
      onClose={onClose}
      footer={
        <>
          <span className="modal-reason" role="status">
            {!busy && !patientId
              ? "Choose a patient first"
              : !usable.length
                ? "Add at least one line with a description"
                : ""}
          </span>
          <button
            className="btn btn-primary btn-sm"
            onClick={submit}
            disabled={busy || !patientId || !usable.length}
          >
            {busy ? <span className="spinner" /> : "Create invoice"}
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
        <label className="field-label">Line items</label>
        <div className="invoice-table">
          <div className="invoice-row invoice-row--head">
            <span>Description</span>
            <span>Qty</span>
            <span>Rate</span>
          </div>
          {lines.map((l) => (
            <div className="invoice-row" key={l.id}>
              <input
                className="input"
                value={l.description}
                onChange={(e) => set(l.id, { description: e.target.value })}
                placeholder="Consultation"
              />
              <input
                className="input"
                type="number"
                min={1}
                value={l.quantity}
                onChange={(e) => set(l.id, { quantity: num(e.target.value) })}
              />
              <input
                className="input"
                type="number"
                min={0}
                value={l.unitPrice}
                onChange={(e) => set(l.id, { unitPrice: num(e.target.value) })}
                placeholder="500"
              />
            </div>
          ))}
        </div>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          style={{ marginTop: 8 }}
          onClick={() => setLines((ls) => [...ls, blankLine()])}
        >
          + Add line
        </button>
        <span className="field-hint">Amounts in rupees.</span>
      </div>

      <div className="field-row">
        <label className="field">
          <span>Discount (₹)</span>
          <input
            className="input"
            type="number"
            min={0}
            value={discount}
            onChange={(e) => setDiscount(e.target.value)}
          />
        </label>
        <label className="field">
          <span>Tax (%)</span>
          <input
            className="input"
            type="number"
            min={0}
            step="0.1"
            value={tax}
            onChange={(e) => setTax(e.target.value)}
          />
        </label>
      </div>

      <div className="invoice-total">
        <div>
          <span>Subtotal</span>
          <b>₹{rupees(subtotal)}</b>
        </div>
        {num(discount) > 0 && (
          <div>
            <span>Discount</span>
            <b>− ₹{rupees(num(discount))}</b>
          </div>
        )}
        {taxAmount > 0 && (
          <div>
            <span>Tax</span>
            <b>₹{rupees(taxAmount)}</b>
          </div>
        )}
        <div className="invoice-total-grand">
          <span>Total</span>
          <b>₹{rupees(total)}</b>
        </div>
      </div>

      <label className="field checkbox-field">
        <input
          type="checkbox"
          checked={paid}
          onChange={(e) => setPaid(e.target.checked)}
        />
        <span>Paid in full</span>
      </label>

      <label className="field checkbox-field">
        <input
          type="checkbox"
          checked={send}
          onChange={(e) => setSend(e.target.checked)}
        />
        <span>Send to the patient's inbox</span>
      </label>
    </Sheet>
  );
}
