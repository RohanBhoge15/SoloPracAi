"""Document templates: prescription, certificate, invoice, weekly report.

Each returns an HTML string; `app.services.pdf` renders it. Keeping them
as templates rather than a canvas-drawing API means a designer can change
the output without touching Python.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional

from app.documents.render import (
    doctor_block,
    fmt_date,
    patient_block,
    rupees,
    shell,
    signature_block,
    watermark,
)

# ── prescription ───────────────────────────────────────────────────


def prescription(
    *,
    doctor: dict,
    patient: dict,
    items: List[Dict[str, Any]],
    notes: str = "",
    follow_up_days: Optional[int] = None,
    reference: str = "",
    issued_at: Optional[datetime] = None,
    diagnosis: str = "",
) -> str:
    """A medication order.

    Notes and follow-up sit inside the document because an instruction
    the doctor only says aloud is an instruction the patient loses.
    """
    if not items:
        raise ValueError("a prescription needs at least one item")

    rows = []
    for index, item in enumerate(items, start=1):
        drug = escape(str(item.get("drug") or "—"))
        # Duration is deliberately NOT joined into `detail`. It gets its own
        # column below, and including it here printed "3 days" twice in the
        # same row - once inside the run-on strength string and once in the
        # Duration cell. A patient reading a prescription should meet each
        # fact exactly once.
        bits = []
        for key in ("dose", "frequency", "route"):
            value = item.get(key)
            if value:
                bits.append(escape(str(value)))
        detail = " · ".join(bits)
        instruction = item.get("instruction")
        if instruction:
            detail += f'<br><em>{escape(str(instruction))}</em>'

        rows.append(
            "<tr>"
            f'<td class="num" style="width:6%">{index}</td>'
            f'<td style="width:34%"><div class="drug-name">{drug}</div>'
            f'<div class="drug-detail">{detail}</div></td>'
            f'<td style="width:44%">{escape(str(item.get("timing", "") or ""))}</td>'
            f'<td style="width:16%">{escape(str(item.get("duration", "") or ""))}</td>'
            "</tr>"
        )

    # The diagnosis is the single most important line on a prescription:
    # it is what tells the patient what the drugs are for. It was accepted
    # by the API and then silently dropped on the way to the page, so a
    # patient got a list of drugs with no indication attached.
    diagnosis_block = ""
    if diagnosis:
        diagnosis_block = (
            '<div class="section"><div class="section-title">Diagnosis</div>'
            f'<div class="diagnosis">{escape(str(diagnosis))}</div></div>'
        )

    notes_block = ""
    if notes:
        notes_block = (
            '<div class="section"><div class="section-title">Notes to patient</div>'
            f'<div class="notes">{escape(notes)}</div></div>'
        )

    follow = ""
    if follow_up_days:
        when = fmt_date(
            datetime.fromtimestamp(
                (issued_at or datetime.now()).timestamp() + follow_up_days * 86400
            )
        )
        follow = (
            '<div class="section"><div class="callout">'
            f"<strong>Follow up:</strong> review on {escape(when)} "
            f"({follow_up_days} days from today). "
            "Bring this prescription and any reports collected since."
            "</div></div>"
        )

    body = f"""
    {diagnosis_block}
    <div class="section">
      <table>
        <thead>
          <tr>
            <th class="num">#</th>
            <th>Medication</th>
            <th>How to take it</th>
            <th>Duration</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    {notes_block}
    {follow}
    <div class="section">
      <div class="callout">
        <strong>General advice.</strong> Take medication only as prescribed.
        Do not self-medicate or take over-the-counter drugs without advice.
        Contact the clinic if you develop fever, rash, or unusual symptoms.
      </div>
    </div>
    """
    return shell(
        kind_title="Prescription",
        reference=reference,
        doctor=doctor,
        patient=patient,
        watermark_text=doctor.get("account", {}).get("display_name", "SoloPrac")
        or "SoloPrac",
        body=body,
        issued_at=issued_at,
        provenance=(
            "This prescription is a rendering of an immutable version of the "
            "clinical record. If a later correction is issued, this document "
            "is marked superseded rather than deleted."
        ),
    )


# ── medical certificate ────────────────────────────────────────────


def certificate(
    *,
    doctor: dict,
    patient: dict,
    certificate_type: str,
    body_text: str,
    valid_from: Optional[datetime] = None,
    valid_until: Optional[datetime] = None,
    reference: str = "",
    issued_at: Optional[datetime] = None,
) -> str:
    """Fitness, sickness, or medical certificate.

    Certificate text is passed through verbatim rather than generated:
    a statutory certificate is the clinician's own words, and an LLM has
    no business rewriting it.
    """
    validity = ""
    if valid_from or valid_until:
        validity = (
            '<div class="section"><div class="callout">'
            f"<strong>Valid:</strong> {fmt_date(valid_from)}"
            + (f" &ndash; {fmt_date(valid_until)}" if valid_until else "")
            + "</div></div>"
        )

    body = f"""
    <div class="section">
      <div class="notes">{escape(body_text)}</div>
    </div>
    {validity}
    <div class="section">
      <div class="callout">
        <strong>Declaration.</strong> I have examined the above-named patient
        and certify the statement above to the best of my clinical judgement.
      </div>
    </div>
    """
    return shell(
        kind_title=certificate_type or "Medical Certificate",
        reference=reference,
        doctor=doctor,
        patient=patient,
        watermark_text=(doctor.get("account", {}).get("display_name", "SoloPrac") or "SoloPrac"),
        body=body,
        issued_at=issued_at,
        provenance=(
            f"Issued {fmt_date(issued_at or datetime.now())}. "
            "This certificate is digitally issued and watermarked with the "
            "issuing clinician's name."
        ),
    )


# ── invoice ────────────────────────────────────────────────────────


def invoice(
    *,
    doctor: dict,
    patient: dict,
    lines: List[Dict[str, Any]],
    discount: int = 0,
    tax_percent: float = 0.0,
    paid: bool = False,
    reference: str = "",
    issued_at: Optional[datetime] = None,
) -> str:
    """Consultation bill.

    Arithmetic is done here in integers, not left to the template, so a
    rounding error cannot silently change what a patient is charged.
    """
    if not lines:
        raise ValueError("an invoice needs at least one line")

    subtotal = 0
    rows = []
    for line in lines:
        qty = max(1, int(line.get("quantity", 1)))
        price = max(0, int(line.get("unit_price", 0)))
        amount = qty * price
        subtotal += amount
        rows.append(
            "<tr>"
            f'<td>{escape(str(line.get("description", "")))}</td>'
            f'<td class="num">{qty}</td>'
            f'<td class="num">{rupees(price)}</td>'
            f'<td class="num">{rupees(amount)}</td>'
            "</tr>"
        )

    discount = max(0, min(discount, subtotal))
    taxable = subtotal - discount
    tax = int(round(taxable * tax_percent / 100.0))
    total = taxable + tax

    totals = f"""
      <tr><td>Subtotal</td><td class="num"></td><td class="num"></td>
          <td class="num">{rupees(subtotal)}</td></tr>
    """
    if discount:
        totals += (
            '<tr><td>Discount</td><td class="num"></td><td class="num"></td>'
            f'<td class="num">-{rupees(discount)}</td></tr>'
        )
    if tax:
        totals += (
            f'<tr><td>Tax ({tax_percent:g}%)</td><td class="num"></td>'
            f'<td class="num"></td><td class="num">{rupees(tax)}</td></tr>'
        )
    totals += (
        '<tr class="grand"><td>Total</td><td class="num"></td><td class="num"></td>'
        f'<td class="num">{rupees(total)}</td></tr>'
    )

    status = (
        '<div class="callout"><strong>Paid in full.</strong></div>'
        if paid
        else '<div class="callout">Please settle at the clinic counter.</div>'
    )

    body = f"""
    <div class="section">
      <table>
        <thead>
          <tr><th>Description</th><th class="num">Qty</th>
              <th class="num">Rate</th><th class="num">Amount</th></tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <table class="totals">{totals}</table>
    </div>
    <div class="section">{status}</div>
    """
    return shell(
        kind_title="Invoice",
        reference=reference,
        doctor=doctor,
        patient=patient,
        watermark_text=(doctor.get("account", {}).get("display_name", "SoloPrac") or "SoloPrac"),
        body=body,
        issued_at=issued_at,
        provenance=f"Issued {fmt_date(issued_at or datetime.now())}.",
    )


# ── weekly report ──────────────────────────────────────────────────


def weekly_report(
    *,
    doctor: dict,
    patient: dict,
    summary: str,
    findings: List[Dict[str, Any]],
    medications: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
    retrieved_context: List[str],
    period_from: Optional[datetime] = None,
    period_to: Optional[datetime] = None,
    reference: str = "",
    issued_at: Optional[datetime] = None,
) -> str:
    """A clinician-facing summary of one week.

    The corrections section is the point of the whole system: a week is
    reported with what changed in it, not just what was measured.
    """
    period = ""
    if period_from or period_to:
        period = (
            f"{fmt_date(period_from)} &ndash; {fmt_date(period_to)}"
            if period_from and period_to
            else fmt_date(period_from or period_to)
        )

    narrative = ""
    if summary:
        narrative = (
            '<div class="section">'
            '<div class="section-title">Clinical summary</div>'
            f'<div class="notes">{escape(summary)}</div></div>'
        )

    def strip(rows: List[Dict[str, Any]], key: str, label: str) -> str:
        if not rows:
            return ""
        cells = []
        for row in rows:
            cls = row.get("_trend", "")
            trend = f' <span class="{cls}">{row.get("_arrow", "")}</span>' if cls else ""
            cells.append(
                f'<div class="strip-item">'
                f'<div class="strip-label">{escape(str(row.get(key, label)))}</div>'
                f'<div class="strip-value">{escape(str(row.get("value", "-")))}'
                f"{trend}</div>"
                f'<div class="strip-date">{escape(str(row.get("date", "")))}</div>'
                "</div>"
            )
        return (
            f'<div class="section"><div class="section-title">{escape(label)}</div>'
            f'<div class="strip">{"".join(cells)}</div></div>'
        )

    meds = ""
    if medications:
        rows = []
        for med in medications:
            status = str(med.get("status", "active")).lower()
            name = escape(str(med.get("drug", "")))
            if status == "stopped":
                name = f'<s>{name}</s> <em>(stopped)</em>'
            detail = " · ".join(
                escape(str(med[k]))
                for k in ("dose", "frequency")
                if med.get(k)
            )
            rows.append(
                "<tr>"
                f'<td><div class="drug-name">{name}</div>'
                f'<div class="drug-detail">{detail}</div></td>'
                f'<td>{escape(str(med.get("since", "")))}</td>'
                "</tr>"
            )
        meds = (
            '<div class="section"><div class="section-title">Medication</div>'
            '<table><thead><tr><th>Drug</th><th>Since</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )

    corr = ""
    if corrections:
        rows = []
        for c in corrections:
            rows.append(
                "<tr>"
                f'<td>{escape(str(c.get("test", c.get("drug", "Record"))))}</td>'
                f'<td><s>{escape(str(c.get("old", "-")))}</s></td>'
                f'<td><strong>{escape(str(c.get("new", "-")))}</strong></td>'
                f'<td>{escape(str(c.get("date", "")))}</td>'
                "</tr>"
            )
        corr = (
            '<div class="section">'
            '<div class="section-title">Corrections this week</div>'
            '<table><thead><tr><th>Record</th><th>Was</th><th>Now</th>'
            f'<th>Date</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            "</div>"
        )

    sources = ""
    if retrieved_context:
        chips = "".join(
            f'<code>{escape(str(s))}</code> ' for s in retrieved_context
        )
        sources = (
            '<div class="section">'
            '<div class="section-title">Records consulted</div>'
            f'<div class="provenance" style="border:0;margin:0;padding:0">{chips}</div>'
            "</div>"
        )

    body = f"""
    {f'<div class="callout"><strong>Period:</strong> {period}</div><br>' if period else ''}
    {narrative}
    {strip(findings, "test", "Investigation")}
    {meds}
    {corr}
    {sources}
    """
    return shell(
        kind_title="Weekly Clinical Report",
        reference=reference,
        doctor=doctor,
        patient=patient,
        watermark_text=(doctor.get("account", {}).get("display_name", "SoloPrac") or "SoloPrac"),
        body=body,
        issued_at=issued_at,
        provenance=(
            f"Generated {fmt_date(issued_at or datetime.now())} from the "
            "patient's versioned record. Superseded values are shown struck "
            "through and are never presented as current."
        ),
    )
