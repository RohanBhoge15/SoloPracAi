"""Render every document type to PDF and prove the bytes are real.

Runs against the API container, because WeasyPrint needs Pango and the
host venv does not have it. If this passes, the document pipeline works
in the same environment a deployment would use.

The script registers a real doctor, because every document route requires
a signed-in clinician - a test that skipped auth would prove nothing.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

API = "http://localhost:8000"

#: SeaweedFS on the compose network, for when this runs inside the container.
INTERNAL_S3_HOST = "seaweedfs"


def req(method, path, payload=None, timeout=120, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    r = urllib.request.Request(API + path, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def _reachable(url: str) -> str:
    """Rewrite a presigned URL to an address this process can reach.

    The API signs URLs against the browser-facing host, which is correct -
    a physician's browser is not on the compose network. But this script
    runs *inside* the container, where `localhost:8333` is the container
    itself and nothing is listening. Without this the download step fails
    for every document and looks like a product bug.

    Only the host is swapped. The path, query and signature are left
    exactly as issued, so this still exercises the real presigned URL and
    still fails if the signature is wrong.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        return urllib.parse.urlunparse(
            parsed._replace(netloc=f"{INTERNAL_S3_HOST}:{parsed.port or 80}")
        )
    return url


def main() -> int:
    failures = 0
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_pdf_out")

    print("=" * 66)
    print("SOLUM - document pipeline verification")
    print("=" * 66)

    # ── capabilities ────────────────────────────────────────────
    status, caps = req("GET", "/api/v1/documents/capabilities")
    if status != 200:
        print(f"  API unreachable: {caps}")
        return 1
    print(f"\n[1] capabilities")
    print(f"    pdf engine : {caps['pdf']}")
    print(f"    storage    : {caps['storage']}")
    print(f"    llm        : {caps['llm']}")
    if not caps["pdf"]:
        print("    PDF engine unavailable; cannot continue")
        return 1

    # ── doctor session ───────────────────────────────────────────
    email = f"doc-{uuid.uuid4().hex[:8]}@example.com"
    status, reg = req("POST", "/api/v1/auth/register/doctor", {
        "email": email,
        "password": "a-strong-password",
        "display_name": "Dr Anita Desai",
        "specialty": "General Medicine",
        "clinic_name": "Desai Clinic & Diagnostics",
        "nmc_number": "MCI-2019-4471",
        "registration_state": "Maharashtra Medical Council",
    })
    if status != 201:
        print(f"  doctor registration failed: {status} {reg}")
        return 1
    auth = {"Authorization": f"Bearer {reg['token']}"}
    print(f"\n[2] doctor session   {email}")

    # ── patient with a correction in its record ──────────────────
    status, patients = req("GET", "/api/v1/patients")
    if patients:
        patient_id = patients[0]["id"]
        patient_name = patients[0]["display_name"]
    else:
        status, p = req("POST", "/api/v1/patients", {
            "display_name": "Rakesh Mehta", "age": 54, "phone": "9876543210",
        })
        patient_id, patient_name = p["id"], "Rakesh Mehta"
    print(f"[3] patient           {patient_name}  {patient_id[:8]}")

    created = []

    # ── prescription ─────────────────────────────────────────────
    print("\n[4] prescription")
    status, res = req("POST", "/api/v1/documents/prescription", {
        "patient_id": patient_id,
        "items": [
            {"drug": "Metformin", "dose": "1000 mg", "frequency": "twice daily",
             "duration": "30 days", "timing": "After food, morning and night",
             "instruction": "Do not take if vomiting"},
            {"drug": "Telmisartan", "dose": "40 mg", "frequency": "once daily",
             "duration": "ongoing", "timing": "Morning, before food"},
            {"drug": "Atorvastatin", "dose": "10 mg", "frequency": "at bedtime",
             "duration": "90 days", "timing": "Bedtime, after food"},
            {"drug": "Insulin Glargine", "dose": "10 units",
             "frequency": "once daily", "duration": "ongoing",
             "timing": "Subcutaneous, at bedtime",
             "instruction": "Rotate injection site"},
        ],
        "notes": "Renal function has declined (eGFR 24). Metformin discontinued "
                 "and insulin started.\n\nReview injection technique at the "
                 "next visit.",
        "follow_up_days": 14,
    }, 120, auth)
    if status == 201:
        created.append(res)
        print(f"    OK  {res['reference']}  {res['size_bytes']:,} bytes  "
              f"issued={res['issued_to_patient']}")
    else:
        failures += 1
        print(f"    FAIL {status} {str(res)[:180]}")

    # ── certificate ──────────────────────────────────────────────
    print("\n[5] medical certificate")
    status, res = req("POST", "/api/v1/documents/certificate", {
        "patient_id": patient_id,
        "certificate_type": "Fitness Certificate",
        "body_text": "This is to certify that Mr. Rakesh Mehta has been "
                     "examined at this clinic on the date below and is fit "
                     "for work.\n\nHe is currently under treatment for type 2 "
                     "diabetes mellitus, well controlled on the present "
                     "regimen. He is advised to follow his medication schedule "
                     "and dietary advice.",
    }, 120, auth)
    if status == 201:
        created.append(res)
        print(f"    OK  {res['reference']}  {res['size_bytes']:,} bytes")
    else:
        failures += 1
        print(f"    FAIL {status} {str(res)[:180]}")

    # ── invoice ──────────────────────────────────────────────────
    print("\n[6] invoice")
    status, res = req("POST", "/api/v1/documents/invoice", {
        "patient_id": patient_id,
        "lines": [
            {"description": "Consultation - General Medicine",
             "quantity": 1, "unit_price": 600},
            {"description": "HbA1c", "quantity": 1, "unit_price": 450},
            {"description": "Renal function panel", "quantity": 1,
             "unit_price": 900},
            {"description": "Follow-up review", "quantity": 1, "unit_price": 300},
        ],
        "discount": 150,
    }, 120, auth)
    if status == 201:
        created.append(res)
        c = res.get("content", {})
        print(f"    OK  {res['reference']}  {res['size_bytes']:,} bytes")
        print(f"    subtotal {c.get('subtotal')}  discount {c.get('discount')}  "
              f"total {c.get('total')}")
        # 600+450+900+300 = 2250, minus 150 = 2100
        if c.get("total") != 2100:
            failures += 1
            print(f"    FAIL: total should be 2100, got {c.get('total')}")
    else:
        failures += 1
        print(f"    FAIL {status} {str(res)[:180]}")

    # ── weekly report (LLM) ──────────────────────────────────────
    print("\n[7] weekly report (LLM drafted)")
    status, res = req("POST", "/api/v1/documents/weekly-report", {
        "patient_id": patient_id,
        "days": 400,
    }, 240, auth)
    if status == 201:
        created.append(res)
        c = res.get("content", {})
        print(f"    OK  {res['reference']}  {res['size_bytes']:,} bytes")
        print(f"    findings={len(c.get('findings', []))} "
              f"corrections={len(c.get('corrections', []))} "
              f"meds={len(c.get('medications', []))}")
        summary = (c.get("summary") or "").replace("\n", " ")
        print(f"    summary: {summary[:240]}")
        if not summary:
            failures += 1
            print("    FAIL: empty summary")
    else:
        failures += 1
        print(f"    FAIL {status} {str(res)[:180]}")

    # ── download and validate the bytes ──────────────────────────
    if created:
        print("\n[8] download and PDF validation")
        os.makedirs(out_dir, exist_ok=True)
        for doc in created:
            status, link = req(
                "GET", f"/api/v1/documents/{doc['id']}/download", None, 60, auth
            )
            if status != 200 or not isinstance(link, dict):
                print(f"    {doc['reference']}: download link failed {status}")
                failures += 1
                continue
            try:
                with urllib.request.urlopen(_reachable(link["url"]), timeout=60) as r:
                    data = r.read()
            except Exception as exc:  # noqa: BLE001
                print(f"    {doc['reference']}: fetch failed {exc}")
                failures += 1
                continue

            if data[:5] != b"%PDF-":
                print(f"    {doc['reference']}: NOT A PDF ({data[:8]!r})")
                failures += 1
                continue

            path = os.path.join(out_dir, f"{doc['reference']}-{doc['kind']}.pdf")
            with open(path, "wb") as fh:
                fh.write(data)
            pages = data.count(b"/Type /Page") or data.count(b"/Type/Page")
            print(f"    {doc['reference']}-{doc['kind']}: "
                  f"{len(data):,} bytes, ~{pages} pages -> saved")

    # ── superseding a prescription ───────────────────────────────
    if created:
        print("\n[9] supersede a prescription (versioned, not overwritten)")
        rx = created[0]
        status, res2 = req("POST", "/api/v1/documents/prescription", {
            "patient_id": patient_id,
            "items": [{"drug": "Metformin", "dose": "500 mg",
                       "frequency": "twice daily", "duration": "7 days",
                       "timing": "After food"}],
            "notes": "Dose reduced pending renal review.",
            "supersedes_id": rx["id"],
        }, 120, auth)
        if status == 201:
            print(f"    OK  {res2['reference']} supersedes {rx['reference']}")
            status, listed = req(
                "GET", f"/api/v1/documents/patient/{patient_id}", None, 60, auth
            )
            rx_list = [d for d in (listed or []) if d["kind"] == "prescription"]
            # Earlier runs leave prescriptions behind for the same seeded
            # patient, so this asserts on the pair just created rather
            # than on a total.
            old = next(
                (d for d in rx_list if d["reference"] == rx["reference"]), None
            )
            new = next(
                (d for d in rx_list if d["reference"] == res2["reference"]), None
            )
            current = [d for d in rx_list if d["is_current"]]

            print(f"    {len(rx_list)} prescription versions total, "
                  f"{len(current)} current")
            if old is None:
                failures += 1
                print("    FAIL: the original was removed rather than retained")
            elif old["is_current"]:
                failures += 1
                print("    FAIL: the original is still marked current")
            else:
                print("    OK  original retained, marked superseded")
            if new is not None and not new["is_current"]:
                failures += 1
                print("    FAIL: the replacement is not current")
        else:
            failures += 1
            print(f"    FAIL {status} {str(res2)[:180]}")

    print("\n" + "=" * 66)
    print("ALL DOCUMENT TESTS PASSED" if failures == 0
          else f"{failures} FAILURE(S)")
    print("=" * 66)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
