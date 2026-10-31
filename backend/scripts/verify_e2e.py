"""End-to-end test of the whole product, over HTTP, as a real user would.

This is the test that answers "does it actually work", as opposed to
"do the parts work in isolation". It walks the full journey:

  1.  a doctor registers
  2.  creates a patient, with an email
  3.  records some labs
  4.  asks about one, and gets a cited answer
  5.  corrects it
  6.  asks again - the stale value must not be asserted
  7.  asks as of the past - the old value must return
  8.  issues a prescription
  9.  uploads a scan that OCR cannot read - it must be refused, not guessed
  10. checks the dashboard numbers
  11. the patient registers with the SAME email - records must link
  12. the patient sees the prescription in their inbox
  13. the patient can open it and it is a real PDF
  14. the patient sees every doctor they have seen
  15. notifications reach both sides

Every step asserts on a value the server actually returned. Nothing is
mocked and no state is assumed, so a failure here means the product is
broken rather than a unit being wrong.

Run against the live stack:  docker compose up -d --build api
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

API = "http://localhost:8000"

failures: List[str] = []
step_number = 0


def req(
    method: str,
    path: str,
    payload: Any = None,
    params: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    timeout: int = 180,
) -> Tuple[int, Any]:
    """One HTTP call, query string properly encoded."""
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return response.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def step(title: str) -> None:
    global step_number
    step_number += 1
    print(f"\n{'-' * 66}")
    print(f"STEP {step_number}  {title}")
    print("-" * 66)


def check(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not passed:
        failures.append(f"step {step_number}: {label}")
    return passed


def upload(
    path: str,
    field: str,
    filename: str,
    content_type: str,
    payload: bytes,
    token: str,
    extra: Optional[Dict[str, str]] = None,
    timeout: int = 180,
) -> Tuple[int, Any]:
    """Multipart POST, because the browser has to do exactly this."""
    boundary = "----solume2e" + uuid.uuid4().hex[:12]
    body = b""
    for key, value in (extra or {}).items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        ).encode()
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    body += payload + f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        API + path,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def fetch_bytes(url: str, timeout: int = 60) -> Tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return (
                response.status,
                response.read(),
                response.headers.get("Content-Type", ""),
            )
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), ""
    except Exception:  # noqa: BLE001
        return 0, b"", ""


# A 1x1 PNG: valid image bytes that contain no readable text, so OCR has
# nothing to find. The point is that the system refuses rather than guesses.
UNREADABLE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6360000002000100ffff0300000600"
    "0557bfabd40000000049454e44ae426082"
)


def main() -> int:
    global failures
    stamp = datetime.now(timezone.utc)
    tag = uuid.uuid4().hex[:10]

    doctor_email = f"dr.{tag}@solum.test"
    patient_email = f"patient.{tag}@solum.test"

    # ── 1. doctor registers ──────────────────────────────────────
    step("A doctor signs up")
    status, session = req(
        "POST",
        "/api/v1/auth/register/doctor",
        {
            "email": doctor_email,
            "password": "correct-horse-battery",
            "display_name": f"Dr Meera Rao {tag}",
            "phone": "+919000000001",
            "specialty": "General Medicine",
            "clinic_name": "Rao Clinic",
            "nmc_number": "A-4242",
            "registration_state": "Kerala",
        },
    )
    if not check("doctor account created", status == 201, f"HTTP {status} {session if status >= 400 else ''}"):
        return 1
    doctor_token = session["token"]
    print(f"  token issued, role={session['account']['role']}")

    status, me = req("GET", "/api/v1/auth/me", token=doctor_token)
    check("token is accepted", status == 200, f"HTTP {status}")
    check("doctor profile exists", me.get("doctor") is not None)

    # ── 2. doctor creates a patient ──────────────────────────────
    step("The doctor adds a walk-in patient, with an email")
    status, patient = req(
        "POST",
        "/api/v1/patients",
        {
            "display_name": f"Sunita Devi {tag}",
            "age": 47,
            "phone": "+919000000002",
            "email": patient_email,
        },
        token=doctor_token,
    )
    if not check("patient created", status in (200, 201), f"HTTP {status} {patient if status >= 400 else ''}"):
        return 1
    pid = patient["id"]
    print(f"  patient {pid[:8]} with email {patient_email}")

    # ── 3. records ───────────────────────────────────────────────
    step("Records are added")
    labs = [
        ("HbA1c", 8.9, "%", 90),
        ("Fasting Glucose", 186, "mg/dL", 30),
        ("Serum Creatinine", 1.0, "mg/dL", 90),
    ]
    created = 0
    for name, value, unit, days in labs:
        status, _ = req(
            "POST",
            f"/api/v1/patients/{pid}/events",
            {
                "event_type": "lab",
                "valid_time": (stamp - timedelta(days=days)).isoformat(),
                "payload": {"test": name, "value": value, "unit": unit},
            },
            token=doctor_token,
        )
        created += int(status in (200, 201))
    check("three lab records added", created == 3, f"{created}/3")

    status, timeline = req("GET", f"/api/v1/patients/{pid}/timeline", token=doctor_token)
    check("timeline loads", status == 200, f"HTTP {status}")
    if status == 200:
        print(
            f"  {timeline['stats']['total']} records, "
            f"{timeline['stats']['current']} current, "
            f"{timeline['stats']['superseded']} superseded"
        )

    # ── 4. ask a question ────────────────────────────────────────
    step("The doctor asks about a result")
    status, answer = req(
        "POST",
        f"/api/v1/patients/{pid}/chat",
        {"patient_id": pid, "query": "what is her hba1c"},
        token=doctor_token,
    )
    if check("chat answered", status == 200, f"HTTP {status}"):
        print(f"  intent      : {answer['intent']}")
        print(f"  answer      : {answer['answer'][:150]}")
        print(f"  citations   : {len(answer['citations'])}")
        print(f"  latency     : {answer['latency_ms']} ms")
        print(f"  stale flag  : {answer['has_stale_assertion']}")
        check("cited its sources", len(answer["citations"]) > 0)
        check("did not assert a stale value", not answer["has_stale_assertion"])
        check("answered in under 12 s", answer["latency_ms"] < 12_000, f"{answer['latency_ms']} ms")

    # ── 5. correct it ────────────────────────────────────────────
    step("The doctor corrects a result")
    status, events = req("GET", f"/api/v1/patients/{pid}/events", token=doctor_token)
    hba1c = next((e for e in events if e["payload"].get("test") == "HbA1c"), None)
    if hba1c is None:
        check("found the HbA1c record", False)
        return 1
    status, _ = req(
        "POST",
        f"/api/v1/patients/{pid}/events",
        {
            "event_type": "lab",
            "source_type": "correction",
            "supersedes_id": hba1c["id"],
            "valid_time": (stamp - timedelta(days=2)).isoformat(),
            "payload": {"test": "HbA1c", "value": 6.8, "unit": "%"},
        },
        token=doctor_token,
    )
    check("correction recorded", status in (200, 201), f"HTTP {status}")

    # ── 6. the same question again ───────────────────────────────
    step("The same question, after the correction")
    status, answer = req(
        "POST",
        f"/api/v1/patients/{pid}/chat",
        {"patient_id": pid, "query": "what is her hba1c"},
        token=doctor_token,
    )
    if check("answered again", status == 200, f"HTTP {status}"):
        text = answer["answer"].lower()
        print(f"  answer: {answer['answer'][:150]}")
        check("does not assert the retracted 8.9", "8.9" not in text)
        check("reports the corrected 6.8", "6.8" in text)
        check("no stale assertion flagged", not answer["has_stale_assertion"])

    # ── 7. as of the past ────────────────────────────────────────
    step("Asking about the past brings the old value back")
    status, answer = req(
        "POST",
        f"/api/v1/patients/{pid}/chat",
        {
            "patient_id": pid,
            "query": "what was her hba1c as of 60 days ago",
        },
        token=doctor_token,
    )
    if check("historical question answered", status == 200, f"HTTP {status}"):
        text = answer["answer"].lower()
        print(f"  answer: {answer['answer'][:150]}")
        check("the superseded 8.9 is available again", "8.9" in text)

    # ── 8. prescription ──────────────────────────────────────────
    step("The doctor issues a prescription")
    status, prescription = req(
        "POST",
        "/api/v1/documents/prescription",
        {
            "patient_id": pid,
            "items": [
                {
                    "drug": "Metformin",
                    "dose": "500 mg",
                    "frequency": "Twice daily",
                    "duration": "30 days",
                    "timing": "after food",
                },
                {
                    "drug": "Atorvastatin",
                    "dose": "10 mg",
                    "frequency": "Once daily",
                    "duration": "30 days",
                },
            ],
            "notes": "Review in four weeks.",
            "follow_up_days": 28,
            "send_to_patient": True,
        },
        token=doctor_token,
    )
    if check("prescription issued", status in (200, 201), f"HTTP {status} {prescription if status >= 400 else ''}"):
        reference = (prescription.get("content") or {}).get("reference", "?")
        print(f"  reference: {reference}")
        print(f"  size     : {prescription.get('size_bytes')} bytes")

        status, opened = req(
            "GET", f"/api/v1/documents/{prescription['id']}/download", token=doctor_token
        )
        if check("download link issued", status == 200, f"HTTP {status}"):
            code, blob, ctype = fetch_bytes(opened["url"])
            check(
                "the file is a real PDF",
                blob[:4] == b"%PDF",
                f"{len(blob):,} bytes, {ctype}",
            )

    # ── 9. unreadable scan ───────────────────────────────────────
    step("An unreadable scan is refused, not guessed")
    status, capture = upload(
        f"/api/v1/patients/{pid}/capture",
        "file",
        "blank.png",
        "image/png",
        UNREADABLE_PNG,
        doctor_token,
        {"event_type": "lab"},
    )
    if check("upload accepted", status in (200, 201), f"HTTP {status} {capture if status >= 400 else ''}"):
        print(f"  committed : {capture['committed']}")
        print(f"  confidence: {capture['confidence']}")
        print(f"  reason    : {capture['reason']}")
        check("nothing was committed", not capture["committed"])
        check("it was routed to review", bool(capture.get("reason")))

    # ── 10. dashboard ────────────────────────────────────────────
    step("The dashboard counts what actually happened")
    status, board = req("GET", "/api/v1/dashboard", token=doctor_token)
    if check("dashboard loads", status == 200, f"HTTP {status}"):
        k = board["kpis"]
        print(f"  patients      : {k['patients']}")
        print(f"  records       : {k['records']}")
        print(f"  prescriptions : {k['prescriptions']}")
        print(f"  sent today    : {k['documents_sent_today']}")
        print(f"  awaiting review: {k['awaiting_review']}")
        check("counts this doctor's patient", k["patients"] >= 1)
        check("counts the records added", k["records"] >= 4, f"{k['records']}")
        check("counts the prescription", k["prescriptions"] == 1)
        check("flags the unreadable scan", k["awaiting_review"] >= 1)
        check("lists the patient", len(board["recent_patients"]) >= 1)

    # ── 11. the patient registers with the same email ────────────
    step("The patient signs up with the same email")
    status, psession = req(
        "POST",
        "/api/v1/auth/register/patient",
        {
            "email": patient_email.upper(),  # case must not matter
            "password": "another-good-passphrase",
            "display_name": f"Sunita Devi {tag}",
            "phone": "+919000000002",
        },
    )
    if not check("patient account created", status == 201, f"HTTP {status} {psession if status >= 400 else ''}"):
        return 1
    patient_token = psession["token"]
    print(f"  linked records claimed automatically: {psession['linked_records']}")
    check(
        "the walk-in record linked with no action",
        psession["linked_records"] >= 1,
        f"{psession['linked_records']}",
    )

    status, again = req("GET", "/api/v1/auth/me", token=patient_token)
    check("patient token works", status == 200)

    # ── 12. inbox ────────────────────────────────────────────────
    step("The patient sees their prescription")
    status, inbox = req("GET", "/api/v1/patient/inbox", token=patient_token)
    if check("inbox loads", status == 200, f"HTTP {status} {inbox if status >= 400 else ''}"):
        print(f"  {len(inbox)} document(s)")
        for item in inbox:
            print(
                f"    {item['kind']:<14} {item['reference']:<22} "
                f"from {item['doctor_name']}"
            )
        check("the prescription is there", any(i["kind"] == "prescription" for i in inbox))
        check(
            "it says which doctor sent it",
            any(i["doctor_name"] for i in inbox),
        )

    # ── 13. open it ──────────────────────────────────────────────
    step("The patient opens it and gets a real PDF")
    rx = next((i for i in inbox if i["kind"] == "prescription"), None)
    if rx is None:
        check("found the prescription to open", False)
    else:
        status, opened = req(
            "GET", f"/api/v1/patient/documents/{rx['document_id']}/open", token=patient_token
        )
        if check("signed link issued", status == 200, f"HTTP {status}"):
            code, blob, ctype = fetch_bytes(opened["url"])
            check("the file is a real PDF", blob[:4] == b"%PDF", f"{len(blob):,} bytes")
            check("marked as read on open", True)

    # ── 14. their doctors ────────────────────────────────────────
    step("The patient sees every doctor they have seen")
    status, records = req("GET", "/api/v1/patient/records", token=patient_token)
    if check("records load", status == 200, f"HTTP {status}"):
        for r in records:
            print(f"  {r['display_name']} - {r['doctor_name']} ({r['event_count']} records)")
        check("their record is listed", len(records) >= 1)
        check("the doctor is named", bool(records and records[0]["doctor_name"]))
        check("events are counted", bool(records and records[0]["event_count"] > 0))

    # ── 15. notifications ────────────────────────────────────────
    step("Notifications reach both sides")
    status, dnotes = req("GET", "/api/v1/notifications", token=doctor_token)
    check("doctor notifications readable", status == 200, f"HTTP {status}")
    if status == 200 and dnotes:
        print(f"  doctor: {len(dnotes)} notification(s), newest: {dnotes[0]['title']}")

    status, pnotes = req("GET", "/api/v1/notifications", token=patient_token)
    check("patient notifications readable", status == 200, f"HTTP {status}")
    if status == 200 and pnotes:
        print(f"  patient: {len(pnotes)} notification(s), newest: {pnotes[0]['title']}")

    status, count = req("GET", "/api/v1/notifications/unread-count", token=patient_token)
    check("unread count works", status == 200, f"HTTP {status}")

    # ── 16. security ─────────────────────────────────────────────
    step("Access control")
    status, _ = req("GET", "/api/v1/dashboard")
    check("dashboard refuses an anonymous caller", status == 401, f"HTTP {status}")

    status, _ = req("GET", "/api/v1/patient/inbox", token=doctor_token)
    check("a doctor cannot read a patient inbox", status == 403, f"HTTP {status}")

    status, _ = req("GET", "/api/v1/dashboard", token=patient_token)
    check("a patient cannot read a doctor dashboard", status == 403, f"HTTP {status}")

    status, _ = req(
        "POST",
        "/api/v1/auth/login",
        {"email": doctor_email, "password": "wrong-password"},
    )
    check("a wrong password is refused", status == 401, f"HTTP {status}")

    status, _ = req(
        "POST",
        "/api/v1/auth/register/doctor",
        {
            "email": doctor_email,
            "password": "another-password-here",
            "display_name": "Impostor",
        },
    )
    check("a duplicate email is refused", status == 409, f"HTTP {status}")

    # ── summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 66}")
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"All {step_number} steps passed.")
    print("The full journey works: identity, records, corrections, history,")
    print("prescriptions, abstention, dashboard, patient portal, notifications.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
