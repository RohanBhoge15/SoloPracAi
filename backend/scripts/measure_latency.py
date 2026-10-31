"""Measure real request latencies against the running stack.

Answers the questions a user actually asks about a clinical tool: how long
until the screen moves, how long until the answer arrives, and how long an
unreadable scan takes to be refused.

Every number here is a measurement against the live API with real models
loaded, not a design target.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

API = "http://localhost:8000"

# A valid 1x1 PNG with no readable text: the point is how fast the system
# refuses it rather than guessing at it.
UNREADABLE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6360000002000100ffff0300000600"
    "0557bfabd40000000049454e44ae426082"
)


def req(
    method: str,
    path: str,
    payload: Any = None,
    token: Optional[str] = None,
    timeout: int = 180,
) -> Tuple[int, Any]:
    url = API + path
    headers = {"Content-Type": "application/json"}
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


def upload(
    path: str,
    filename: str,
    content_type: str,
    payload: bytes,
    token: str,
    extra: Optional[Dict[str, str]] = None,
) -> Tuple[int, Any, float]:
    boundary = "----solum" + uuid.uuid4().hex[:12]
    body = b""
    for key, value in (extra or {}).items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        ).encode()
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
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
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode()
            elapsed = (time.perf_counter() - started) * 1000
            return response.status, (json.loads(raw) if raw else None), elapsed
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = raw
        return exc.code, parsed, (time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}", (time.perf_counter() - started) * 1000


def pct(values: List[float], p: int) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int((p / 100) * len(ordered)))
    return ordered[index]


def report(label: str, values: List[float]) -> None:
    if not values:
        print(f"  {label:<34} no samples")
        return
    print(
        f"  {label:<34} p50 {statistics.median(values):7.1f} ms   "
        f"p95 {pct(values, 95):7.1f} ms   max {max(values):7.1f} ms"
    )


def main() -> int:
    stamp = datetime.now(timezone.utc)
    tag = uuid.uuid4().hex[:10]

    status, session = req(
        "POST",
        "/api/v1/auth/register/doctor",
        {
            "email": f"perf.{tag}@solum.test",
            "password": "correct-horse-battery",
            "display_name": f"Dr Perf {tag}",
            "specialty": "General Medicine",
            "clinic_name": "Perf Clinic",
        },
    )
    if status != 201:
        print(f"could not create a doctor: HTTP {status} {session}")
        return 1
    token = session["token"]

    status, patient = req(
        "POST",
        "/api/v1/patients",
        {"display_name": f"Perf Patient {tag}", "age": 50, "email": f"pperf.{tag}@solum.test"},
        token=token,
    )
    if status not in (200, 201):
        print(f"could not create a patient: HTTP {status} {patient}")
        return 1
    pid = patient["id"]

    # Twelve records, so retrieval has something to actually rank.
    tests = [
        ("HbA1c", 8.9, "%"), ("Fasting Glucose", 186, "mg/dL"),
        ("Serum Creatinine", 1.0, "mg/dL"), ("Potassium", 4.8, "mEq/L"),
        ("TSH", 6.61, "mIU/L"), ("Haemoglobin", 11.2, "g/dL"),
        ("Total Cholesterol", 232, "mg/dL"), ("LDL", 154, "mg/dL"),
        ("Uric Acid", 7.1, "mg/dL"), ("Vitamin D", 18, "ng/mL"),
        ("Ferritin", 24, "ng/mL"), ("CRP", 12, "mg/L"),
    ]
    write_times: List[float] = []
    for i, (name, value, unit) in enumerate(tests):
        started = time.perf_counter()
        req(
            "POST",
            f"/api/v1/patients/{pid}/events",
            {
                "event_type": "lab",
                "valid_time": (stamp - timedelta(days=30 * (i + 1))).isoformat(),
                "payload": {"test": name, "value": value, "unit": unit},
            },
            token=token,
        )
        write_times.append((time.perf_counter() - started) * 1000)

    print("\nRECORD WRITE  (create + index, includes Qdrant upsert)")
    report("lab record", write_times)

    # Retrieval-only: the timeline endpoint, no generation at all.
    read_times: List[float] = []
    for _ in range(12):
        started = time.perf_counter()
        req("GET", f"/api/v1/patients/{pid}/timeline", token=token)
        read_times.append((time.perf_counter() - started) * 1000)
    print("\nRETRIEVAL  (timeline: validity graph + hybrid search, no LLM)")
    report("timeline load", read_times)

    # The doctor-facing question path, warm.
    chat_times: List[float] = []
    server_times: List[float] = []
    citation_counts: List[int] = []
    answers: List[str] = []
    questions = [
        "what is her hba1c",
        "what is her potassium level",
        "which test was most abnormal",
        "what is her ldl",
        "summarise her lab results",
    ]
    for q in questions * 2:
        started = time.perf_counter()
        status, answer = req(
            "POST",
            f"/api/v1/patients/{pid}/chat",
            {"patient_id": pid, "query": q},
            token=token,
        )
        chat_times.append((time.perf_counter() - started) * 1000)
        if status == 200 and isinstance(answer, dict):
            server_times.append(answer.get("latency_ms", 0.0))
            citation_counts.append(len(answer.get("citations", [])))
            answers.append(answer.get("answer", ""))

    print("\nQUESTION  (retrieval + LLM generation + provenance check)")
    report("chat round trip", chat_times)
    report("  server-side only", server_times)
    if answers:
        print(
            f"  citations per answer                mean "
            f"{sum(citation_counts) / len(citation_counts):.1f}, "
            f"min {min(citation_counts)}"
        )
        print(f"  answers returned                    {len(answers)}/{len(questions) * 2}")

    # Historical: the path that makes this system different.
    hist_times: List[float] = []
    for q in ["what was her hba1c as of 90 days ago", "what was her hba1c 90 days ago"]:
        started = time.perf_counter()
        req(
            "POST",
            f"/api/v1/patients/{pid}/chat",
            {"patient_id": pid, "query": q},
            token=token,
        )
        hist_times.append((time.perf_counter() - started) * 1000)
    print("\nHISTORICAL QUESTION  (as-of resolution + generation)")
    report("as-of round trip", hist_times)

    # Correction, then the same question again.
    status, events = req("GET", f"/api/v1/patients/{pid}/events", token=token)
    hba1c = next((e for e in events if e.get("payload", {}).get("test") == "HbA1c"), None)
    if hba1c:
        started = time.perf_counter()
        req(
            "POST",
            f"/api/v1/patients/{pid}/events",
            {
                "event_type": "lab",
                "source_type": "correction",
                "supersedes_id": hba1c["id"],
                "valid_time": (stamp - timedelta(days=1)).isoformat(),
                "payload": {"test": "HbA1c", "value": 6.8, "unit": "%"},
            },
            token=token,
        )
        corr = (time.perf_counter() - started) * 1000
        print(f"\nCORRECTION  {corr:7.1f} ms")

        started = time.perf_counter()
        _, after = req(
            "POST",
            f"/api/v1/patients/{pid}/chat",
            {"patient_id": pid, "query": "what is her hba1c"},
            token=token,
        )
        again = (time.perf_counter() - started) * 1000
        print(f"  re-ask after correction            {again:7.1f} ms")
        if isinstance(after, dict):
            text = after.get("answer", "")
            print(f"  asserts 8.9 (must be False)       {'8.9' in text}")
            print(f"  asserts 6.8 (must be True)        {'6.8' in text}")
            print(f"  stale flag (must be False)        {after.get('has_stale_assertion')}")
            print("\n  answer text:")
            for line in text.splitlines() or [text]:
                print(f"    | {line}")

    # OCR: how fast is a refusal.
    ocr_times: List[float] = []
    committed_flags: List[bool] = []
    for i in range(3):
        status, capture, elapsed = upload(
            f"/api/v1/patients/{pid}/capture",
            f"blank{i}.png",
            "image/png",
            UNREADABLE_PNG,
            token,
            {"event_type": "lab"},
        )
        ocr_times.append(elapsed)
        if isinstance(capture, dict):
            committed_flags.append(bool(capture.get("committed")))
    print("\nOCR  (unreadable scan, must be refused)")
    report("capture + OCR", ocr_times)
    if committed_flags:
        print(f"  committed anything (must be False) {any(committed_flags)}")

    # Document generation, which is what the patient actually waits for.
    rx_times: List[float] = []
    for i in range(3):
        started = time.perf_counter()
        status, rx = req(
            "POST",
            "/api/v1/documents/prescription",
            {
                "patient_id": pid,
                "items": [
                    {"drug": "Metformin", "dose": "500 mg", "frequency": "Twice daily"},
                    {"drug": "Atorvastatin", "dose": "10 mg", "frequency": "Once daily"},
                ],
                "notes": f"Perf run {i}.",
                "send_to_patient": True,
            },
            token=token,
        )
        rx_times.append((time.perf_counter() - started) * 1000)
        if i == 0 and status in (200, 201):
            print(f"  PDF size                           {rx.get('size_bytes', 0):,} bytes")
    print("\nDOCUMENT  (prescription -> rendered PDF -> object storage)")
    report("prescription", rx_times)

    print("\nAll figures are wall-clock against the live stack, real models loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
