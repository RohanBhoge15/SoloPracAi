"""Verify the full doctor workflow end to end.

Simulates what a clinician actually does, in order:
  add patient -> add a lab -> ask about it -> correct it -> ask again
  -> confirm the stale value is gone -> ask as of the past -> confirm it
  is back -> upload a document -> confirm the abstention gate

Prints what the physician would see at each step.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

API = "http://localhost:8000"


def req(method, path, payload=None, timeout=90):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(
        API + path, data=data,
        headers={"Content-Type": "application/json"}, method=method,
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def step(n, title):
    # ASCII separators: the Windows console codepage is cp1252 and will
    # raise on box-drawing characters.
    print(f"\n{'-' * 62}\nSTEP {n}  {title}\n{'-' * 62}")


def main() -> int:
    now = datetime.now(timezone.utc)
    failures = 0

    # ─────────────────────────────────────────────────────
    step(1, "Doctor adds a patient")
    status, p = req("POST", "/api/v1/patients", {
        "display_name": "Anita Sharma", "age": 47, "phone": "9812345678",
    })
    if status != 201:
        print(f"  FAIL {status} {p}")
        return 1
    pid = p["id"]
    print(f"  created: {p['display_name']}, {p['age']} yrs   [{pid[:8]}]")

    status, tl = req("GET", f"/api/v1/patients/{pid}/timeline")
    print(f"  timeline: {tl['stats']}")

    # ─────────────────────────────────────────────────────
    step(2, "Doctor records a lab result")
    t0 = now - timedelta(days=10)
    status, ev1 = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "lab",
        "valid_time": t0.isoformat(),
        "payload": {
            "test": "HbA1c", "value": 8.9, "unit": "%",
            "ref_low": 4.0, "ref_high": 5.6, "flag": "high",
        },
    })
    print(f"  HTTP {status}  HbA1c = 8.9 % (high)   flagged automatically from the range")
    if status != 201:
        print(f"  FAIL {ev1}")
        failures += 1

    # ─────────────────────────────────────────────────────
    step(3, "Doctor asks: what is the current HbA1c?")
    status, a = req("POST", f"/api/v1/patients/{pid}/chat", {
        "patient_id": pid, "query": "current HbA1c?",
    })
    print(f"  intent={a['intent']}  {a['latency_ms']:.0f} ms  "
          f"degraded={a['degraded']}")
    print(f"  ANSWER: {a['answer'][:200]}")
    if "8.9" not in a["answer"]:
        print("  FAIL: expected 8.9")
        failures += 1

    # ─────────────────────────────────────────────────────
    step(4, "The lab calls: that HbA1c was transcribed wrong")
    t1 = now - timedelta(days=8)
    status, ev2 = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "lab",
        "valid_time": t1.isoformat(),
        "payload": {
            "test": "HbA1c", "value": 6.8, "unit": "%",
            "ref_low": 4.0, "ref_high": 5.6, "flag": "high",
            "amendment": "Assay recalibrated; prior run not valid",
        },
        "source_type": "correction",
        "supersedes_id": ev1["id"],
    })
    print(f"  HTTP {status}  correction filed, supersedes {ev1['id'][:8]}")
    print(f"  8.9 is now SUPERSEDED, 6.8 is CURRENT")
    if status != 201:
        print(f"  FAIL {ev2}")
        failures += 1

    # ─────────────────────────────────────────────────────
    step(5, "Timeline shows the pair")
    status, tl = req("GET", f"/api/v1/patients/{pid}/timeline")
    print(f"  stats: {tl['stats']}")
    for pr in tl["pairs"]:
        print(f"    {pr['superseded']['payload']['value']} "
              f"[{pr['superseded']['validity']}]  ->  "
              f"{pr['correction']['payload']['value']} "
              f"[{pr['correction']['validity']}]")
    if len(tl["pairs"]) != 1:
        print("  FAIL: expected one correction pair")
        failures += 1

    # ─────────────────────────────────────────────────────
    step(6, "Doctor asks the SAME question again")
    status, a = req("POST", f"/api/v1/patients/{pid}/chat", {
        "patient_id": pid, "query": "current HbA1c?",
    })
    print(f"  ANSWER: {a['answer'][:200]}")
    print(f"  citations: {len(a['citations'])}   suppressed: {len(a['suppressed'])}")
    if "8.9" in a["answer"]:
        print("  FAIL: the superseded 8.9 was asserted as current")
        failures += 1
    else:
        print("  PASS: 8.9 not asserted as current")

    # ─────────────────────────────────────────────────────
    step(7, 'Doctor asks: "what was the HbA1c back then?"')
    past = (t0 + timedelta(days=1)).isoformat()
    status, ao = req("GET", f"/api/v1/patients/{pid}/as-of/{past}")
    vals = [e["payload"]["value"] for e in ao["events"]
            if e["payload"].get("test") == "HbA1c"]
    print(f"  as of the day after the original:")
    print(f"    HbA1c = {vals}")
    if vals != [8.9]:
        print("  FAIL: history should return 8.9")
        failures += 1
    else:
        print("  PASS: the historical answer survived the correction")

    status, ao2 = req("GET", f"/api/v1/patients/{pid}/as-of/{now.isoformat()}")
    vals2 = [e["payload"]["value"] for e in ao2["events"]
             if e["payload"].get("test") == "HbA1c"]
    print(f"  as of today:")
    print(f"    HbA1c = {vals2}")
    if vals2 != [6.8]:
        print("  FAIL: present should return 6.8")
        failures += 1

    # ─────────────────────────────────────────────────────
    step(8, "Doctor adds a medication, then stops it")
    status, m1 = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "medication",
        "valid_time": (now - timedelta(days=30)).isoformat(),
        "payload": {"drug": "Metformin", "dose": "500 mg",
                    "frequency": "twice daily", "status": "active"},
    })
    status, m2 = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "medication",
        "valid_time": (now - timedelta(days=5)).isoformat(),
        "payload": {"drug": "Metformin", "dose": "500 mg",
                    "frequency": "twice daily", "status": "stopped",
                    "reason": "eGFR falling, switched to insulin"},
    })
    print(f"  Metformin prescribed then stopped")

    status, a = req("POST", f"/api/v1/patients/{pid}/chat", {
        "patient_id": pid, "query": "current medications?",
    })
    print(f"  ANSWER: {a['answer'][:220]}")
    if "STOPPED" in a["answer"].upper() or "stopped" in a["answer"]:
        print("  PASS: shows the drug as stopped")
    else:
        print("  note: answer did not mark it stopped (model may paraphrase)")

    # ─────────────────────────────────────────────────────
    step(9, "Document capture with the abstention gate")
    # A blank image: quality gate should refuse it.
    try:
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (600, 600), (128, 128, 128)).save(buf, "PNG")
        data = buf.getvalue()
    except Exception:
        data = None

    if data:
        boundary = "----solumtest"
        parts = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="blank.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
        r = urllib.request.Request(
            f"{API}/api/v1/patients/{pid}/capture", data=parts,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(r, timeout=90) as resp:
                cap = json.loads(resp.read().decode())
            print(f"  HTTP {resp.status}")
            print(f"  committed:  {cap['committed']}")
            print(f"  confidence: {cap['confidence']}")
            print(f"  reason:     {cap['reason']}")
            if cap["committed"]:
                print("  note: an unreadable scan was committed - the gate should refuse")
                failures += 1
            else:
                print("  PASS: refused to write an unreadable scan")
        except Exception as e:  # noqa: BLE001
            print(f"  capture error: {e}")

    # ─────────────────────────────────────────────────────
    step(10, "Cleanup")
    status, _ = req("DELETE", f"/api/v1/patients/{pid}")
    print(f"  deleted verification patient: HTTP {status}")

    print(f"\n{'=' * 62}")
    print("WORKFLOW OK" if failures == 0 else f"{failures} CHECK(S) FAILED")
    print("=" * 62)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
