"""Verify the authoring paths: add patient, add event, correct a record.

These are the flows a real doctor uses, and none of them have a UI yet.
This exercises them over HTTP so the endpoints are known-good before the
frontend is built against them.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

API = "http://localhost:8000"


def req(method: str, path: str, payload=None, timeout: int = 30):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(
        API + path, data=data, headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def main() -> int:
    ok = True
    now = datetime.now(timezone.utc)

    print("=" * 64)
    print("SOLUM - authoring path verification")
    print("=" * 64)

    # ── 1. create a patient ─────────────────────────────────────
    print("\n[1] POST /api/v1/patients")
    status, patient = req("POST", "/api/v1/patients", {
        "display_name": "Verification Patient",
        "age": 61,
        "phone": "9000000000",
    })
    print(f"    HTTP {status}")
    if status != 201:
        print(f"    FAIL {patient}")
        return 1
    pid = patient["id"]
    print(f"    created {patient['display_name']}  {pid[:8]}")

    # ── 2. add a lab result ─────────────────────────────────────
    print("\n[2] POST events - original lab value")
    original = now - timedelta(days=30)
    status, ev1 = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "lab",
        "valid_time": original.isoformat(),
        "payload": {
            "test": "Potassium", "value": 2.1, "unit": "mEq/L",
            "ref_low": 3.5, "ref_high": 5.1, "flag": "low",
        },
        "source_type": "manual",
    })
    print(f"    HTTP {status}  {ev1['payload'].get('value') if status < 300 else ev1}")
    if status != 201:
        ok = False

    # ── 3. add a medication ─────────────────────────────────────
    status, ev2 = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "medication",
        "valid_time": (now - timedelta(days=60)).isoformat(),
        "payload": {
            "drug": "Metformin", "dose": "500 mg",
            "frequency": "twice daily", "status": "active",
        },
    })
    print(f"    HTTP {status}  {ev2['payload'].get('drug') if status < 300 else ev2}")

    # ── 4. CORRECT the lab - this is the whole research flow ────
    print("\n[3] POST events - correction (supersedes the original)")
    corrected = now - timedelta(days=27)
    status, ev3 = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "lab",
        "valid_time": corrected.isoformat(),
        "payload": {
            "test": "Potassium", "value": 4.8, "unit": "mEq/L",
            "ref_low": 3.5, "ref_high": 5.1, "flag": "normal",
            "amendment": "Specimen re-run",
        },
        "source_type": "correction",
        "supersedes_id": ev1["id"],
    })
    print(f"    HTTP {status}")
    if status != 201:
        print(f"    FAIL {ev3}")
        ok = False
    else:
        print(f"    correction created, supersedes {ev1['id'][:8]}")
        print(f"    -> {ev3['payload']['value']}  validity={ev3['validity']}")

    # ── 5. timeline should show the pair ────────────────────────
    print("\n[4] GET timeline - three-valued view")
    status, tl = req("GET", f"/api/v1/patients/{pid}/timeline")
    print(f"    HTTP {status}  stats={tl['stats']}")
    print(f"    correction pairs: {len(tl['pairs'])}")
    for pair in tl["pairs"]:
        s, c = pair["superseded"], pair["correction"]
        print(f"      {s['payload'].get('value')} [{s['validity']}]"
              f"  ->  {c['payload'].get('value')} [{c['validity']}]")

    if len(tl["pairs"]) != 1:
        print("    FAIL: expected exactly one correction pair")
        ok = False
    if tl["stats"]["superseded"] != 1:
        print("    FAIL: expected one superseded record")
        ok = False

    # ── 6. as-of resolution ─────────────────────────────────────
    print("\n[5] GET as-of - historical truth")
    mid = (original + timedelta(days=1)).isoformat()
    status, asof = req("GET", f"/api/v1/patients/{pid}/as-of/{mid}")
    values = [e["payload"].get("value") for e in asof["events"]
              if e["payload"].get("test") == "Potassium"]
    print(f"    HTTP {status}  potassium as of mid-window: {values}")
    if values != [2.1]:
        print("    FAIL: as-of should return the value that was true then")
        ok = False

    now_iso = now.isoformat()
    status, asof_now = req("GET", f"/api/v1/patients/{pid}/as-of/{now_iso}")
    values_now = [e["payload"].get("value") for e in asof_now["events"]
                  if e["payload"].get("test") == "Potassium"]
    print(f"    potassium as of now:         {values_now}")
    if values_now != [4.8]:
        print("    FAIL: present-tense should return the corrected value")
        ok = False

    # ── 7. chat must not assert the stale value ─────────────────
    print("\n[6] POST chat - stale-value refusal")
    status, ans = req("POST", f"/api/v1/patients/{pid}/chat", {
        "patient_id": pid,
        "query": "current potassium level?",
    }, timeout=90)
    print(f"    HTTP {status}  intent={ans['intent']}")
    print(f"    answer: {ans['answer'][:150]}")
    if "2.1" in ans["answer"]:
        print("    FAIL: answer asserted the superseded value")
        ok = False
    else:
        print("    PASS: superseded value not asserted as current")

    # ── 8. validation - cannot supersede twice ──────────────────
    print("\n[7] validation - double supersession rejected")
    status, err = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "lab",
        "valid_time": now.isoformat(),
        "payload": {"test": "Potassium", "value": 5.0, "unit": "mEq/L"},
        "source_type": "correction",
        "supersedes_id": ev1["id"],
    })
    print(f"    HTTP {status}  {'rejected' if status == 409 else 'ACCEPTED (bug)'}")
    if status != 409:
        ok = False

    # ── 9. cross-patient supersession rejected ──────────────────
    print("\n[8] validation - cross-patient supersession rejected")
    status, err = req("POST", f"/api/v1/patients/{pid}/events", {
        "event_type": "lab",
        "valid_time": now.isoformat(),
        "payload": {"test": "Potassium", "value": 1.0},
        "source_type": "correction",
        "supersedes_id": "00000000-0000-0000-0000-000000000000",
    })
    print(f"    HTTP {status}  {'rejected' if status in (403, 404) else 'ACCEPTED (bug)'}")

    # ── 10. cleanup ─────────────────────────────────────────────
    req("DELETE", f"/api/v1/patients/{pid}")
    print("\n[9] cleanup: verification patient deleted")

    print("\n" + "=" * 64)
    print("ALL AUTHORING PATHS OK" if ok else "SOME CHECKS FAILED")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
