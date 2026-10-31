"""Verify the frontend is actually served and wired to the API.

Checks the dev server, the production build output, and that the SPA can
reach the backend through whatever origin it is served from. Run with the
dev server (5173) or after `npm run build` + nginx (8080) up.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

DEV = "http://localhost:5173"
PROD = "http://localhost:8080"
API = "http://localhost:8000"


def get(url: str, timeout: int = 12) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def post(url: str, payload: dict, timeout: int = 60) -> tuple[int, str]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def main() -> int:
    origin = DEV
    print("=" * 62)
    print("SOLUM - frontend verification")
    print("=" * 62)

    status, body = get(DEV)
    if status != 200:
        print(f"dev server not serving ({status}) - trying production build")
        origin = PROD
        status, body = get(PROD)
        if status != 200:
            print(f"  neither :5173 nor :8080 is serving. status={status}")
            print(f"  {body[:180]}")
            return 1

    print(f"[1] origin            {origin}  -> HTTP {status}")
    title = ""
    if "<title>" in body:
        title = body.split("<title>")[1].split("</title>")[0]
    print(f"    title            {title}")

    # module graph must transform without error
    status, mod = get(f"{origin}/src/main.tsx")
    print(f"[2] main.tsx          HTTP {status}  ({len(mod)} bytes served)")
    if status != 200:
        print("    FAIL: entry module did not transform")
        return 1

    status, css = get(f"{origin}/src/styles.css")
    print(f"[3] styles.css        HTTP {status}  ({len(css)} bytes)")

    status, dna = get(f"{origin}/src/components/DNAHelix.tsx")
    print(f"[4] DNAHelix.tsx      HTTP {status}  ({len(dna)} bytes)")

    # API reachability from this origin
    status, health = get(f"{origin}/health")
    print(f"[5] /health via origin HTTP {status}")
    if status == 200:
        print(f"    {health[:120]}")
    else:
        print(f"    note: no proxy on {origin}; the SPA will call :8000 directly")

    status, patients = get(f"{origin}/api/v1/patients")
    print(f"[6] /api/v1/patients  HTTP {status}")
    if status != 200:
        print(f"    {patients[:160]}")
        return 1

    try:
        list_ = json.loads(patients)
    except json.JSONDecodeError:
        print("    FAIL: not JSON")
        return 1

    if not list_:
        print("    no patients - run scripts/seed_demo.py")
        return 1

    pid = list_[0]["id"]
    print(f"    {len(list_)} patient(s); using {list_[0]['display_name']}")

    status, tl = get(f"{origin}/api/v1/patients/{pid}/timeline")
    print(f"[7] timeline          HTTP {status}")
    if status == 200:
        stats = json.loads(tl).get("stats", {})
        print(f"    stats            {stats}")

    status, ans = post(
        f"{origin}/api/v1/patients/{pid}/chat",
        {"patient_id": pid, "query": "current potassium level?"},
    )
    print(f"[8] chat              HTTP {status}")
    if status == 200:
        data = json.loads(ans)
        print(f"    intent           {data['intent']}")
        print(f"    latency          {data['latency_ms']} ms")
        print(f"    citations        {len(data['citations'])}")
        print(f"    suppressed       {len(data['suppressed'])}")
        print(f"    degraded         {data['degraded']}")
        print("    answer           " + (data["answer"][:160] or "(empty)"))
    else:
        print(f"    {ans[:200]}")
        return 1

    print("=" * 62)
    print("frontend is serving and talking to the API")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
