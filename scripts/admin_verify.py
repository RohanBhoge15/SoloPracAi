"""SoloPrac Admin — interactive doctor-verification CLI.

Lets the backend/admin team review pending doctor sign-ups from the terminal
without needing Swagger `/docs` or a dedicated UI page. Uses the same three
endpoints as the admin router:

  GET  /admin/verifications/pending
  POST /admin/verifications/{id}/approve
  POST /admin/verifications/{id}/reject   {"reason": "..."}

Usage
─────
  # interactive (walks through every pending doctor)
  python scripts/admin_verify.py

  # non-interactive: just list pending
  python scripts/admin_verify.py --list

  # non-interactive: approve or reject one by ID
  python scripts/admin_verify.py --approve <doctor-uuid>
  python scripts/admin_verify.py --reject  <doctor-uuid> --reason "License blurry"

  # different backend host / different admin
  python scripts/admin_verify.py --base-url http://localhost:8000 \
                                 --email admin@soloprac.io

Credentials
───────────
The script reads the admin password from:
  1. --password flag                    (least secure — leaves in shell history)
  2. $SOLOPRAC_ADMIN_PASSWORD env var   (recommended for scripting)
  3. interactive prompt via getpass     (default — nothing echoed to terminal)

Zero third-party dependencies: uses urllib + json from the stdlib so the
same script runs anywhere Python 3.9+ runs, no `pip install` needed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Any


DEFAULT_BASE_URL = os.environ.get("SOLOPRAC_BASE_URL", "http://localhost:8000")
DEFAULT_ADMIN_EMAIL = os.environ.get("SOLOPRAC_ADMIN_EMAIL", "admin@soloprac.io")


# ── HTTP helpers ─────────────────────────────────────────────────────


def _request(method: str, url: str, *, token: str | None = None, body: dict | None = None) -> Any:
    """Minimal JSON HTTP client with stdlib. Raises on non-2xx with server detail."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:
            pass
        raise SystemExit(f"[HTTP {e.code}] {method} {url}\n  {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"[NETWORK] {method} {url}\n  {e.reason}\n  Is the backend running at {DEFAULT_BASE_URL}?")


def login(base_url: str, email: str, password: str) -> str:
    """POST /auth/login → returns the bearer access_token."""
    resp = _request("POST", f"{base_url}/auth/login", body={"email": email, "password": password})
    tok = resp.get("access_token")
    if not tok:
        raise SystemExit(f"login response missing access_token: {resp}")
    return tok


def list_pending(base_url: str, token: str) -> list[dict]:
    return _request("GET", f"{base_url}/admin/verifications/pending", token=token) or []


def approve(base_url: str, token: str, doctor_id: str) -> dict:
    return _request("POST", f"{base_url}/admin/verifications/{doctor_id}/approve", token=token, body={})


def reject(base_url: str, token: str, doctor_id: str, reason: str) -> dict:
    return _request("POST", f"{base_url}/admin/verifications/{doctor_id}/reject", token=token, body={"reason": reason})


# ── Display helpers ──────────────────────────────────────────────────


def _fmt_doctor(d: dict, idx: int, total: int) -> str:
    lines = [
        f"\n─── {idx}/{total} ─────────────────────────────────────────────────",
        f"  id           : {d.get('id')}",
        f"  name         : {d.get('name')}",
        f"  email        : {d.get('email')}",
        f"  phone        : {d.get('phone') or '—'}",
        f"  speciality   : {d.get('speciality') or '—'}",
        f"  reg number   : {d.get('registration_number') or '—'}",
        f"  clinic       : {d.get('clinic_name') or '—'}",
        f"  address      : {d.get('clinic_address') or '—'}",
        f"  license doc  : {d.get('license_document_path') or '(none uploaded)'}",
        f"  submitted at : {d.get('created_at')}",
    ]
    return "\n".join(lines)


# ── Interactive loop ────────────────────────────────────────────────


def interactive(base_url: str, token: str) -> None:
    pending = list_pending(base_url, token)
    if not pending:
        print("[OK] No doctors pending verification.")
        return

    print(f"[OK] {len(pending)} doctor(s) pending verification.")
    approved = rejected = skipped = 0

    for i, d in enumerate(pending, 1):
        print(_fmt_doctor(d, i, len(pending)))
        while True:
            choice = input("  [a]pprove / [r]eject / [s]kip / [q]uit ? ").strip().lower()
            if choice in ("a", "approve"):
                res = approve(base_url, token, d["id"])
                print(f"  ✓ approved — {res.get('message', '')}")
                approved += 1
                break
            if choice in ("r", "reject"):
                reason = input("  reason: ").strip() or "Verification documents did not meet requirements"
                res = reject(base_url, token, d["id"], reason)
                print(f"  ✗ rejected — {res.get('message', '')}")
                rejected += 1
                break
            if choice in ("s", "skip", ""):
                print("  … skipped")
                skipped += 1
                break
            if choice in ("q", "quit"):
                print(
                    f"\n[SUMMARY] approved={approved}  rejected={rejected}  skipped={skipped + (len(pending) - i)}  (quit early)"
                )
                return
            print("  (please answer a / r / s / q)")

    print(f"\n[SUMMARY] approved={approved}  rejected={rejected}  skipped={skipped}")


# ── CLI ─────────────────────────────────────────────────────────────


def _resolve_password(args) -> str:
    if args.password:
        return args.password
    env_pw = os.environ.get("SOLOPRAC_ADMIN_PASSWORD")
    if env_pw:
        return env_pw
    return getpass.getpass(f"Password for {args.email}: ")


def main() -> int:
    ap = argparse.ArgumentParser(description="SoloPrac admin doctor-verification CLI")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Backend base URL (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--email", default=DEFAULT_ADMIN_EMAIL, help=f"Admin email (default: {DEFAULT_ADMIN_EMAIL})")
    ap.add_argument("--password", help="Admin password (else env SOLOPRAC_ADMIN_PASSWORD or prompt)")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="Just list pending doctors and exit")
    mode.add_argument("--approve", metavar="DOCTOR_ID", help="Approve one doctor by UUID and exit")
    mode.add_argument("--reject", metavar="DOCTOR_ID", help="Reject one doctor by UUID and exit")

    ap.add_argument(
        "--reason",
        default="Verification documents did not meet requirements",
        help="Rejection reason (used with --reject)",
    )

    args = ap.parse_args()
    password = _resolve_password(args)

    print(f"[..] Logging in {args.email} @ {args.base_url}")
    token = login(args.base_url, args.email, password)
    print("[OK] Authenticated.")

    if args.list:
        pending = list_pending(args.base_url, token)
        if not pending:
            print("[OK] No pending doctors.")
            return 0
        print(f"[OK] {len(pending)} pending:")
        for i, d in enumerate(pending, 1):
            print(_fmt_doctor(d, i, len(pending)))
        return 0

    if args.approve:
        res = approve(args.base_url, token, args.approve)
        print(json.dumps(res, indent=2))
        return 0

    if args.reject:
        res = reject(args.base_url, token, args.reject, args.reason)
        print(json.dumps(res, indent=2))
        return 0

    # default: interactive loop
    interactive(args.base_url, token)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[abort]")
        sys.exit(130)
