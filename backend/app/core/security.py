"""Security primitives: password hashing and signed tokens.

Deliberately stdlib-only (hashlib, hmac, secrets). No passlib, no bcrypt
binary, no PyJWT. A final-year project should not acquire a native
dependency for two functions, and PBKDF2-HMAC-SHA256 is exactly what
passlib would have called anyway.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

# ── passwords ───────────────────────────────────────────────────────

PBKDF2_ROUNDS = 240_000
SALT_BYTES = 16


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256, 240k rounds.

    The round count is the work factor and is stored with the digest, so
    it can be raised later without invalidating existing hashes.
    """
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time verification. Never raises on a malformed hash."""
    try:
        algo, rounds, salt_b64, digest_b64 = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = _unb64(salt_b64)
        expected = _unb64(digest_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt, int(rounds)
        )
        return hmac.compare_digest(expected, actual)
    except (ValueError, TypeError, AttributeError):
        return False


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


# ── tokens ──────────────────────────────────────────────────────────

ALGORITHM = "HS256"


@dataclass(frozen=True)
class TokenError(Exception):
    message: str


class TokenSigner:
    """HMAC-signed JSON tokens.

    Not a full JWT library - just a header, a payload, and a signature.
    A standard JWT is unnecessary here because tokens are never verified
    by a third party.
    """

    def __init__(self, secret: str, ttl_seconds: int = 60 * 60 * 24 * 7):
        if len(secret) < 32:
            raise ValueError("token secret must be at least 32 characters")
        self._secret = secret.encode()
        self.ttl = ttl_seconds

    def sign(self, payload: Dict[str, Any], ttl: Optional[int] = None) -> str:
        now = int(time.time())
        body = {**payload, "iat": now, "exp": now + (ttl or self.ttl)}
        raw = _b64(
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
        )
        return f"{raw}.{_b64(self._signature(raw))}"

    def verify(self, token: str) -> Dict[str, Any]:
        try:
            raw, signature = token.rsplit(".", 1)
        except ValueError as exc:
            raise TokenError("malformed token") from exc

        if not hmac.compare_digest(_unb64(signature), self._signature(raw)):
            raise TokenError("bad signature")

        try:
            body = json.loads(_unb64(raw))
        except (ValueError, TypeError) as exc:
            raise TokenError("malformed payload") from exc

        if int(body.get("exp", 0)) < int(time.time()):
            raise TokenError("token expired")
        return body

    def _signature(self, raw: str) -> bytes:
        return hmac.new(self._secret, raw.encode(), hashlib.sha256).digest()


# ── email ───────────────────────────────────────────────────────────

#: Deliberately permissive but structurally real. Rejecting valid
#: addresses is worse than accepting a rare malformed one, because the
#: patient who gets locked out cannot receive their prescriptions.
_EMAIL_LOCAL = r"[A-Za-z0-9._%+\-]+"
_EMAIL_DOMAIN = r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+"
EMAIL_RE = __import__("re").compile(
    rf"^{_EMAIL_LOCAL}@{_EMAIL_DOMAIN}$"
)


def normalise_email(raw: Optional[str]) -> Optional[str]:
    """Lowercase and validate.

    Normalisation matters for identity: `Rakesh@Gmail.com` and
    `rakesh@gmail.com` are one person, and treating them as two would
    silently break the patient-linking guarantee.
    """
    if not raw:
        return None
    cleaned = raw.strip().lower()
    return cleaned if EMAIL_RE.match(cleaned) else None


def email_is_valid(raw: Optional[str]) -> bool:
    return normalise_email(raw) is not None
