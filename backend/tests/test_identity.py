"""Tests for identity linking.

The product requirement being protected:

    A doctor registers a walk-in patient with an email. The patient
    creates an account later with the same email. They must become one
    identity, across every doctor who has seen them, with no action from
    anyone.

These are the cases that matter, including the ones that should NOT link.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

os.environ.setdefault("POSTGRES_HOST", "localhost")

from app.db.identity import Account, Doctor, IdentityLinkLog  # noqa: E402
from app.db.models import Event, Patient  # noqa: E402
from app.db.session import get_engine, get_session_factory, init_models  # noqa: E402
from app.identity.linker import IdentityLinker  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
async def client():
    await init_models()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    await get_engine().dispose()


def _unp(suffix: str) -> str:
    return f"link-{suffix}-{uuid.uuid4().hex[:8]}@example.com"


# ── security primitives ────────────────────────────────────────────


def test_password_hash_is_salted():
    from app.core.security import hash_password

    a = hash_password("correct horse battery")
    b = hash_password("correct horse battery")
    assert a != b, "identical passwords must not produce identical hashes"
    assert a.startswith("pbkdf2_sha256$")


def test_password_verify():
    from app.core.security import hash_password, verify_password

    h = hash_password("hunter2hunter2")
    assert verify_password("hunter2hunter2", h)
    assert not verify_password("wrong", h)


def test_malformed_hash_is_rejected_not_crashed():
    from app.core.security import verify_password

    assert not verify_password("x", "garbage")
    assert not verify_password("x", "")
    assert not verify_password("x", "pbkdf2_sha256$notanint$a$b")


def test_token_roundtrip():
    from app.core.security import TokenSigner

    signer = TokenSigner(secret="a" * 40, ttl_seconds=60)
    token = signer.sign({"sub": "abc", "role": "doctor"})
    claims = signer.verify(token)
    assert claims["sub"] == "abc"


def test_tampered_token_rejected():
    from app.core.security import TokenError, TokenSigner

    signer = TokenSigner(secret="a" * 40, ttl_seconds=60)
    token = signer.sign({"sub": "abc"})
    raw, sig = token.rsplit(".", 1)
    forged = raw[:-2] + "xy" + "." + sig
    with pytest.raises(TokenError):
        signer.verify(forged)


def test_expired_token_rejected():
    import time

    from app.core.security import TokenError, TokenSigner

    signer = TokenSigner(secret="a" * 40, ttl_seconds=60)
    # ttl of -1 is already expired at the moment of signing.
    token = signer.sign({"sub": "abc"}, ttl=-1)
    with pytest.raises(TokenError):
        signer.verify(token)
    assert time.time() > 0


def test_short_secret_rejected():
    from app.core.security import TokenSigner

    with pytest.raises(ValueError):
        TokenSigner(secret="tooshort")


def test_email_normalisation():
    from app.core.security import normalise_email

    assert normalise_email("  Rakesh@Gmail.COM ") == "rakesh@gmail.com"
    assert normalise_email("not-an-email") is None
    assert normalise_email(None) is None


# ── registration ───────────────────────────────────────────────────


async def test_doctor_registers(client):
    email = _unp("doc")
    r = await client.post(
        "/api/v1/auth/register/doctor",
        json={
            "email": email,
            "password": "a-strong-password",
            "display_name": "Dr Anita Desai",
            "specialty": "General Medicine",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["account"]["role"] == "doctor"
    assert body["account"]["email"] == email
    assert body["doctor"]["specialty"] == "General Medicine"
    assert body["token"]


async def test_doctor_email_required_and_validated(client):
    r = await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": "not-an-email", "password": "a-strong-password",
              "display_name": "X"},
    )
    assert r.status_code == 422


async def test_duplicate_email_rejected(client):
    email = _unp("dup")
    payload = {"email": email, "password": "a-strong-password",
               "display_name": "First"}
    assert (await client.post("/api/v1/auth/register/doctor", json=payload)).status_code == 201
    payload["display_name"] = "Second"
    r = await client.post("/api/v1/auth/register/doctor", json=payload)
    assert r.status_code == 409


async def test_short_password_rejected(client):
    r = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": _unp("short"), "password": "abc", "display_name": "X"},
    )
    assert r.status_code == 422


async def test_login_succeeds(client):
    email = _unp("login")
    await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": email, "password": "a-strong-password", "display_name": "Doc"},
    )
    r = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "a-strong-password"}
    )
    assert r.status_code == 200
    assert r.json()["account"]["email"] == email


async def test_login_email_is_case_insensitive(client):
    email = _unp("case")
    await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": email, "password": "a-strong-password", "display_name": "Doc"},
    )
    r = await client.post(
        "/api/v1/auth/login",
        json={"email": email.upper(), "password": "a-strong-password"},
    )
    assert r.status_code == 200, "email must normalise before comparison"


async def test_login_wrong_password_401(client):
    email = _unp("wrongpw")
    await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": email, "password": "a-strong-password", "display_name": "Doc"},
    )
    r = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "nope-nope-nope"}
    )
    assert r.status_code == 401


async def test_login_unknown_email_same_message(client):
    """Must not reveal whether an address is registered."""
    r = await client.post(
        "/api/v1/auth/login",
        json={"email": _unp("ghost"), "password": "whatever123"},
    )
    assert r.status_code == 401


async def test_me_requires_token(client):
    assert (await client.get("/api/v1/auth/me")).status_code == 401


async def test_me_with_bad_token(client):
    r = await client.get(
        "/api/v1/auth/me", headers={"Authorization": "Bearer nonsense.sig"}
    )
    assert r.status_code == 401


# ── THE linking requirement ────────────────────────────────────────


async def test_patient_inherits_doctor_created_record(client):
    """The core product requirement, end to end over HTTP."""
    email = _unp("inherit")
    token = (await client.get("/api/v1/auth/register/doctor" and "/health")).status_code
    assert token == 200  # sanity: api is up

    # A doctor registers.
    doc = await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": _unp("d"), "password": "a-strong-password",
              "display_name": "Dr Who"},
    )
    assert doc.status_code == 201
    d = doc.json()
    dheaders = {"Authorization": f"Bearer {d['token']}"}

    # The doctor sees a walk-in and records their email.
    p = await client.post(
        "/api/v1/patients",
        headers=dheaders,
        json={"display_name": "Sunita Verma", "email": email, "age": 52},
    )
    assert p.status_code == 201, p.text
    patient_id = p.json()["id"]
    assert p.json().get("user_id") is None, "a walk-in has no account yet"

    # Months later the patient registers with the same address.
    reg = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": "a-strong-password",
              "display_name": "Sunita Verma"},
    )
    assert reg.status_code == 201, reg.text
    body = reg.json()
    assert body["linked_records"] == 1, "the record should be claimed"
    assert body["patients_visible"] >= 1

    # And the row is now attached.
    async with get_session_factory()() as session:
        row = await session.get(Patient, uuid.UUID(patient_id))
        assert row.user_id == uuid.UUID(body["account"]["id"])
        assert row.identity_verified is True


async def test_linking_is_case_insensitive(client):
    email = _unp("case2")
    doc = await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": _unp("d2"), "password": "a-strong-password",
              "display_name": "Dr Case"},
    )
    dheaders = {"Authorization": f"Bearer {doc.json()['token']}"}
    await client.post(
        "/api/v1/patients",
        headers=dheaders,
        json={"display_name": "Mixed Case", "email": email.upper()},
    )
    reg = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email.lower(), "password": "a-strong-password",
              "display_name": "Mixed Case"},
    )
    assert reg.json()["linked_records"] == 1, "case must not break linking"


async def test_linking_is_idempotent(client):
    email = _unp("idem")
    doc = await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": _unp("d3"), "password": "a-strong-password",
              "display_name": "Dr Idem"},
    )
    dheaders = {"Authorization": f"Bearer {doc.json()['token']}"}
    await client.post(
        "/api/v1/patients", headers=dheaders,
        json={"display_name": "Idem", "email": email},
    )
    first = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": "a-strong-password", "display_name": "Idem"},
    )
    assert first.json()["linked_records"] == 1

    # Logging in again must not double-count.
    login = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "a-strong-password"}
    )
    assert login.status_code == 200
    assert login.json()["patients_visible"] == 1


async def test_records_across_two_doctors_share_one_identity(client):
    """The same person, seen by two doctors, is one account."""
    email = _unp("twodoc")
    ids = []
    for n in ("A", "B"):
        d = await client.post(
            "/api/v1/auth/register/doctor",
            json={"email": _unp(f"doc{n}"), "password": "a-strong-password",
                  "display_name": f"Dr {n}"},
        )
        dh = {"Authorization": f"Bearer {d.json()['token']}"}
        p = await client.post(
            "/api/v1/patients", headers=dh,
            json={"display_name": "Two Doctors", "email": email},
        )
        ids.append(p.json()["id"])

    reg = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": "a-strong-password",
              "display_name": "Two Doctors"},
    )
    assert reg.json()["linked_records"] == 2, "both records should be claimed"

    async with get_session_factory()() as session:
        rows = (
            await session.execute(
                select(Patient).where(
                    Patient.user_id == uuid.UUID(reg.json()["account"]["id"])
                )
            )
        ).scalars().all()
        assert len(rows) == 2
        # Each row is still owned by a different doctor: records are
        # isolated per doctor even though the identity is shared.
        assert rows[0].doctor_id != rows[1].doctor_id


async def test_patient_without_email_does_not_link(client):
    """A walk-in with no email simply never links. Not an error."""
    d = await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": _unp("d4"), "password": "a-strong-password",
              "display_name": "Dr NoEmail"},
    )
    dh = {"Authorization": f"Bearer {d.json()['token']}"}
    p = await client.post(
        "/api/v1/patients", headers=dh,
        json={"display_name": "Anonymous Walkin", "email": None},
    )
    assert p.status_code == 201
    assert p.json().get("user_id") is None


async def test_link_does_not_steal_another_accounts_record(client):
    """Two accounts cannot both claim the same address.

    A shared family email is a real scenario. The second registration
    must not silently take the first account's records.
    """
    email = _unp("shared")
    d = await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": _unp("d5"), "password": "a-strong-password",
              "display_name": "Dr Shared"},
    )
    dh = {"Authorization": f"Bearer {d.json()['token']}"}
    p = await client.post(
        "/api/v1/patients", headers=dh,
        json={"display_name": "First User", "email": email},
    )
    patient_id = p.json()["id"]

    first = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": "a-strong-password",
              "display_name": "First User"},
    )
    assert first.json()["linked_records"] == 1

    # A different account registering with the same address is blocked at
    # the account level, because email is unique.
    second = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": "a-strong-password",
              "display_name": "Second User"},
    )
    assert second.status_code == 409

    async with get_session_factory()() as session:
        row = await session.get(Patient, uuid.UUID(patient_id))
        assert row.user_id == uuid.UUID(first.json()["account"]["id"]), \
            "the record must stay with the first account"


async def test_link_log_records_every_link(client):
    email = _unp("logged")
    d = await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": _unp("d6"), "password": "a-strong-password",
              "display_name": "Dr Log"},
    )
    dh = {"Authorization": f"Bearer {d.json()['token']}"}
    await client.post(
        "/api/v1/patients", headers=dh,
        json={"display_name": "Logged", "email": email},
    )
    reg = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": "a-strong-password",
              "display_name": "Logged"},
    )
    async with get_session_factory()() as session:
        logs = (
            await session.execute(
                select(IdentityLinkLog).where(
                    IdentityLinkLog.account_id
                    == uuid.UUID(reg.json()["account"]["id"])
                )
            )
        ).scalars().all()
        assert len(logs) == 1
        assert logs[0].method == "email_exact"
        assert logs[0].confidence == 1.0


async def test_unlink_reverses_a_link(client):
    email = _unp("unlink")
    d = await client.post(
        "/api/v1/auth/register/doctor",
        json={"email": _unp("d7"), "password": "a-strong-password",
              "display_name": "Dr Unlink"},
    )
    dh = {"Authorization": f"Bearer {d.json()['token']}"}
    p = await client.post(
        "/api/v1/patients", headers=dh,
        json={"display_name": "Unlink Me", "email": email},
    )
    patient_id = uuid.UUID(p.json()["id"])
    await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": "a-strong-password",
              "display_name": "Unlink Me"},
    )

    async with get_session_factory()() as session:
        linker = IdentityLinker(session)
        assert await linker.unlink(patient_id) is True
        await session.commit()
        row = await session.get(Patient, patient_id)
        assert row.user_id is None
        assert row.identity_verified is False
