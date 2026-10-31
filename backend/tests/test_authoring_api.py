"""
Tests for the authoring paths.

These endpoints had never been exercised before the UI was built, and a
real bug was hiding in them: a new event carried an empty-string id into a
UUID column, so every write returned 500. The unit tests covered the pure
logic but never the conversion into a database row.

These run against a live Postgres and create a real patient, because the
bug lived in the SQL layer, not in Python.

A note on the Authorization headers, which nearly every test now carries.
They were absent until 2026-10-04, and they were not an oversight in the
tests so much as a record of the code. `get_patient` resolved a patient id
with no ownership check and no authentication, so these requests
succeeded with no token at all - including the write. A stranger could
append records to a patient's chart, and could supersede an existing
record. Fixing that (see test_doctor_isolation.py) meant these tests had
to start authenticating, which is the correct direction for them to move.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.session import init_models
from app.main import app

os.environ.setdefault("POSTGRES_HOST", "localhost")

NOW = datetime.now(timezone.utc)


async def _init_db():
    await init_models()


@pytest.fixture
async def client():
    """A client bound to this test's event loop.

    The engine is a module-level singleton, so each test needs its own
    loop and the engine disposed afterwards - otherwise asyncpg holds a
    connection to a closed loop and the next test fails on teardown.
    """
    from app.db.session import get_engine

    await _init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    await get_engine().dispose()


@pytest.fixture
async def doctor_token(client):
    """A real doctor token.

    These tests used to create patients with no Authorization header at
    all, which only worked because `POST /patients` did not require one.
    They now authenticate, because the endpoint does.
    """
    r = await client.post(
        "/api/v1/auth/register/doctor",
        json={
            "email": f"authoring.{uuid.uuid4().hex[:10]}@solum.test",
            "password": "correct-horse",
            "display_name": "Dr Authoring Test",
            "clinic_name": "Authoring Clinic",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def patient_id(client, doctor_token):
    r = await client.post(
        "/api/v1/patients",
        headers=_auth(doctor_token),
        json={"display_name": "Authoring Test", "age": 61, "phone": "9000000000"},
    )
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    yield pid
    await client.delete(f"/api/v1/patients/{pid}", headers=_auth(doctor_token))


# ── patient creation ────────────────────────────────────────────


async def test_create_patient(client, doctor_token):
    r = await client.post(
        "/api/v1/patients",
        headers=_auth(doctor_token),
        json={"display_name": "New Person"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["display_name"] == "New Person"
    assert uuid.UUID(body["id"])
    await client.delete(f"/api/v1/patients/{body['id']}", headers=_auth(doctor_token))


async def test_create_patient_requires_name(client, doctor_token):
    r = await client.post(
        "/api/v1/patients", headers=_auth(doctor_token), json={"display_name": ""}
    )
    assert r.status_code == 422


async def test_missing_patient_returns_404(client, doctor_token):
    r = await client.get(
        f"/api/v1/patients/{uuid.uuid4()}", headers=_auth(doctor_token)
    )
    assert r.status_code == 404


async def test_every_patient_route_requires_a_token(client, patient_id):
    """The guard that was missing until 2026-10-04.

    Before the ownership fix, each of these answered 200 or 201 with no
    Authorization header whatsoever. The write is the serious one: an
    unauthenticated caller could append to a chart, or supersede a record,
    on a patient they had never seen.
    """
    write = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        json={
            "event_type": "lab",
            "valid_time": NOW.isoformat(),
            "payload": {"test": "Unauthenticated", "value": 1.0},
        },
    )
    assert write.status_code == 401

    for path in ("timeline", "events", "stats"):
        r = await client.get(f"/api/v1/patients/{patient_id}/{path}")
        assert r.status_code == 401, f"{path} was reachable without a token"

    r = await client.get(f"/api/v1/patients/{patient_id}")
    assert r.status_code == 401

    r = await client.get(f"/api/v1/patients/{patient_id}/as-of/{NOW.isoformat()}")
    assert r.status_code == 401

    r = await client.post(
        f"/api/v1/patients/{patient_id}/chat",
        json={"patient_id": patient_id, "query": "what is the potassium?"},
    )
    assert r.status_code == 401

    # And nothing was written by any of it.
    events = await client.get(
        f"/api/v1/patients/{patient_id}/events", headers=_auth(doctor_token)
    )
    assert "Unauthenticated" not in events.text


# ── the write path that was broken ─────────────────────────────


async def test_create_event_succeeds(client, patient_id, doctor_token):
    """Regression: this returned 500 because the id was an empty string."""
    r = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=_auth(doctor_token),
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=10)).isoformat(),
            "payload": {"test": "Potassium", "value": 2.1, "unit": "mEq/L"},
        },
    )
    assert r.status_code == 201, r.text
    assert uuid.UUID(r.json()["id"]), "the id must be a real uuid"
    assert r.json()["validity"] == "current"


async def test_created_event_appears_in_timeline(client, patient_id, doctor_token):
    await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=_auth(doctor_token),
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=10)).isoformat(),
            "payload": {"test": "Sodium", "value": 140, "unit": "mEq/L"},
        },
    )
    r = await client.get(
        f"/api/v1/patients/{patient_id}/timeline", headers=_auth(doctor_token)
    )
    assert r.status_code == 200
    assert r.json()["stats"]["total"] >= 1


# ── the correction flow: the whole research contribution ───────


async def test_correction_creates_a_pair(client, patient_id, doctor_token):
    h = _auth(doctor_token)
    r1 = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=30)).isoformat(),
            "payload": {"test": "Potassium", "value": 2.1, "unit": "mEq/L"},
        },
    )
    assert r1.status_code == 201, r1.text
    original = r1.json()["id"]

    r2 = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=27)).isoformat(),
            "payload": {
                "test": "Potassium", "value": 4.8, "unit": "mEq/L",
                "amendment": "Specimen re-run",
            },
            "source_type": "correction",
            "supersedes_id": original,
        },
    )
    assert r2.status_code == 201, r2.text
    assert r2.json()["validity"] == "current"

    tl = (
        await client.get(f"/api/v1/patients/{patient_id}/timeline", headers=h)
    ).json()
    assert len(tl["pairs"]) == 1
    assert tl["pairs"][0]["superseded"]["validity"] == "superseded"
    assert tl["pairs"][0]["correction"]["validity"] == "current"


async def test_superseded_event_is_marked(client, patient_id, doctor_token):
    h = _auth(doctor_token)
    r1 = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=30)).isoformat(),
            "payload": {"test": "Potassium", "value": 2.1},
        },
    )
    original = r1.json()["id"]
    await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=27)).isoformat(),
            "payload": {"test": "Potassium", "value": 4.8},
            "source_type": "correction",
            "supersedes_id": original,
        },
    )
    events = (
        await client.get(f"/api/v1/patients/{patient_id}/events", headers=h)
    ).json()
    stale = next(e for e in events if e["id"] == original)
    assert stale["validity"] == "superseded"
    assert stale["is_superseded"] is True


async def test_correction_requires_a_target(client, patient_id, doctor_token):
    r = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=_auth(doctor_token),
        json={
            "event_type": "lab",
            "valid_time": NOW.isoformat(),
            "payload": {"test": "Potassium", "value": 4.8},
            "source_type": "correction",
        },
    )
    assert r.status_code == 422


async def test_double_supersession_rejected(client, patient_id, doctor_token):
    """A record can only be superseded once; correct the correction."""
    h = _auth(doctor_token)
    r1 = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=30)).isoformat(),
            "payload": {"test": "Potassium", "value": 2.1},
        },
    )
    target = r1.json()["id"]
    await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=27)).isoformat(),
            "payload": {"test": "Potassium", "value": 4.8},
            "source_type": "correction",
            "supersedes_id": target,
        },
    )
    r3 = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": NOW.isoformat(),
            "payload": {"test": "Potassium", "value": 5.0},
            "source_type": "correction",
            "supersedes_id": target,
        },
    )
    assert r3.status_code == 409, "superseding an already-superseded record"


async def test_supersede_unknown_target_rejected(client, patient_id, doctor_token):
    r = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=_auth(doctor_token),
        json={
            "event_type": "lab",
            "valid_time": NOW.isoformat(),
            "payload": {"test": "Potassium", "value": 4.8},
            "source_type": "correction",
            "supersedes_id": str(uuid.uuid4()),
        },
    )
    assert r.status_code == 404


async def test_cannot_supersede_another_patients_event(client, patient_id, doctor_token):
    """Same doctor, different patient row. The boundary must hold.

    Sharing a practice is not a licence to reach across patients inside
    it: a supersession rewrites what is historically true about one person
    using a value measured on another.
    """
    h = _auth(doctor_token)
    other = await client.post(
        "/api/v1/patients", headers=h, json={"display_name": "Other"},
    )
    other_id = other.json()["id"]
    theirs = await client.post(
        f"/api/v1/patients/{other_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": NOW.isoformat(),
            "payload": {"test": "Potassium", "value": 3.0},
        },
    )
    r = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": NOW.isoformat(),
            "payload": {"test": "Potassium", "value": 9.9},
            "source_type": "correction",
            "supersedes_id": theirs.json()["id"],
        },
    )
    assert r.status_code == 403, "cross-patient supersession must be refused"
    await client.delete(f"/api/v1/patients/{other_id}", headers=h)


# ── as-of resolution through the API ───────────────────────────


async def test_as_of_returns_the_historical_value(client, patient_id, doctor_token):
    """The behaviour the whole project exists to get right."""
    h = _auth(doctor_token)
    r1 = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=30)).isoformat(),
            "payload": {"test": "Potassium", "value": 2.1},
        },
    )
    await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=27)).isoformat(),
            "payload": {"test": "Potassium", "value": 4.8},
            "source_type": "correction",
            "supersedes_id": r1.json()["id"],
        },
    )

    mid = (NOW - timedelta(days=29)).isoformat()
    past = (
        await client.get(f"/api/v1/patients/{patient_id}/as-of/{mid}", headers=h)
    ).json()
    values = [
        e["payload"].get("value")
        for e in past["events"]
        if e["payload"].get("test") == "Potassium"
    ]
    assert values == [2.1], "the value that was true then"

    present = (
        await client.get(
            f"/api/v1/patients/{patient_id}/as-of/{NOW.isoformat()}", headers=h
        )
    ).json()
    values_now = [
        e["payload"].get("value")
        for e in present["events"]
        if e["payload"].get("test") == "Potassium"
    ]
    assert values_now == [4.8], "the corrected value now"


# ── chat refuses the stale value ───────────────────────────────


async def test_chat_does_not_assert_the_superseded_value(client, patient_id, doctor_token):
    h = _auth(doctor_token)
    r1 = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=30)).isoformat(),
            "payload": {"test": "Potassium", "value": 2.1, "unit": "mEq/L"},
        },
    )
    await client.post(
        f"/api/v1/patients/{patient_id}/events",
        headers=h,
        json={
            "event_type": "lab",
            "valid_time": (NOW - timedelta(days=27)).isoformat(),
            "payload": {"test": "Potassium", "value": 4.8, "unit": "mEq/L"},
            "source_type": "correction",
            "supersedes_id": r1.json()["id"],
        },
    )
    r = await client.post(
        f"/api/v1/patients/{patient_id}/chat",
        headers=h,
        json={"patient_id": patient_id, "query": "current potassium level?"},
    )
    assert r.status_code == 200
    assert "2.1" not in r.json()["answer"]
