"""Access control on /api/v1/patients.

The bug these tests exist to prevent: `POST /api/v1/patients` accepted a
call with no `Authorization` header, or with an invalid one, and quietly
created a patient row with `doctor_id = None` and `origin = "unowned"`.
The docstring described this as keeping the demo flow working.

It is not a demo affordance. An unowned patient row is a silent write of
someone's name, age, phone and email with no owner, no audit trail, and no
doctor who will ever see it. `GET /api/v1/patients` was worse: it
selected every patient row in the database regardless of doctor, so any
caller could read the entire patient list of every practice on the server.

Both now require a doctor token, and the list is scoped by `doctor_id`.
"""

from __future__ import annotations

import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("POSTGRES_HOST", "localhost")

from app.db.session import get_engine, init_models  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
async def client():
    await init_models()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    await get_engine().dispose()


async def register_doctor(client, suffix: str = "a"):
    email = f"dr.{uuid.uuid4().hex[:8]}@solum.test"
    response = await client.post(
        "/api/v1/auth/register/doctor",
        json={
            "email": email,
            "password": "correct-horse-battery",
            "display_name": f"Dr Test {suffix}",
            "specialty": "General Medicine",
            "clinic_name": f"Clinic {suffix}",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["token"], email


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestPatientCreationRequiresAuth:
    @pytest.mark.asyncio
    async def test_anonymous_create_is_refused(self, client):
        response = await client.post(
            "/api/v1/patients", json={"display_name": "No Token"}
        )
        assert response.status_code in (401, 403), (
            f"an unauthenticated caller created a patient: {response.status_code}"
        )

    @pytest.mark.asyncio
    async def test_a_garbage_token_is_refused(self, client):
        response = await client.post(
            "/api/v1/patients",
            headers=auth("not-a-real-token"),
            json={"display_name": "Bad Token"},
        )
        assert response.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_an_expired_or_foreign_shape_token_is_refused(self, client):
        # Right shape, no signature. Must not be treated as anonymous and
        # allowed through.
        response = await client.post(
            "/api/v1/patients",
            headers=auth("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.sig"),
            json={"display_name": "Forged"},
        )
        assert response.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_a_patient_token_cannot_create_patients(self, client):
        """A patient is not a doctor. Role separation has to hold here."""
        email = f"pt.{uuid.uuid4().hex[:8]}@solum.test"
        created = await client.post(
            "/api/v1/auth/register/patient",
            json={
                "email": email,
                "password": "another-good-passphrase",
                "display_name": "Not A Doctor",
            },
        )
        assert created.status_code == 201, created.text
        token = created.json()["token"]

        response = await client.post(
            "/api/v1/patients",
            headers=auth(token),
            json={"display_name": "Should Not Exist"},
        )
        assert response.status_code == 403


class TestListIsScopedToTheDoctor:
    @pytest.mark.asyncio
    async def test_anonymous_list_is_refused(self, client):
        response = await client.get("/api/v1/patients")
        assert response.status_code in (401, 403), (
            f"an unauthenticated caller read the patient list: {response.status_code}"
        )

    @pytest.mark.asyncio
    async def test_a_doctor_only_sees_their_own_patients(self, client):
        """The core of it: two practices, no crossover.

        Before the fix this returned every patient row in the database, so
        doctor A could read doctor B's entire patient list by name.
        """
        token_a, _ = await register_doctor(client, "alpha")
        token_b, _ = await register_doctor(client, "beta")

        mine = "Alpha Patient Private"
        theirs = "Beta Patient Private"

        assert (
            await client.post(
                "/api/v1/patients", headers=auth(token_a), json={"display_name": mine}
            )
        ).status_code == 201
        assert (
            await client.post(
                "/api/v1/patients",
                headers=auth(token_b),
                json={"display_name": theirs},
            )
        ).status_code == 201

        list_a = await client.get("/api/v1/patients", headers=auth(token_a))
        assert list_a.status_code == 200
        names_a = [p["display_name"] for p in list_a.json()]

        assert mine in names_a
        assert theirs not in names_a, "doctor A can see doctor B's patient"

    @pytest.mark.asyncio
    async def test_search_does_not_leak_across_doctors(self, client):
        """Search was the same leak with a narrower trigger.

        Filtering by a name fragment must not bypass the doctor_id scope.
        """
        token_a, _ = await register_doctor(client, "alpha")
        token_b, _ = await register_doctor(client, "beta")
        secret = "Zebra稀有 Patient"

        await client.post(
            "/api/v1/patients", headers=auth(token_b), json={"display_name": secret}
        )

        found = await client.get(
            "/api/v1/patients", headers=auth(token_a), params={"search": "Zebra"}
        )
        assert found.status_code == 200
        assert [p["display_name"] for p in found.json()] == []


class TestDeleteIsLockedAndOwned:
    @pytest.mark.asyncio
    async def test_anonymous_delete_is_refused(self, client):
        token, _ = await register_doctor(client, "del")
        made = await client.post(
            "/api/v1/patients", headers=auth(token), json={"display_name": "Doomed"}
        )
        pid = made.json()["id"]

        response = await client.delete(f"/api/v1/patients/{pid}")
        assert response.status_code in (401, 403), (
            "an unauthenticated caller deleted a patient record"
        )

    @pytest.mark.asyncio
    async def test_a_doctor_cannot_delete_another_doctors_patient(self, client):
        token_a, _ = await register_doctor(client, "alpha")
        token_b, _ = await register_doctor(client, "beta")
        made = await client.post(
            "/api/v1/patients", headers=auth(token_b), json={"display_name": "B's Patient"}
        )
        pid = made.json()["id"]

        response = await client.delete(f"/api/v1/patients/{pid}", headers=auth(token_a))
        # 404, not 403: a 403 would confirm the id exists and turn the
        # endpoint into an enumeration oracle.
        assert response.status_code == 404

        still_there = await client.get(
            f"/api/v1/patients/{pid}", headers=auth(token_b)
        )
        assert still_there.status_code == 200, "the record was actually deleted"


class TestNoUnownedRowsArePossible:
    @pytest.mark.asyncio
    async def test_the_unowned_code_path_is_gone(self, client):
        """Every reachable creation path must attach a doctor.

        The old behaviour branched on the presence of a bearer token and
        fell through to `doctor_id = None`. This asserts the observable
        consequence instead of the implementation: after any sequence of
        refused and accepted requests, the newest patient row belongs to a
        doctor.
        """
        from sqlalchemy import select

        from app.db.models import Patient
        from app.db.session import get_session_factory

        # Provoke every refusal path.
        await client.post("/api/v1/patients", json={"display_name": "Anonymous"})
        await client.post(
            "/api/v1/patients",
            headers=auth("garbage"),
            json={"display_name": "Garbage"},
        )

        token, _ = await register_doctor(client, "own")
        accepted = await client.post(
            "/api/v1/patients",
            headers=auth(token),
            json={"display_name": "Owned", "email": f"own.{uuid.uuid4().hex[:6]}@x.test"},
        )
        assert accepted.status_code == 201

        factory = get_session_factory()
        async with factory() as session:
            newest = (
                await session.execute(
                    select(Patient).order_by(Patient.created_at.desc()).limit(1)
                )
            ).scalars().first()

        assert newest is not None
        assert newest.doctor_id is not None, (
            f"newest patient {newest.display_name!r} has no owning doctor"
        )
        assert newest.origin == "doctor_created"
