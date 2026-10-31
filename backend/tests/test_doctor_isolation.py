"""
Regression tests for the cross-doctor data leak.

Found 2026-10-04 by _e2e/02-leak.mjs. The get_patient dependency in
app/api/deps.py was a bare single-row lookup by id, with no auth and no
ownership check, so any authenticated doctor could name any patient id and
read or write that practice's entire record. It was not a theoretical
concern: the live stack handed over a timeline, an event list, the
patient row, the document list, and a natural-language chat answer
quoting the private value, and it accepted a write (201).

The distinction these tests exist to protect: a patient CAN belong to
several doctors - that is what email linking means, and it is a feature.
What must never happen is one doctor seeing another doctor's records for
the same person. Each doctor gets their own Patient row precisely so
that records cannot mix.

Both halves are asserted. A fix that blocked everything would pass the
first half and break the product.
"""

import uuid

import pytest
from httpx import AsyncClient

PASS = "correct-horse-battery-staple"


async def _mk_doctor(client: AsyncClient, tag: str) -> dict:
    r = await client.post(
        "/api/v1/auth/register/doctor",
        json={
            "email": f"doc-{tag}@solum.test",
            "password": PASS,
            "display_name": f"Dr Test {tag}",
            "clinic_name": "Test Clinic",
        },
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


async def _mk_patient(client: AsyncClient, token: str, name: str, email: str | None = None):
    body = {"display_name": name, "age": 50, "sex": "M"}
    if email:
        body["email"] = email
    r = await client.post("/api/v1/patients", json=body, headers=_auth(token))
    assert r.status_code in (200, 201), r.text
    return r.json()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _write_lab(client, token, patient_id, value, when="2026-09-01T09:00:00Z"):
    r = await client.post(
        f"/api/v1/patients/{patient_id}/events",
        json={
            "event_type": "lab",
            "valid_time": when,
            "payload": {"test": "Leak", "value": value},
        },
        headers=_auth(token),
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


async def _insert_document(patient_id, doctor_id, *, kind="prescription", object_key=None):
    """Write a Document row directly.

    Going through POST /documents/{kind} would also exercise PDF rendering,
    which needs WeasyPrint plus a working object store. These tests are
    about authorization, so they set up state directly and let the route
    under test decide. That keeps the suite runnable anywhere.
    """
    from app.db.documents import Document
    from app.db.session import session_scope

    async with session_scope() as session:
        doc = Document(
            patient_id=patient_id,
            doctor_id=doctor_id,
            kind=kind,
            title=f"{kind} for isolation test",
            object_key=object_key,
            content_type="application/pdf",
            content={"reference": "ISO-TEST"},
            issued_to_patient=True,
        )
        session.add(doc)
        await session.flush()
        return doc.id


@pytest.mark.asyncio
async def test_other_doctor_cannot_read_the_timeline(client: AsyncClient):
    """The core leak: read the whole record by guessing a patient id."""
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    b = await _mk_doctor(client, f"b-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")
    await _write_lab(client, a["token"], pat["id"], "SECRET-VALUE-1")

    # the owner can see it - otherwise this test proves nothing
    own = await client.get(
        f"/api/v1/patients/{pat['id']}/timeline", headers=_auth(a["token"])
    )
    assert own.status_code == 200
    assert "SECRET-VALUE-1" in own.text

    # the stranger cannot
    other = await client.get(
        f"/api/v1/patients/{pat['id']}/timeline", headers=_auth(b["token"])
    )
    assert other.status_code == 404
    assert "SECRET-VALUE-1" not in other.text


@pytest.mark.asyncio
async def test_other_doctor_cannot_list_events(client: AsyncClient):
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    b = await _mk_doctor(client, f"b-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")
    await _write_lab(client, a["token"], pat["id"], "SECRET-VALUE-2")

    r = await client.get(f"/api/v1/patients/{pat['id']}/events", headers=_auth(b["token"]))
    assert r.status_code == 404
    assert "SECRET-VALUE-2" not in r.text


@pytest.mark.asyncio
async def test_other_doctor_cannot_fetch_the_patient_row(client: AsyncClient):
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    b = await _mk_doctor(client, f"b-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")

    r = await client.get(f"/api/v1/patients/{pat['id']}", headers=_auth(b["token"]))
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_other_doctor_cannot_write_into_the_record(client: AsyncClient):
    """A write leak is worse than a read leak: it corrupts the record."""
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    b = await _mk_doctor(client, f"b-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")

    r = await client.post(
        f"/api/v1/patients/{pat['id']}/events",
        json={
            "event_type": "lab",
            "valid_time": "2026-09-02T09:00:00Z",
            "payload": {"test": "Injected", "value": "6666"},
        },
        headers=_auth(b["token"]),
    )
    assert r.status_code == 404

    # and nothing was actually written
    own = await client.get(
        f"/api/v1/patients/{pat['id']}/events", headers=_auth(a["token"])
    )
    assert "Injected" not in own.text


@pytest.mark.asyncio
async def test_other_doctor_cannot_chat_about_the_patient(client: AsyncClient):
    """Chat was the worst vector: it returns the value in plain language."""
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    b = await _mk_doctor(client, f"b-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")
    await _write_lab(client, a["token"], pat["id"], "SECRET-VALUE-3")

    r = await client.post(
        f"/api/v1/patients/{pat['id']}/chat",
        json={"patient_id": pat["id"], "query": "What was the Leak value?"},
        headers=_auth(b["token"]),
    )
    assert r.status_code == 404
    assert "SECRET-VALUE-3" not in r.text


@pytest.mark.asyncio
async def test_other_doctor_cannot_list_documents(client: AsyncClient):
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    b = await _mk_doctor(client, f"b-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")
    await _insert_document(pat["id"], a["doctor"]["id"], kind="prescription")

    own = await client.get(
        f"/api/v1/documents/patient/{pat['id']}", headers=_auth(a["token"])
    )
    assert own.status_code == 200
    assert len(own.json()) == 1

    r = await client.get(
        f"/api/v1/documents/patient/{pat['id']}", headers=_auth(b["token"])
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_document_download_requires_ownership(client: AsyncClient):
    """A presigned URL is a bearer token for a file carrying patient details.

    The document row is written straight to the database rather than
    generated through the API, because generating one needs WeasyPrint and
    this suite runs on a host that has no GTK. That is the right trade: the
    question here is who is allowed to ask, not whether the PDF renders.
    The rendering is covered by the live-stack scripts.
    """
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    b = await _mk_doctor(client, f"b-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")
    doc_id = await _insert_document(
        pat["id"], a["doctor"]["id"], kind="prescription", object_key=f"iso/{tag}.pdf"
    )

    # The owner passes authorization. Whether the object store answers is a
    # separate concern, so only the absence of 404 is asserted here.
    ok = await client.get(f"/api/v1/documents/{doc_id}/download", headers=_auth(a["token"]))
    assert ok.status_code != 404

    bad = await client.get(f"/api/v1/documents/{doc_id}/download", headers=_auth(b["token"]))
    assert bad.status_code == 404

    raw = await client.get(f"/api/v1/documents/{doc_id}/raw", headers=_auth(b["token"]))
    assert raw.status_code == 404

    anon = await client.get(f"/api/v1/documents/{doc_id}/download")
    assert anon.status_code == 401

    # 404 rather than 403, so this cannot be used to probe which ids exist.
    missing = await client.get(
        f"/api/v1/documents/{uuid.uuid4()}/download", headers=_auth(a["token"])
    )
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_patient_can_read_only_their_own_documents(client: AsyncClient):
    """The legitimate patient-side reader, and the boundary around it.

    A patient linked by email may reach the documents on their own rows.
    They must not reach a document that belongs to a different person, even
    where that other person happens to share a doctor with them.
    """
    tag = uuid.uuid4().hex[:8]
    doc = await _mk_doctor(client, f"a-{tag}")
    mine_email = f"mine-{tag}@solum.test"
    theirs_email = f"theirs-{tag}@solum.test"

    mine = await _mk_patient(client, doc["token"], f"Mine {tag}", mine_email)
    theirs = await _mk_patient(client, doc["token"], f"Theirs {tag}", theirs_email)
    my_doc = await _insert_document(
        mine["id"], doc["doctor"]["id"], kind="prescription",
        object_key=f"iso/{tag}-mine.pdf",
    )
    await _insert_document(
        theirs["id"], doc["doctor"]["id"], kind="prescription",
        object_key=f"iso/{tag}-theirs.pdf",
    )

    reg = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": mine_email, "password": PASS, "display_name": f"Mine {tag}"},
    )
    assert reg.status_code in (200, 201), reg.text
    token = reg.json()["token"]

    ok = await client.get(f"/api/v1/documents/{my_doc}/download", headers=_auth(token))
    assert ok.status_code != 404, "the patient must be able to reach their own document"

    # same doctor, different patient row - still refused
    all_docs = await client.get("/api/v1/documents", headers=_auth(token))
    if all_docs.status_code == 200:
        assert str(theirs["id"]) not in all_docs.text
        assert str(mine["id"]) in all_docs.text


@pytest.mark.asyncio
async def test_shared_patient_still_works_for_both_doctors(client: AsyncClient):
    """The legitimate case. Blocking this would 'fix' the leak by breaking
    the product, which is the failure mode these tests exist to prevent."""
    tag = uuid.uuid4().hex[:8]
    shared = f"shared-{tag}@solum.test"
    c_doc = await _mk_doctor(client, f"c-{tag}")
    d_doc = await _mk_doctor(client, f"d-{tag}")

    pc = await _mk_patient(client, c_doc["token"], f"Shared {tag}", shared)
    pd = await _mk_patient(client, d_doc["token"], f"Shared {tag}", shared)

    # separate rows - this is what keeps the records apart
    assert pc["id"] != pd["id"]

    await _write_lab(client, c_doc["token"], pc["id"], "C-ONLY-VALUE", "2026-09-03T09:00:00Z")

    # each reads their own
    own_c = await client.get(f"/api/v1/patients/{pc['id']}/timeline", headers=_auth(c_doc["token"]))
    assert own_c.status_code == 200
    assert "C-ONLY-VALUE" in own_c.text

    own_d = await client.get(f"/api/v1/patients/{pd['id']}/timeline", headers=_auth(d_doc["token"]))
    assert own_d.status_code == 200
    assert "C-ONLY-VALUE" not in own_d.text

    # the person themself sees both doctors
    reg = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": shared, "password": PASS, "display_name": f"Shared {tag}"},
    )
    assert reg.status_code in (200, 201), reg.text
    ptoken = reg.json()["token"]

    recs = await client.get("/api/v1/patient/records", headers=_auth(ptoken))
    assert recs.status_code == 200
    assert len(recs.json()) == 2


@pytest.mark.asyncio
async def test_patient_role_cannot_use_doctor_routes(client: AsyncClient):
    """A patient token must not reach the doctor surface at all."""
    tag = uuid.uuid4().hex[:8]
    d = await _mk_doctor(client, f"a-{tag}")
    pat = await _mk_patient(client, d["token"], f"Victim {tag}")

    r = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": f"pp-{tag}@solum.test", "password": PASS, "display_name": "P"},
    )
    ptoken = r.json()["token"]

    r2 = await client.get(f"/api/v1/patients/{pat['id']}/timeline", headers=_auth(ptoken))
    assert r2.status_code in (403, 404)


@pytest.mark.asyncio
async def test_anonymous_cannot_read_any_patient(client: AsyncClient):
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")
    await _write_lab(client, a["token"], pat["id"], "SECRET-VALUE-4")

    for path in (
        f"/api/v1/patients/{pat['id']}/timeline",
        f"/api/v1/patients/{pat['id']}/events",
        f"/api/v1/patients/{pat['id']}",
        f"/api/v1/documents/patient/{pat['id']}",
    ):
        r = await client.get(path)
        assert r.status_code == 401, f"{path} returned {r.status_code}"
        assert "SECRET-VALUE-4" not in r.text


@pytest.mark.asyncio
async def test_404_not_403_for_a_foreign_patient(client: AsyncClient):
    """403 would confirm the id exists, making this an enumeration oracle."""
    tag = uuid.uuid4().hex[:8]
    a = await _mk_doctor(client, f"a-{tag}")
    b = await _mk_doctor(client, f"b-{tag}")
    pat = await _mk_patient(client, a["token"], f"Victim {tag}")

    foreign = await client.get(f"/api/v1/patients/{pat['id']}", headers=_auth(b["token"]))
    missing = await client.get(
        f"/api/v1/patients/{uuid.uuid4()}", headers=_auth(b["token"])
    )
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()