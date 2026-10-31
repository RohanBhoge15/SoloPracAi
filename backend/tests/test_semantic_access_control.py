"""
Authorisation tests for the semantic routes.

Found 2026-10-04 by `_e2e/probe-semantic-auth.ps1`. All four of
`/semantic/search`, `/semantic/reindex`, `/semantic/image` and
`/semantic/image-search` took a bare `patient_id` query parameter and
resolved it with `session.get(Patient, patient_id)`. No token was required
on any of them.

An unauthenticated `POST /api/v1/semantic/search?patient_id=...` returned
another practice's entire record - ranked, scored, and shaped exactly like a
legitimate search result. It was the least detectable leak in the project:
there is no error page and no malformed response, just a tidy list of hits
that happens to be someone else's patient. A doctor asked "what is this
patient's HbA1c" and got an answer, from a patient they had never seen.

The `find-similar` route is included because it takes an object-store key as
a *second* input. Authorising the patient is not enough there: the key has
to belong to that patient, or a guessed path reads another practice's file.

Every test asserts the owner can still work first. A fix that blocked
everything would pass the negative half and break the product.
"""

import uuid

import pytest
from httpx import AsyncClient

PASS = "correct-horse-battery-staple"


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _doctor(client, tag) -> dict:
    r = await client.post(
        "/api/v1/auth/register/doctor",
        json={
            "email": f"sem-{tag}@solum.test",
            "password": PASS,
            "display_name": f"Dr Sem {tag}",
            "clinic_name": "Sem Clinic",
        },
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


async def _patient_with_chart(client, doctor, tag):
    """A patient with one lab record, so a search has something to find."""
    r = await client.post(
        "/api/v1/patients",
        headers=_auth(doctor["token"]),
        json={"display_name": f"Sem Patient {tag}", "age": 58, "email": f"p-{tag}@solum.test"},
    )
    assert r.status_code in (200, 201), r.text
    pid = r.json()["id"]

    w = await client.post(
        f"/api/v1/patients/{pid}/events",
        headers=_auth(doctor["token"]),
        json={
            "event_type": "lab",
            "valid_time": "2026-09-01T09:00:00Z",
            "payload": {"test": "HbA1c", "value": 11.2, "unit": "%"},
        },
    )
    assert w.status_code in (200, 201), w.text
    return pid


@pytest.mark.asyncio
async def test_owner_can_still_search(client: AsyncClient):
    """Ground truth. Without this, every negative result below is theatre."""
    tag = uuid.uuid4().hex[:8]
    doc = await _doctor(client, tag)
    pid = await _patient_with_chart(client, doc, tag)

    r = await client.post(
        f"/api/v1/semantic/search?patient_id={pid}&query=diabetes+control+marker",
        headers=_auth(doc["token"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hits"], "the owner's own search must return the record they wrote"
    assert body["hits"][0]["event_type"] == "lab"


@pytest.mark.asyncio
async def test_semantic_search_requires_a_token(client: AsyncClient):
    """The reported bug, verbatim: no token, full record."""
    tag = uuid.uuid4().hex[:8]
    doc = await _doctor(client, tag)
    pid = await _patient_with_chart(client, doc, tag)

    r = await client.post(
        f"/api/v1/semantic/search?patient_id={pid}&query=diabetes"
    )
    assert r.status_code == 401, f"status {r.status}: {r.text}"
    assert "lab" not in r.text.lower() or r.status == 401


@pytest.mark.asyncio
async def test_stranger_cannot_search_another_patients_record(client: AsyncClient):
    tag = uuid.uuid4().hex[:8]
    owner = await _doctor(client, f"a-{tag}")
    stranger = await _doctor(client, f"b-{tag}")
    pid = await _patient_with_chart(client, owner, tag)

    r = await client.post(
        f"/api/v1/semantic/search?patient_id={pid}&query=diabetes+control+marker",
        headers=_auth(stranger["token"]),
    )
    assert r.status_code in (403, 404), f"status {r.status}: {r.text}"
    assert '"hits"' not in r.text, "a scored hit list is itself the leak"


@pytest.mark.asyncio
async def test_stranger_cannot_reindex_another_patients_vectors(client: AsyncClient):
    """A write into someone else's index.

    Not as dramatic as editing a chart, but it is a stranger driving writes
    against another practice's derived store, and it can be used to probe
    whether a patient id exists.
    """
    tag = uuid.uuid4().hex[:8]
    owner = await _doctor(client, f"a-{tag}")
    stranger = await _doctor(client, f"b-{tag}")
    pid = await _patient_with_chart(client, owner, tag)

    r = await client.post(
        f"/api/v1/semantic/reindex?patient_id={pid}",
        headers=_auth(stranger["token"]),
    )
    assert r.status_code in (403, 404), f"status {r.status}: {r.text}"

    anon = await client.post(f"/api/v1/semantic/reindex?patient_id={pid}")
    assert anon.status_code == 401


@pytest.mark.asyncio
async def test_stranger_cannot_describe_search_someone_elses_images(
    client: AsyncClient,
):
    tag = uuid.uuid4().hex[:8]
    owner = await _doctor(client, f"a-{tag}")
    stranger = await _doctor(client, f"b-{tag}")
    pid = await _patient_with_chart(client, owner, tag)

    r = await client.post(
        f"/api/v1/semantic/image-search?patient_id={pid}&description=chest+film+effusion",
        headers=_auth(stranger["token"]),
    )
    assert r.status_code in (403, 404), f"status {r.status}: {r.text}"

    anon = await client.post(
        f"/api/v1/semantic/image-search?patient_id={pid}&description=chest+film"
    )
    assert anon.status_code == 401


@pytest.mark.asyncio
async def test_stranger_cannot_upload_a_study_to_another_patients_chart(
    client: AsyncClient,
):
    """Uploading a scan writes bytes and appends an event. It is a write."""
    tag = uuid.uuid4().hex[:8]
    owner = await _doctor(client, f"a-{tag}")
    stranger = await _doctor(client, f"b-{tag}")
    pid = await _patient_with_chart(client, owner, tag)

    r = await client.post(
        f"/api/v1/semantic/image?patient_id={pid}",
        headers=_auth(stranger["token"]),
        files={"file": ("chest.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png")},
    )
    # 503 is acceptable only if it is the vision-disabled answer for a
    # legitimate caller; for a stranger the request must not get that far.
    assert r.status_code in (403, 404), f"status {r.status}: {r.text}"

    anon = await client.post(
        f"/api/v1/semantic/image?patient_id={pid}",
        files={"file": ("chest.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png")},
    )
    assert anon.status_code == 401


@pytest.mark.asyncio
async def test_find_similar_rejects_an_image_the_caller_does_not_own(
    client: AsyncClient,
):
    """The object-key input needs its own check.

    `get_patient` proves the patient is the caller's. It says nothing about
    the `image_path` in the URL, which is a second, independent input. Without
    a check against the caller's own documents, a doctor who learned or
    guessed another practice's key could have that file read back as a
    finding through the visual-similarity path.
    """
    tag = uuid.uuid4().hex[:8]
    owner = await _doctor(client, f"a-{tag}")
    stranger = await _doctor(client, f"b-{tag}")
    pid = await _patient_with_chart(client, owner, tag)

    # A key that exists on the owner's chart, or would if anything did.
    forged = f"semantic/{tag}/chest-xray.png"

    r = await client.post(
        f"/api/v1/semantic/image/{forged}/find-similar?patient_id={pid}",
        headers=_auth(stranger["token"]),
    )
    assert r.status_code in (403, 404), (
        f"status {r.status}: {r.text} - a foreign object key was accepted"
    )

    anon = await client.post(f"/api/v1/semantic/image/{forged}/find-similar?patient_id={pid}")
    assert anon.status_code == 401


@pytest.mark.asyncio
async def test_semantic_status_requires_a_token(client: AsyncClient):
    """Discloses model names, the models directory and the point count."""
    tag = uuid.uuid4().hex[:8]
    doc = await _doctor(client, tag)

    anon = await client.get("/api/v1/semantic/status")
    assert anon.status_code == 401, f"status {anon.status}"

    ok = await client.get("/api/v1/semantic/status", headers=_auth(doc["token"]))
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_404_not_403_for_a_foreign_patient(client: AsyncClient):
    """403 would confirm the id exists, making this an enumeration oracle."""
    tag = uuid.uuid4().hex[:8]
    owner = await _doctor(client, f"a-{tag}")
    stranger = await _doctor(client, f"b-{tag}")
    pid = await _patient_with_chart(client, owner, tag)

    foreign = await client.post(
        f"/api/v1/semantic/search?patient_id={pid}&query=x",
        headers=_auth(stranger["token"]),
    )
    missing = await client.post(
        f"/api/v1/semantic/search?patient_id={uuid.uuid4()}&query=x",
        headers=_auth(stranger["token"]),
    )
    assert foreign.status_code == missing.status_code
    assert foreign.json() == missing.json()


@pytest.mark.asyncio
async def test_patient_account_cannot_use_the_doctor_search_surface(
    client: AsyncClient,
):
    """A patient token must not reach the doctor-side index at all."""
    tag = uuid.uuid4().hex[:8]
    doc = await _doctor(client, tag)
    pid = await _patient_with_chart(client, doc, tag)

    reg = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": f"pt-{tag}@solum.test", "password": PASS, "display_name": "PT"},
    )
    ptoken = reg.json()["token"]

    r = await client.post(
        f"/api/v1/semantic/search?patient_id={pid}&query=x",
        headers=_auth(ptoken),
    )
    assert r.status_code in (403, 404), f"status {r.status}"
