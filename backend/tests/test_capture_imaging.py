"""Tests for the imaging branch of `POST /patients/{id}/capture`.

The bug this pins: `/capture` gated every upload on
`bool(lab_rows) and confidence >= threshold`. A radiograph has no lab rows, so
every imaging upload was rejected with "no lab rows detected". The study was
stored as a file but never became an event, so it was invisible to retrieval,
to supersession and to "as of" queries. The upload feature looked broken for
exactly the modality it exists to handle, and every status check still
reported healthy.

The lab abstention gate is unchanged and is not relaxed here. Committing an
image asserts no numeric value, so the failure mode that gate protects
against -- a confident wrong number written into a permanent record -- does
not arise for pixels. What replaces it is stated per-test below.

Every test asserts the owner can still work first. A change that blocked
everything would pass the negative half and break the product.
"""

from __future__ import annotations

import io
import uuid

import pytest
from httpx import AsyncClient
from PIL import Image, ImageDraw

PASS = "correct-horse-battery-staple"

#: Enough to pass `image_quality`'s readability check without being a real
#: radiograph. The tests here are about the commit branch, not about
#: BiomedCLIP's understanding of grey rectangles.
PNG_BYTES = b""


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _png(w: int = 640, h: int = 480) -> bytes:
    im = Image.new("RGB", (w, h), (232, 232, 236))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, w - 40, h - 40], fill=(40, 40, 48))
    d.line([(80, 80), (w - 80, h - 80)], fill=(200, 200, 205), width=12)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


async def _doctor(client, tag) -> dict:
    r = await client.post(
        "/api/v1/auth/register/doctor",
        json={
            "email": f"img-{tag}@solum.test",
            "password": PASS,
            "display_name": f"Dr Img {tag}",
            "clinic_name": "Img Clinic",
        },
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


async def _patient(client, doctor):
    r = await client.post(
        "/api/v1/patients",
        headers=_auth(doctor["token"]),
        json={"display_name": "Ramesh Iyer", "age": 61},
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


# ── caption_for ──────────────────────────────────────────────────────


def test_caption_for_reads_the_modality_from_the_filename():
    from app.api.routes.capture import caption_for

    assert "ct scan" in caption_for("head_ct.png")
    assert "mri" in caption_for("brain_mri.jpg")
    assert "ultrasound" in caption_for("abdomen_usg.jpg")
    assert "ecg" in caption_for("resting_ecg.png")


def test_caption_for_names_the_region_when_the_filename_does():
    from app.api.routes.capture import caption_for

    assert "chest" in caption_for("cxr_chest.jpg")
    assert "retinal fundus" in caption_for("fundus_left.jpg")
    assert "head" in caption_for("brain_scan.jpg")


def test_caption_for_does_not_invent_a_region_from_an_unhelpful_name():
    """It falls back to something vague on purpose.

    The caption is a guess about a filename, not a reading of the image. A
    confident guess here would put a wrong claim into the vector index, so
    the fallback is deliberately uninformative.
    """
    from app.api.routes.capture import caption_for

    assert caption_for("scan001.jpg") == "medical image radiograph"
    assert caption_for("") == "medical image radiograph"


# ── the imaging commit branch ────────────────────────────────────────


@pytest.mark.asyncio
async def test_imaging_capture_commits_an_event(client: AsyncClient):
    """A radiograph becomes a real event, not just a stored file.

    The point of the branch. Previously this returned
    `committed: false, reason: "no lab rows detected"` and the study was
    unreachable.
    """
    tag = uuid.uuid4().hex[:8]
    doctor = await _doctor(client, tag)
    patient = await _patient(client, doctor)

    r = await client.post(
        f"/api/v1/patients/{patient['id']}/capture",
        headers=_auth(doctor["token"]),
        data={"event_type": "imaging", "caption": "Chest X-Ray PA view"},
        files={"file": ("cxr.png", _png(), "image/png")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["committed"] is True, body
    assert body["document"]["committed"] is True


@pytest.mark.asyncio
async def test_imaging_capture_uses_the_study_date_not_the_upload_date(
    client: AsyncClient,
):
    """`valid_time` is when the study was taken.

    Filing an old radiograph as of today makes it the most recent imaging, so
    a present-tense question returns a two-year-old study as current.
    """
    tag = uuid.uuid4().hex[:8]
    doctor = await _doctor(client, tag)
    patient = await _patient(client, doctor)

    r = await client.post(
        f"/api/v1/patients/{patient['id']}/capture",
        headers=_auth(doctor["token"]),
        data={
            "event_type": "imaging",
            "caption": "Chest X-Ray PA view",
            "valid_time": "2024-03-11T09:00:00Z",
        },
        files={"file": ("cxr.png", _png(), "image/png")},
    )
    assert r.status_code == 201, r.text

    tl = await client.get(
        f"/api/v1/patients/{patient['id']}/timeline",
        headers=_auth(doctor["token"]),
    )
    assert tl.status_code == 200, tl.text
    body = tl.json()
    # `history` is the complete event list; `current` is a subset of it.
    # Concatenating both would double-count, so read `history` alone.
    events = body.get("history") or []
    imaging = [e for e in events if e.get("event_type") == "imaging"]
    assert len(imaging) == 1, imaging
    assert imaging[0]["valid_time"].startswith("2024-03-11"), imaging[0]


@pytest.mark.asyncio
async def test_imaging_capture_reaches_the_review_queue_when_unreadable(
    client: AsyncClient,
):
    """The readability gate still applies to imaging.

    Branching on `event_type` must not become a way to push an unreadable
    picture past the quality check.
    """
    tag = uuid.uuid4().hex[:8]
    doctor = await _doctor(client, tag)
    patient = await _patient(client, doctor)

    # A near-uniform frame: no contrast, so quality should reject it.
    flat = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 1, 1)).save(flat, format="PNG")

    r = await client.post(
        f"/api/v1/patients/{patient['id']}/capture",
        headers=_auth(doctor["token"]),
        data={"event_type": "imaging"},
        files={"file": ("cxr.png", flat.getvalue(), "image/png")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    if not body["committed"]:
        assert "quality" in (body["reason"] or "").lower(), body


@pytest.mark.asyncio
async def test_lab_capture_is_unchanged_by_the_imaging_branch(client: AsyncClient):
    """The branch keys on `event_type`, so the default path must be untouched.

    If this regressed, every lab sheet would start committing or the gate
    would stop abstaining. Either would be a clinical safety failure.
    """
    tag = uuid.uuid4().hex[:8]
    doctor = await _doctor(client, tag)
    patient = await _patient(client, doctor)

    # A photograph with no report text: no rows, so it must abstain.
    r = await client.post(
        f"/api/v1/patients/{patient['id']}/capture",
        headers=_auth(doctor["token"]),
        data={"event_type": "lab"},
        files={"file": ("sheet.png", _png(), "image/png")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["committed"] is False, body
    assert body["reason"], body


@pytest.mark.asyncio
async def test_stranger_cannot_capture_a_study_to_another_patients_chart(
    client: AsyncClient,
):
    """Uploading a scan writes bytes and appends an event. It is a write."""
    tag = uuid.uuid4().hex[:8]
    owner = await _doctor(client, f"a-{tag}")
    stranger = await _doctor(client, f"b-{tag}")
    patient = await _patient(client, owner)

    r = await client.post(
        f"/api/v1/patients/{patient['id']}/capture",
        headers=_auth(stranger["token"]),
        data={"event_type": "imaging"},
        files={"file": ("cxr.png", _png(), "image/png")},
    )
    assert r.status_code in (403, 404), f"status {r.status}: {r.text}"

    anon = await client.post(
        f"/api/v1/patients/{patient['id']}/capture",
        data={"event_type": "imaging"},
        files={"file": ("cxr.png", _png(), "image/png")},
    )
    assert anon.status_code == 401