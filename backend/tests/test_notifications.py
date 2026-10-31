"""
Regression tests for the notification path.

Found 2026-10-04 by `_e2e/probe-bell.mjs`. Every notification raised
AttributeError: 'Document' object has no attribute 'reference'. The
reference lives inside the JSON `content` column, not on the row.

It stayed invisible because the route that calls this deliberately wraps
the call in a bare `except Exception`, so that a Redis outage cannot cost
a doctor the prescription they just wrote. That is the right call for
availability, and it hid a total failure of a whole feature: documents
were issued correctly, the inbox was correct, and the patient's bell sat
at zero forever. One WARNING per document in the log, which is exactly
what you stop reading.

These tests assert the notification actually arrives. That is the gap the
blanket catch created, so the test has to reach past the catch.
"""

import uuid

import pytest
from httpx import AsyncClient

PASS = "correct-horse-battery-staple"


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _mk_doctor(client, tag) -> dict:
    r = await client.post(
        "/api/v1/auth/register/doctor",
        json={
            "email": f"notif-doc-{tag}@solum.test",
            "password": PASS,
            "display_name": f"Dr Notif {tag}",
            "clinic_name": "Notif Clinic",
        },
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


async def _mk_patient_row(client, token, name, email) -> dict:
    r = await client.post(
        "/api/v1/patients",
        headers=_auth(token),
        json={"display_name": name, "age": 40, "email": email},
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


@pytest.fixture(autouse=True)
def stub_pdf(monkeypatch):
    """Make the PDF engine look available, and return plausible bytes.

    The route checks `pdf_available()` and returns 503 before it reaches the
    notification code at all. WeasyPrint needs Pango and only exists in the
    API image, so on a bare host every one of these tests would otherwise
    stop at the door - and the whole feature could regress again unnoticed.

    Stubbing the renderer rather than skipping keeps the real route running:
    the document is created, the event is written, the notification is
    published, and all of it is asserted. Only the bytes are fake, and PDF
    rendering is covered properly by the live-stack scripts in `_e2e/`.
    """
    from app.api.routes import documents as documents_route

    monkeypatch.setattr(documents_route, "pdf_available", lambda: True)
    monkeypatch.setattr(
        documents_route, "render_html_to_pdf", lambda html, *a, **k: b"%PDF-1.7 stub"
    )


class FakeBus:
    """An in-memory stand-in for the Redis notification bus.

    The real bus is a Redis stream and `redis` is only installed in the API
    image, not in the bare host venv. Substituting it here keeps these tests
    about the thing that actually broke - building the notification from a
    Document row - instead of about the availability of a message broker.

    It is also a better assertion surface than the real thing. What matters
    is *which* recipient and *what* payload, and a fake records both
    exactly, with no consumer-group semantics in the way.
    """

    def __init__(self) -> None:
        self.published: list[dict] = []

    async def publish(self, *, recipient_account_id, kind, title, body, payload):
        entry = {
            "recipient": str(recipient_account_id),
            "kind": kind,
            "title": title,
            "body": body,
            "payload": payload,
        }
        self.published.append(entry)
        return f"stub-{len(self.published)}"

    def for_recipient(self, recipient: str) -> list[dict]:
        return [n for n in self.published if n["recipient"] == str(recipient)]


@pytest.fixture
def bus(monkeypatch) -> FakeBus:
    fake = FakeBus()
    monkeypatch.setattr("app.services.notifications.get_bus", lambda: fake)
    return fake


async def _patient_account(client, email, name) -> dict:
    r = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": PASS, "display_name": name},
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


async def _issue(client, token, patient_id, *, send=True) -> dict:
    r = await client.post(
        "/api/v1/documents/prescription",
        headers=_auth(token),
        json={
            "patient_id": patient_id,
            "items": [{"drug": "NotifDrug", "dose": "1 mg", "frequency": "daily"}],
            "send_to_patient": send,
        },
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


@pytest.mark.asyncio
async def test_issuing_a_document_notifies_the_patient(client: AsyncClient, bus: FakeBus):
    """The whole feature, end to end, with the patient already registered."""
    tag = uuid.uuid4().hex[:8]
    email = f"notif-{tag}@solum.test"

    doc = await _mk_doctor(client, tag)
    pat = await _mk_patient_row(client, doc["token"], f"Person {tag}", email)

    pt = await _patient_account(client, email, f"Person {tag}")
    account_id = pt["account"]["id"]

    assert bus.published == [], "nothing should be published before any document"

    issued = await _issue(client, doc["token"], pat["id"])

    mine = bus.for_recipient(account_id)
    assert len(mine) == 1, f"expected one notification, got {bus.published}"

    n = mine[0]
    assert n["kind"] == "document_issued"
    assert "prescription" in n["title"].lower(), n["title"]
    # This is the field whose absence raised AttributeError. If it is
    # missing the notification was built from a stale row shape and the
    # bell cannot resolve back to a document.
    assert n["payload"].get("reference"), n["payload"]
    assert n["payload"]["document_id"] == issued["id"]
    assert n["payload"]["patient_id"] == pat["id"]
    assert n["payload"]["doctor_name"], "the patient should see who issued it"


@pytest.mark.asyncio
async def test_document_issued_before_registration_still_reaches_the_inbox(
    client: AsyncClient, bus: FakeBus
):
    """Order should not matter for the document itself.

    The doctor may well issue a prescription before the patient has ever
    signed up. The notification is skipped - there is no account to notify
    - but the document must still be waiting when they arrive. Otherwise
    the first thing a new patient sees is an empty portal, and the doctor
    believes it was delivered.
    """
    tag = uuid.uuid4().hex[:8]
    email = f"late-{tag}@solum.test"

    doc = await _mk_doctor(client, tag)
    pat = await _mk_patient_row(client, doc["token"], f"Late {tag}", email)

    issued = await _issue(client, doc["token"], pat["id"])

    reg = await client.post(
        "/api/v1/auth/register/patient",
        json={"email": email, "password": PASS, "display_name": f"Late {tag}"},
    )
    assert reg.status_code in (200, 201), reg.text
    ptoken = reg.json()["token"]

    inbox = await client.get("/api/v1/patient/inbox", headers=_auth(ptoken))
    assert inbox.status_code == 200, inbox.text
    rows = inbox.json()
    assert len(rows) == 1, f"the document must survive being issued early: {rows}"
    assert rows[0]["document_id"] == issued["id"]
    assert rows[0]["read"] is False, "it should arrive unread"

    # Registration links the record *and* announces what is waiting, so the
    # bus already holds the announcement by the time we get here.
    #
    # This is the ordinary order for a new patient: the doctor writes the
    # prescription at the consultation, and the person signs up afterwards.
    # Announcing only on issue meant the bell was silent for exactly the
    # people who had the most to catch up on.
    # The bus is addressed by ACCOUNT id. `link_for_account` returns PATIENT
    # row ids, which are different identifiers - publishing to those puts
    # the announcement somewhere no session will ever read it. Asserting on
    # the account id is what pins that distinction down.
    account_id = reg.json()["account"]["id"]
    announced = bus.for_recipient(account_id)
    assert len(announced) == 1, (
        f"expected one announcement addressed to account {account_id}, "
        f"bus holds {bus.published}"
    )

    n = announced[0]
    assert n["kind"] == "documents_waiting"
    assert n["payload"]["backfilled"] is True
    assert n["payload"]["count"] == 1
    assert issued["id"] in n["payload"]["document_ids"]

    # One summary, not one notification per document, and the summary says
    # what is waiting rather than just that something is.
    assert "1 document waiting" in n["title"], n["title"]
    assert "prescription" in n["body"], n["body"]


@pytest.mark.asyncio
async def test_one_summary_covers_every_doctors_documents(
    client: AsyncClient, bus: FakeBus
):
    """A person with two doctors gets one summary, not two bells.

    Email linking exists precisely so that one person can be a patient of
    several practices at once. Announcing per doctor would make the second
    one feel like a separate event about a separate person.
    """
    tag = uuid.uuid4().hex[:8]
    email = f"shared-{tag}@solum.test"

    d_a = await _mk_doctor(client, f"a-{tag}")
    d_b = await _mk_doctor(client, f"b-{tag}")
    p_a = await _mk_patient_row(client, d_a["token"], f"Shared {tag}", email)
    p_b = await _mk_patient_row(client, d_b["token"], f"Shared {tag}", email)
    assert p_a["id"] != p_b["id"], "the two doctors must hold separate rows"

    # Issued before any account exists, so neither rings at the time.
    await _issue(client, d_a["token"], p_a["id"])
    await _issue(client, d_b["token"], p_b["id"])

    pt = await _patient_account(client, email, f"Shared {tag}")
    account_id = pt["account"]["id"]

    announced = bus.for_recipient(account_id)
    assert len(announced) == 1, f"expected a single summary, got {bus.published}"

    n = announced[0]
    assert n["kind"] == "documents_waiting"
    assert n["payload"]["count"] == 2, n["payload"]
    assert len(n["payload"]["document_ids"]) == 2
    assert "2 documents waiting" in n["title"], n["title"]


@pytest.mark.asyncio
async def test_announcement_is_not_repeated_on_later_sign_ins(
    client: AsyncClient, bus: FakeBus
):
    """A bell that re-rings every visit is worse than no bell."""
    tag = uuid.uuid4().hex[:8]
    email = f"repeat-{tag}@solum.test"

    doc = await _mk_doctor(client, tag)
    pat = await _mk_patient_row(client, doc["token"], f"Repeat {tag}", email)
    await _issue(client, doc["token"], pat["id"])

    await _patient_account(client, email, f"Repeat {tag}")
    after_first = len(bus.published)
    assert after_first == 1

    for _ in range(3):
        r = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": PASS}
        )
        assert r.status_code in (200, 201), r.text

    assert len(bus.published) == after_first, (
        f"re-announced on every sign-in: {bus.published}"
    )


@pytest.mark.asyncio
async def test_private_documents_are_not_announced_on_linking(
    client: AsyncClient, bus: FakeBus
):
    """Linking must not announce documents the clinic kept to itself."""
    tag = uuid.uuid4().hex[:8]
    email = f"quiet-{tag}@solum.test"

    doc = await _mk_doctor(client, tag)
    pat = await _mk_patient_row(client, doc["token"], f"Quiet {tag}", email)

    await _issue(client, doc["token"], pat["id"], send=False)

    await _patient_account(client, email, f"Quiet {tag}")

    assert bus.published == [], f"an unsent document must not be announced: {bus.published}"


@pytest.mark.asyncio
async def test_no_notification_when_the_doctor_does_not_send(
    client: AsyncClient, bus: FakeBus
):
    """A private document must not be announced.

    `send_to_patient` false means the doctor is keeping it in the clinic.
    Publishing a notification would tell the patient about a document they
    are not allowed to open.
    """
    tag = uuid.uuid4().hex[:8]
    email = f"private-{tag}@solum.test"

    doc = await _mk_doctor(client, tag)
    pat = await _mk_patient_row(client, doc["token"], f"Private {tag}", email)
    await _patient_account(client, email, f"Private {tag}")

    await _issue(client, doc["token"], pat["id"], send=False)

    assert bus.published == [], "an unsent document must not raise a notification"


@pytest.mark.asyncio
async def test_a_walk_in_with_no_account_does_not_break_issuing(
    client: AsyncClient, bus: FakeBus
):
    """The documented skip must stay quiet and must not raise.

    This is the early-return branch, which also named `document.reference`
    before the fix. A patient with no email has no account, so nothing is
    published - but issuing the prescription still has to succeed.
    """
    tag = uuid.uuid4().hex[:8]
    doc = await _mk_doctor(client, tag)
    pat = await _mk_patient_row(client, doc["token"], "No Email Walk In", None)

    issued = await _issue(client, doc["token"], pat["id"])

    assert issued["id"]
    assert (issued["content"] or {}).get("reference"), "the document keeps its reference"

    # And the doctor can still read it back.
    listed = await client.get(
        f"/api/v1/documents/patient/{pat['id']}", headers=_auth(doc["token"])
    )
    assert listed.status_code == 200
    assert any(d["id"] == issued["id"] for d in listed.json())


@pytest.mark.asyncio
async def test_notifications_are_scoped_to_their_recipient(client: AsyncClient, bus: FakeBus):
    """One patient's bell must not carry another's documents."""
    tag = uuid.uuid4().hex[:8]
    doc = await _mk_doctor(client, tag)

    mine = f"mine-{tag}@solum.test"
    theirs = f"theirs-{tag}@solum.test"
    p_mine = await _mk_patient_row(client, doc["token"], "Mine", mine)
    p_theirs = await _mk_patient_row(client, doc["token"], "Theirs", theirs)

    a_mine = (await _patient_account(client, mine, "Mine"))["account"]["id"]
    a_theirs = (await _patient_account(client, theirs, "Theirs"))["account"]["id"]

    await _issue(client, doc["token"], p_mine["id"])
    await _issue(client, doc["token"], p_theirs["id"])

    assert len(bus.published) == 2, bus.published

    mine_ids = {n["payload"]["patient_id"] for n in bus.for_recipient(a_mine)}
    theirs_ids = {n["payload"]["patient_id"] for n in bus.for_recipient(a_theirs)}

    assert mine_ids == {p_mine["id"]}, f"leaked into the wrong bell: {mine_ids}"
    assert theirs_ids == {p_theirs["id"]}, f"leaked into the wrong bell: {theirs_ids}"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_the_bell_counts_and_clears(client: AsyncClient):
    """An unread marker has to mean something until it is looked at.

    Against the real Redis bus, because the bug is in how read state is
    persisted: `read()` used to leave `read` at its default False on every
    notification, and `mark_read` only advanced a consumer group that
    `read()` never consulted. So the badge counted everything ever
    published and could never fall. A stubbed bus cannot reproduce that,
    which is why this one skips rather than faking.

    Skips cleanly where `redis` is not installed - the API image, not the
    bare host venv.
    """
    pytest.importorskip("redis", reason="redis client not installed in this venv")

    tag = uuid.uuid4().hex[:8]
    email = f"read-{tag}@solum.test"

    health = await client.get("/api/v1/notifications/health")
    if health.status_code != 200 or not health.json().get("ok"):
        pytest.skip("no Redis reachable from here")

    doc = await _mk_doctor(client, tag)
    pat = await _mk_patient_row(client, doc["token"], f"Read {tag}", email)
    ptoken = (
        await client.post(
            "/api/v1/auth/register/patient",
            json={"email": email, "password": PASS, "display_name": f"Read {tag}"},
        )
    ).json()["token"]

    issued = await _issue(client, doc["token"], pat["id"])

    count = await client.get("/api/v1/notifications/unread-count", headers=_auth(ptoken))
    assert count.json()["unread"] == 1, count.json()

    items = (await client.get("/api/v1/notifications", headers=_auth(ptoken))).json()
    mine = [n for n in items if n["payload"].get("document_id") == issued["id"]]
    assert len(mine) == 1, f"the notification must carry the document id: {items}"
    assert mine[0]["payload"].get("reference"), "and the reference"
    assert mine[0]["read"] is False, "a fresh notification is unread"

    ids = [n["id"] for n in items]
    marked = await client.post(
        "/api/v1/notifications/read", headers=_auth(ptoken), json={"ids": ids}
    )
    assert marked.status_code == 200, marked.text
    assert marked.json()["marked"] == len(ids), marked.json()

    after = await client.get("/api/v1/notifications/unread-count", headers=_auth(ptoken))
    assert after.json()["unread"] == 0, f"the bell should clear once read: {after.json()}"

    # The list itself must also report them read, not just the count.
    reread = (await client.get("/api/v1/notifications", headers=_auth(ptoken))).json()
    assert all(n["read"] for n in reread), reread

    # And it must survive a fresh fetch rather than being a cached effect.
    again = await client.get("/api/v1/notifications/unread-count", headers=_auth(ptoken))
    assert again.json()["unread"] == 0, again.json()


@pytest.mark.asyncio
async def test_reading_notifications_does_not_touch_another_patient(
    client: AsyncClient,
):
    """Acknowledgement is per recipient.

    The stream and its consumer group are shared by everyone. Recording
    read state on the group would advance it for all of them, so a patient
    clearing their bell could consume another patient's notifications. This
    is the exact reason the state lives in a per-account set.
    """
    pytest.importorskip("redis", reason="redis client not installed in this venv")
    health = await client.get("/api/v1/notifications/health")
    if health.status_code != 200 or not health.json().get("ok"):
        pytest.skip("no Redis reachable from here")

    tag = uuid.uuid4().hex[:8]
    doc = await _mk_doctor(client, tag)

    email_a, email_b = f"iso-a-{tag}@solum.test", f"iso-b-{tag}@solum.test"
    p_a = await _mk_patient_row(client, doc["token"], "Iso A", email_a)
    p_b = await _mk_patient_row(client, doc["token"], "Iso B", email_b)
    tok_a = (await _patient_account(client, email_a, "Iso A"))["token"]
    tok_b = (await _patient_account(client, email_b, "Iso B"))["token"]

    await _issue(client, doc["token"], p_a["id"])
    await _issue(client, doc["token"], p_b["id"])

    assert (await client.get("/api/v1/notifications/unread-count", headers=_auth(tok_b))).json()["unread"] == 1

    # A reads theirs.
    ids_a = [
        n["id"]
        for n in (await client.get("/api/v1/notifications", headers=_auth(tok_a))).json()
    ]
    await client.post("/api/v1/notifications/read", headers=_auth(tok_a), json={"ids": ids_a})

    assert (await client.get("/api/v1/notifications/unread-count", headers=_auth(tok_a))).json()["unread"] == 0
    b_after = await client.get("/api/v1/notifications/unread-count", headers=_auth(tok_b))
    assert b_after.json()["unread"] == 1, (
        f"reading A's notifications consumed B's: {b_after.json()}"
    )
