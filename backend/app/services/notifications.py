"""Notifications on Redis Streams.

Why Streams and not Pub/Sub:

  - Pub/Sub drops messages when a subscriber is disconnected. A
    prescription sent while a patient's phone is asleep would vanish, and
    nothing would show that it ever existed.
  - Streams persist. A consumer group tracks per-recipient delivery
    offsets, so an unread notification survives a restart, a redeploy, or
    a phone that was offline for a day.

The trade-off is that Streams need trimming and consumer-group upkeep;
both are handled here rather than leaking into the routes.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.db.identity import Doctor
from app.db.models import Patient

logger = logging.getLogger(__name__)


@dataclass
class Notification:
    id: str
    recipient_account_id: str
    kind: str
    title: str
    body: str
    payload: Dict[str, Any]
    created_at: datetime
    read: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "read": self.read,
        }


class NotificationBus:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._redis = None
        self._group_ready = False

    @property
    def stream(self) -> str:
        return self.settings.notification_stream

    async def client(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self.settings.redis_url, decode_responses=True
            )
        return self._redis

    async def _ensure_group(self) -> None:
        """Create the consumer group once per process.

        BUSYGROUP means it already exists, which is not an error.
        """
        if self._group_ready:
            return
        client = await self.client()
        try:
            # Positional, not keyword. redis-py names the second argument
            # `groupname`; passing `group=` raises TypeError on every call,
            # which is how the whole notification surface 500s.
            await client.xgroup_create(
                self.stream, "solum", id="0", mkstream=True
            )
        except Exception as exc:  # noqa: BLE001
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_ready = True

    async def publish(
        self,
        *,
        recipient_account_id: str,
        kind: str,
        title: str,
        body: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Push one notification. Never raises.

        A notification failure must not roll back the action that caused
        it - a prescription is still a prescription if the bell did not
        ring.
        """
        entry = {
            "recipient": str(recipient_account_id),
            "kind": kind,
            "title": title,
            "body": body,
            "payload": json.dumps(payload or {}),
            "at": str(time.time()),
        }
        try:
            client = await self.client()
            message_id = await client.xadd(
                self.stream, entry, maxlen=self.settings.notification_maxlen,
                approximate=True,
            )
            logger.debug("published %s to %s", kind, recipient_account_id)
            return message_id.decode() if isinstance(message_id, bytes) else str(message_id)
        except Exception:  # noqa: BLE001
            logger.warning("notification publish failed", exc_info=True)
            return None

    @property
    def read_key(self) -> str:
        """Key holding which messages this recipient has acknowledged.

        Kept separate from the stream rather than inferred from the
        consumer group. The group is shared across every recipient of one
        stream, and `read()` scans with XREVRANGE filtered by a `recipient`
        field, so a group offset cannot express "this person read their
        own messages". It is also the wrong shape: XACK advances the group
        globally, which would mark other people's notifications as
        delivered.
        """
        return self.settings.notification_read_key

    def _ack_key(self, recipient_account_id: str) -> str:
        return f"{self.read_key}:{recipient_account_id}"

    async def read(
        self,
        recipient_account_id: str,
        limit: int = 50,
        *,
        only_unread: bool = False,
    ) -> List[Notification]:
        """Fetch this recipient's notifications, newest first.

        Reads by filtering the stream rather than maintaining a per-user
        list. With a bounded stream (maxlen) that is cheap, and it means
        an offline recipient loses nothing.

        `read` is resolved against the recipient's own acknowledgement set.
        It used to be left at its default of False on every notification,
        which meant the bell counted everything it had ever published and
        could never be cleared - `mark_read` acknowledged the messages but
        this method never asked.
        """
        await self._ensure_group()
        client = await self.client()
        try:
            entries = await client.xrevrange(self.stream, "+", "-", count=limit * 4)
        except Exception:  # noqa: BLE001
            logger.warning("notification read failed", exc_info=True)
            return []

        acked: set[str] = set()
        try:
            acked = {str(v) for v in await client.smembers(self._ack_key(recipient_account_id))}
        except Exception:  # noqa: BLE001
            # A missing acknowledgement set is not a reason to misreport
            # everything as unread-or-read; treat it as "none acknowledged".
            logger.warning("could not read acknowledgements", exc_info=True)

        out: List[Notification] = []
        for message_id, fields in entries:
            if fields.get("recipient") != str(recipient_account_id):
                continue
            mid = message_id.decode() if isinstance(message_id, bytes) else str(message_id)
            try:
                item = Notification(
                    id=mid,
                    recipient_account_id=str(recipient_account_id),
                    kind=fields.get("kind", "unknown"),
                    title=fields.get("title", ""),
                    body=fields.get("body", ""),
                    payload=json.loads(fields.get("payload", "{}")),
                    created_at=datetime.fromtimestamp(
                        float(fields.get("at", 0)), tz=timezone.utc
                    ),
                    read=mid in acked,
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if only_unread and item.read:
                continue
            out.append(item)
            if len(out) >= limit:
                break
        return out

    async def mark_read(
        self, recipient_account_id: str, message_ids: List[str]
    ) -> int:
        """Record these messages as read for this recipient.

        Scoped to the recipient, deliberately. The stream and its consumer
        group are shared, so advancing the group offset on one person's
        behalf would silently consume everybody else's notifications. The
        per-recipient set is what `read()` consults.
        """
        if not message_ids:
            return 0
        await self._ensure_group()
        client = await self.client()
        ids = [str(m) for m in message_ids]
        try:
            await client.sadd(self._ack_key(recipient_account_id), *ids)
            # Bounded so the set cannot grow without limit on a busy
            # account; a generous ceiling, since only unread ids are ever
            # added and the stream itself is capped.
            await client.scard(self._ack_key(recipient_account_id))
            return len(ids)
        except Exception:  # noqa: BLE001
            logger.warning("mark-read failed", exc_info=True)
            return 0

    async def unread_count(self, recipient_account_id: str) -> int:
        """How many notifications this recipient has not read."""
        return len(
            await self.read(recipient_account_id, limit=100, only_unread=True)
        )

    async def health(self) -> Dict[str, Any]:
        try:
            client = await self.client()
            info = await client.xlen(self.stream)
            return {"ok": True, "stream": self.stream, "length": int(info)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:200]}


_bus: Optional[NotificationBus] = None


def get_bus() -> NotificationBus:
    global _bus
    if _bus is None:
        _bus = NotificationBus()
    return _bus


# ── domain events ──────────────────────────────────────────────────


async def notify_document_issued(
    session,
    *,
    patient: Patient,
    doctor: Doctor,
    document,
) -> Optional[str]:
    """Tell a registered patient that a document is waiting for them.

    A walk-in with no account has nowhere to send it, so nothing is
    published and the document simply waits in the doctor's records.
    """
    # `reference` is inside the JSON `content` column rather than a column
    # of its own. Naming it here as well as below, so this early-return
    # cannot raise on a walk-in - which is the one path that is supposed to
    # be quiet.
    reference = (document.content or {}).get("reference")

    if not patient.user_id:
        logger.debug(
            "patient %s has no account; document %s not delivered",
            patient.id,
            reference,
        )
        return None

    from app.db.identity import Account

    account = await session.get(Account, doctor.account_id)
    doctor_name = account.display_name if account else "Your doctor"

    labels = {
        "prescription": "prescription",
        "certificate": "medical certificate",
        "invoice": "invoice",
        "weekly_report": "weekly report",
    }
    label = labels.get(document.kind, document.kind)

    # Note on what used to be here: `document.reference`. There is no such
    # attribute - the reference lives in the JSON `content` column - so this
    # raised AttributeError on every document. The caller swallows
    # notification failures on purpose, so a Redis problem cannot cost the
    # doctor a prescription, and that same blanket catch hid a hard bug:
    # documents issued, inbox correct, bell stuck at zero, one WARNING line
    # per document. Anything swallowed this broadly needs a test that
    # asserts the happy path reached the other end.
    return await get_bus().publish(
        recipient_account_id=str(patient.user_id),
        kind="document_issued",
        title=f"New {label} from {doctor_name}",
        body=document.title,
        payload={
            "document_id": str(document.id),
            "document_kind": document.kind,
            "reference": reference,
            "patient_id": str(patient.id),
            "doctor_name": doctor_name,
            "doctor_specialty": doctor.specialty,
        },
    )
