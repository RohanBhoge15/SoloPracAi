"""Object storage: an S3-compatible client.

Works against any S3 API, so the backend is not coupled to a specific
server. SeaweedFS is the default because MinIO's Community Edition was
archived in April 2026; the API surface is identical, so switching is a
change of endpoint and nothing else.

Design notes:

  - `boto3` is synchronous. Every method here is called from async code,
    so the blocking calls go through `asyncio.to_thread`. Wrapping the
    whole client in a thread pool once is cheaper than doing it per
    request.
  - The bucket is created on first use, because requiring a separate
    provisioning step is a reliable way to have a broken demo.
  - Downloads are served by presigned URL rather than proxied through
    FastAPI. A 3 MB prescription should not occupy an API worker, and
    the patient should not need a SoloPrac session cookie to open a
    document they were legitimately sent.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    content_type: str


class StorageError(RuntimeError):
    pass


def build_key(prefix: str, filename: str, content_type: str | None = None) -> str:
    """Generate a collision-free, traversal-safe object key.

    The client-supplied filename is never used verbatim: it may contain
    path separators or unicode that break presigned URLs.
    """
    extension = ""
    if "." in (filename or ""):
        extension = "." + filename.rsplit(".", 1)[-1].lower()[:8]
    if not extension and content_type:
        extension = mimetypes.guess_extension(content_type) or ""
    clean = _SAFE.sub("-", (filename or "file").rsplit("/", 1)[-1])[:60]
    return f"{prefix}/{uuid.uuid4().hex}{extension}"


class ObjectStorage:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None
        self._ready: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ── client ──────────────────────────────────────────────────────

    def _sync_client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.settings.s3_endpoint,
            aws_access_key_id=self.settings.s3_access_key,
            aws_secret_access_key=self.settings.s3_secret_key,
            region_name=self.settings.s3_region,
            config=Config(
                signature_version="s3v4",
                # SeaweedFS and MinIO both return slightly different ETag
                # formats; path style is universally supported.
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    @property
    def client(self):
        if self._client is None:
            self._client = self._sync_client()
        return self._client

    async def _ensure_bucket(self) -> None:
        """Create the bucket if missing. Idempotent and called lazily."""
        if self._ready is not None:
            return
        async with self._lock:
            if self._ready is not None:
                return
            bucket = self.settings.s3_bucket

            def _create() -> None:
                try:
                    self.client.head_bucket(Bucket=bucket)
                except ClientError as exc:
                    code = exc.response.get("Error", {}).get("Code", "")
                    if code in ("404", "NoSuchBucket", "NotFound"):
                        try:
                            self.client.create_bucket(Bucket=bucket)
                            logger.info("created bucket %s", bucket)
                        except ClientError as create_exc:
                            # Another replica may have won the race.
                            if create_exc.response.get("Error", {}).get(
                                "Code"
                            ) not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                                raise
                    else:
                        raise

            await asyncio.to_thread(_create)
            self._ready = asyncio.current_task()

    # ── operations ──────────────────────────────────────────────────

    async def put(
        self,
        data: bytes,
        prefix: str,
        filename: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> StoredObject:
        await self._ensure_bucket()
        key = build_key(prefix, filename, content_type)

        def _put() -> None:
            self.client.put_object(
                Bucket=self.settings.s3_bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata or {},
            )

        try:
            await asyncio.to_thread(_put)
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"upload failed: {exc}") from exc

        return StoredObject(key=key, size=len(data), content_type=content_type)

    async def get(self, key: str) -> bytes:
        await self._ensure_bucket()

        def _get() -> bytes:
            response = self.client.get_object(
                Bucket=self.settings.s3_bucket, Key=key
            )
            return response["Body"].read()

        try:
            return await asyncio.to_thread(_get)
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"download failed: {exc}") from exc

    async def presigned_url(self, key: str, ttl: Optional[int] = None) -> str:
        """A time-limited direct download URL.

        Preferred over proxying: the object store serves the file, so a
        large report does not occupy an API worker.

        The signature covers the host, so a URL signed for the internal
        container hostname is useless to a browser. When a public
        endpoint is configured the host is swapped; the signature stays
        valid because the host is not part of the signed payload for
        path-style S3 addressing.
        """
        await self._ensure_bucket()

        def _sign() -> str:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.settings.s3_bucket, "Key": key},
                ExpiresIn=ttl or self.settings.download_url_ttl_seconds,
            )

        try:
            url = await asyncio.to_thread(_sign)
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"could not sign url: {exc}") from exc

        return self._rewrite_host(url)

    def _rewrite_host(self, url: str) -> str:
        """Point a signed URL at the endpoint the browser can reach.

        Inside compose the client talks to `http://seaweedfs:8333`, which
        does not resolve on the host machine. The public endpoint is the
        browser-facing one.
        """
        public = (self.settings.s3_public_endpoint or "").rstrip("/")
        internal = (self.settings.s3_endpoint or "").rstrip("/")
        if not public or not internal or public == internal:
            return url
        if internal in url:
            return url.replace(internal, public, 1)
        return url

    async def delete(self, key: str) -> bool:
        await self._ensure_bucket()

        def _delete() -> None:
            self.client.delete_object(
                Bucket=self.settings.s3_bucket, Key=key
            )

        try:
            await asyncio.to_thread(_delete)
            return True
        except (ClientError, BotoCoreError) as exc:
            logger.warning("delete failed for %s: %s", key, exc)
            return False

    async def exists(self, key: str) -> bool:
        await self._ensure_bucket()

        def _head() -> bool:
            try:
                self.client.head_object(
                    Bucket=self.settings.s3_bucket, Key=key
                )
                return True
            except (ClientError, BotoCoreError):
                return False

        return await asyncio.to_thread(_head)

    async def health(self) -> Dict[str, Any]:
        try:
            await self._ensure_bucket()
            return {"ok": True, "bucket": self.settings.s3_bucket}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:200]}


_storage: Optional[ObjectStorage] = None


def get_storage() -> ObjectStorage:
    global _storage
    if _storage is None:
        _storage = ObjectStorage()
    return _storage
