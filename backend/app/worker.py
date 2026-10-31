"""Background worker.

Only one job genuinely belongs here: the MedGemma deep read. It takes
about a minute and 5 GB on a CPU-only host, which is far too long to sit
inside an HTTP request - the client would time out and the work would be
lost with nothing to show for it.

Everything else is synchronous on purpose. A single-record question
answers in milliseconds, and a weekly report is requested deliberately by
a clinician who is waiting for it. Moving those off the request path
would add a queue, a retry policy and a second failure mode to save
latency nobody is currently feeling.

The queue is a Redis stream rather than Celery. Celery is already a
declared dependency for no other reason, and a broker that is not
actually used is a dependency that can break the build.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from typing import Any, Dict, Optional

from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("soloprac.worker")

#: Stream of pending deep reads. Separate from the notification stream:
#: one carries work, the other carries messages to a person.
JOB_STREAM = "solum:jobs"
CONSUMER_GROUP = "solum-workers"
CONSUMER_NAME = "worker-1"

#: A deep read is a CPU-bound forward pass in a thread. The GIL is released
#: for the duration, so the event loop stays responsive, but two at once
#: would need 10 GB and there is not that much.
MAX_CONCURRENT = 1


class DeepReadWorker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._redis = None
        self._stopping = asyncio.Event()

    async def redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self.settings.redis_url, decode_responses=True
            )
        return self._redis

    async def ensure_group(self) -> None:
        client = await self.redis()
        try:
            # Positional: redis-py names the second argument `groupname`.
            await client.xgroup_create(
                JOB_STREAM, CONSUMER_GROUP, id="0", mkstream=True
            )
            logger.info("created job group %s", CONSUMER_GROUP)
        except Exception as exc:  # noqa: BLE001
            if "BUSYGROUP" not in str(exc):
                raise

    async def enqueue(self, *, document_key: str, patient_id: str) -> str:
        """Queue a deep read. Returns the job id."""
        client = await self.redis()
        job = json.dumps(
            {
                "kind": "deep_read",
                "object_key": document_key,
                "patient_id": patient_id,
            }
        )
        return await client.xadd(JOB_STREAM, {"job": job}, maxlen=1000)

    async def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> None:
        await self.ensure_group()
        client = await self.redis()
        logger.info("worker listening on %s", JOB_STREAM)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        while not self._stopping.is_set():
            try:
                batches = await client.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {JOB_STREAM: ">"},
                    count=1,
                    block=5000,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # Redis restarting is normal in development. Back off and
                # keep the loop alive rather than exiting into a crash loop.
                logger.warning("job stream read failed; retrying", exc_info=True)
                await asyncio.sleep(3)
                continue

            for _stream, messages in batches or []:
                for message_id, fields in messages:
                    await semaphore.acquire()
                    try:
                        await self._handle(message_id, fields.get("job", ""))
                    finally:
                        semaphore.release()

    async def _handle(self, message_id: str, raw: str) -> None:
        client = await self.redis()
        try:
            job = json.loads(raw)
        except ValueError:
            logger.warning("unparseable job %s; acknowledging", message_id)
            await client.xack(JOB_STREAM, CONSUMER_GROUP, message_id)
            return

        if job.get("kind") != "deep_read":
            logger.info("ignoring unknown job kind %r", job.get("kind"))
            await client.xack(JOB_STREAM, CONSUMER_GROUP, message_id)
            return

        started = asyncio.get_running_loop().time()
        try:
            ok, detail = await self._deep_read(job)
            logger.info(
                "deep read for %s finished in %.1fs (ok=%s)",
                job.get("patient_id"),
                asyncio.get_running_loop().time() - started,
                ok,
            )
            if not ok:
                logger.warning("deep read declined: %s", detail)
        except Exception:  # noqa: BLE001
            logger.exception("deep read failed for %s", job.get("patient_id"))
        finally:
            # Acknowledged even on failure. Retrying a 5 GB model read that
            # already timed out once would just fill the queue; the
            # document stays in the review queue either way, which is the
            # safe state.
            await client.xack(JOB_STREAM, CONSUMER_GROUP, message_id)

    async def _deep_read(self, job: Dict[str, Any]) -> tuple[bool, str]:
        if not self.settings.enable_vision_llm:
            return False, "MedGemma is disabled on this deployment"

        from app.services.storage import get_storage
        from app.services.vision_llm import transcribe

        raw = await get_storage().get(job["object_key"])
        # The model call is blocking and takes a minute; the thread keeps
        # the event loop free to accept the next request meanwhile.
        result = await asyncio.to_thread(transcribe, raw, self.settings)
        if result.error:
            return False, result.error
        logger.info(
            "transcribed %d row(s) from %s",
            len(result.rows),
            job["object_key"],
        )
        return True, f"{len(result.rows)} rows"


async def main() -> int:
    worker = DeepReadWorker()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.stop()))
        except NotImplementedError:  # pragma: no cover - Windows
            pass

    logger.info(
        "starting; deep reads %s",
        "enabled" if worker.settings.enable_vision_llm else "disabled",
    )
    try:
        await worker.run()
    except asyncio.CancelledError:
        logger.info("cancelled")
    finally:
        if worker._redis is not None:
            await worker._redis.aclose()
        logger.info("worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
