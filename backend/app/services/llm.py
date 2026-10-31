"""LLM client for the OpenRouter free tier.

Three properties this must have, in order of importance:

1. **The de-identification gate is not optional.** Every outbound payload
   is asserted clean before it leaves. A failure blocks the request.
2. **Responses are frozen to Postgres.** Free-tier models change or vanish
   without notice. A paper's numbers must reproduce against a stored
   artifact, not a moving endpoint.
3. **Degradation is graceful.** Free tiers are slow and rate-limited. The
   static-shortcut tier answers common intents with no LLM at all, and
   every stage caches on redis.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import Settings, get_settings
from app.services.pii import Deidentifier, assert_payload_safe

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical record assistant for a solo physician.

You answer ONLY from the records provided. Rules, in priority order:

1. NEVER invent a clinical value. If a number is not in a record, omit it.
2. Cite every claim with the record IDs it came from.
3. If a record has been corrected, report the CORRECTED value. You may
   mention the earlier value only in past tense, and only if asked.
4. Respond in the language of the question.
5. Be brief. A physician is reading this between patients.

Return strict JSON: {"claims": [{"text": "...", "event_ids": ["..."]}]}"""


class LLMError(RuntimeError):
    pass


class DoNotFallBack(LLMError):
    """The request itself is at fault, so no other provider will fare better.

    Distinct from exhaustion. A 400 means the payload or the model name is
    wrong; sending the identical request to a second vendor would multiply
    the latency to reach the same rejection, and would disguise a request bug
    as an availability problem.
    """


class DeidentificationBlocked(LLMError):
    """Raised when an outbound payload cannot be proven safe. Never sent."""


@dataclass
class LLMResult:
    text: str
    model: str
    cached: bool = False
    latency_ms: float = 0.0
    attempts: int = 1
    degraded: bool = False
    #: Which provider actually answered. Recorded because "the answer came
    #: from the fallback" is a materially different fact from "the primary
    #: answered", and the UI surfaces it rather than hiding the difference.
    provider: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "cached": self.cached,
            "latency_ms": round(self.latency_ms, 1),
            "attempts": self.attempts,
            "degraded": self.degraded,
            "provider": self.provider,
        }


@dataclass
class RedisCache:
    """Thin async cache. Degrades to a no-op if redis is unavailable."""

    client: Any = None
    enabled: bool = True

    @staticmethod
    def key_for(prompt: str, system: str, model: str, temperature: float) -> str:
        digest = hashlib.sha256(
            f"{system}\x00{prompt}\x00{model}\x00{temperature}".encode()
        ).hexdigest()
        return f"solum:llm:{digest}"

    async def get(self, key: str) -> Optional[str]:
        if not (self.enabled and self.client):
            return None
        try:
            raw = await self.client.get(key)
            return raw.decode() if isinstance(raw, bytes) else raw
        except Exception:  # pragma: no cover - cache must never break a request
            logger.debug("cache get failed", exc_info=True)
            return None

    async def set(self, key: str, value: str, ttl: int) -> None:
        if not (self.enabled and self.client):
            return
        try:
            await self.client.set(key, value, ex=ttl)
        except Exception:  # pragma: no cover
            logger.debug("cache set failed", exc_info=True)


class CircuitBreaker:
    """Per-provider breaker that stops calling an exhausted endpoint.

    The reason this exists rather than "just fall back": OpenRouter's free
    tier is 50 requests *per day*, and every call to an exhausted endpoint
    is a call that cannot succeed. Falling through on each turn therefore
    spends two requests to serve one answer, so the budget drains twice as
    fast and the fallback inherits the same fate sooner. A breaker converts
    a repeated cost into a single one.

    Cooldown doubles per consecutive failure up to a ceiling, because a
    provider that has just refused us is unlikely to recover in four seconds
    but may well recover in an hour. One trial request is admitted after the
    cooldown (half-open); success closes the breaker, failure reopens it for
    twice as long.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        base_cooldown: float = 900.0,
        max_cooldown: float = 3600.0,
        enabled: bool = True,
        clock: Optional[Any] = None,
    ):
        self.name = name
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown
        self.enabled = enabled
        self._clock = clock or time.monotonic
        self.failures = 0
        self.opened_at: Optional[float] = None

    # ── state ─────────────────────────────────────────────────────────

    def _now(self) -> float:
        return float(self._clock())

    @property
    def cooldown(self) -> float:
        if self.failures <= 1:
            return self.base_cooldown
        return min(self.max_cooldown, self.base_cooldown * (2 ** (self.failures - 1)))

    def state(self) -> str:
        if not self.enabled or self.failures == 0:
            return self.CLOSED
        if self.opened_at is None:
            return self.CLOSED
        if (self._now() - self.opened_at) >= self.cooldown:
            return self.HALF_OPEN
        return self.OPEN

    def allows_request(self) -> bool:
        """Whether to spend a request on this provider at all."""
        return self.state() != self.OPEN

    def seconds_until_probe(self) -> float:
        if self.state() != self.OPEN or self.opened_at is None:
            return 0.0
        return max(0.0, self.cooldown - (self._now() - self.opened_at))

    # ── transitions ───────────────────────────────────────────────────

    def record_success(self) -> None:
        if self.failures:
            logger.info("circuit %s closed after recovery", self.name)
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        self.opened_at = self._now()
        logger.warning(
            "circuit %s open for %.0fs (consecutive failures: %d)",
            self.name, self.cooldown, self.failures,
        )

    def reset(self) -> None:
        self.failures = 0
        self.opened_at = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "state": self.state(),
            "failures": self.failures,
            "cooldown_s": self.cooldown,
            "retry_in_s": round(self.seconds_until_probe(), 1),
        }


class LLMClient:
    """OpenAI-compatible client with a mandatory outbound safety gate.

    Three providers are tried in order, and the order is a setting
    (`llm_primary`). The chain ships as **OpenCode Zen, then Groq, then
    OpenRouter**:

    * `space-bunny-free` is the primary, because two working free tiers
      behind it mean exhaustion costs prose rather than function.
    * Groq's open models answer in ~0.1 s against the primary's seconds,
      which is what keeps the demo responsive.
    * OpenRouter is last because its free allowance is a hard 50/day for the
      whole key and has been observed returning 429.

    Ordering is deterministic, and a fallback only runs once the provider
    ahead of it has genuinely exhausted its retries and opened its circuit
    breaker.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[RedisCache] = None,
        breakers: Optional[Dict[str, CircuitBreaker]] = None,
    ):
        self.settings = settings or get_settings()
        self.cache = cache or RedisCache()
        self._client: Optional[httpx.AsyncClient] = None
        # Process-local on purpose. Breaker state is about "is this endpoint
        # working right now", which a single API process is authoritative
        # for, and sharing it through redis would add a network round trip
        # to the one code path that exists because the network is failing.
        self._breakers: Dict[str, CircuitBreaker] = breakers or {
            name: CircuitBreaker(
                name,
                base_cooldown=self.settings.llm_breaker_cooldown_s,
                max_cooldown=self.settings.llm_breaker_max_cooldown_s,
                enabled=self.settings.llm_breaker_enabled,
            )
            for name in ("opencode", "groq", "openrouter")
        }

    def breaker_state(self) -> List[Dict[str, Any]]:
        return [b.as_dict() for b in self._breakers.values()]

    async def __aenter__(self) -> "LLMClient":
        # No base_url and no auth header here: three providers means three
        # base URLs and three keys, so all are supplied per request. A
        # shared default would silently send one vendor's secret to
        # another vendor.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.llm_timeout_s)
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _headers(provider: str, settings: Settings) -> Dict[str, str]:
        """Auth and identification for one provider.

        Each provider gets its own key. A shared default would send one
        vendor's secret to another vendor, so the branch is exhaustive on
        purpose and an unknown name raises rather than falling through to
        OpenRouter - a mis-set provider must not quietly authenticate as
        whoever happens to be in the `else`.

        The `HTTP-Referer`/`X-Title` pair is OpenRouter's attribution
        convention and is not sent to the others, where it would be
        meaningless noise.
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SoloPrac/1.0 (clinical research; +http://localhost:8080)",
        }
        if provider == "opencode":
            headers["Authorization"] = f"Bearer {settings.opencode_api_key or ''}"
        elif provider == "groq":
            # Groq speaks the same OpenAI dialect and the same Bearer scheme,
            # so the only thing distinguishing it is which key is used.
            headers["Authorization"] = f"Bearer {settings.groq_api_key or ''}"
        elif provider == "openrouter":
            headers["Authorization"] = f"Bearer {settings.openrouter_api_key or ''}"
            headers["HTTP-Referer"] = "http://localhost:8080"
            headers["X-Title"] = "SoloPrac"
        else:
            raise ValueError(f"unknown LLM provider {provider!r}")
        return headers

    # ── the gate ─────────────────────────────────────────────────────

    def _guard(self, payload: Dict[str, Any]) -> None:
        """Block any outbound payload that is not provably de-identified."""
        if not self.settings.enforce_deidentification:
            return
        try:
            assert_payload_safe(payload)
        except Exception as exc:
            raise DeidentificationBlocked(str(exc)) from exc

    # ── one provider, with its own retry budget ──────────────────────

    async def _attempt(
        self,
        provider: str,
        base_url: str,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        use_cache: bool,
        cache_key: str,
        # True when no provider follows this one. Only affects the log
        # wording: with three links, "degrading" on the first two is a lie -
        # the turn is handed over, not abandoned. Defaults True so a lone
        # direct call keeps the old, correct-for-one wording.
        last: bool = True,
    ) -> Optional[LLMResult]:
        """Try one provider. Returns None if it is exhausted, never raises.

        `None` means "this provider cannot serve this prompt" and is the only
        thing that lets the caller advance to the next provider. A safety
        failure is not exhaustion: `DeidentificationBlocked` propagates, so a
        blocked payload can never be re-sent to a second vendor.
        """
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = self._headers(provider, self.settings)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        started = time.perf_counter()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                response = await self._client.post(url, json=body, headers=headers)
                if response.status_code == 429:
                    # A 429 on a shared free tier is usually not "this request
                    # was too big" - it is "this key's daily budget is spent",
                    # which no amount of waiting fixes. Retry once briefly in
                    # case it is transient contention, then hand the turn back
                    # so the next provider gets a turn.
                    if attempt < self.settings.llm_max_retries:
                        delay = min(4.0, (2**attempt))
                        logger.info(
                            "%s rate limited (429); one brief retry in %.1fs "
                            "(attempt %d of %d)",
                            provider, delay, attempt,
                            self.settings.llm_max_retries,
                        )
                        await asyncio.sleep(delay)
                        last_error = LLMError("rate limited")
                        continue
                    logger.warning(
                        "%s rate limited after %d attempts; %s",
                        provider, attempt,
                        "degrading"
                        if last else "advancing to the next provider",
                    )
                    return None
                response.raise_for_status()
                data = response.json()
                text = (data["choices"][0]["message"].get("content") or "").strip()
                latency = (time.perf_counter() - started) * 1000

                # Free tiers occasionally return 200 with an empty body. That
                # is not a usable answer, so it is retried rather than cached:
                # caching an empty response pins the failure for the whole TTL.
                if not text:
                    last_error = LLMError(f"empty response from {provider}")
                    if attempt < self.settings.llm_max_retries:
                        await asyncio.sleep(min(6.0, 2**attempt))
                        continue
                    return None

                if use_cache:
                    await self.cache.set(
                        cache_key,
                        json.dumps({"text": text, "model": data.get("model", model)}),
                        self.settings.llm_cache_ttl_s,
                    )

                return LLMResult(
                    text=text,
                    model=data.get("model", model),
                    latency_ms=latency,
                    attempts=attempt,
                    provider=provider,
                )
            except DeidentificationBlocked:
                raise  # Never retry, and never hand to another vendor.
            except (httpx.HTTPStatusError,) as exc:
                # 401/403 reject OUR credentials at THIS vendor, not the
                # request shape - the next provider has a different key and
                # may serve fine. Seen live: OpenCode's free tier started
                # returning 403 "can only be used from within OpenCode" for
                # all external calls, which is a dead link, not a bad
                # request. Any other 4xx means the payload itself is
                # malformed and every vendor would reject it identically.
                status = exc.response.status_code
                if status in (401, 403):
                    logger.warning(
                        "%s refused our credentials (%d); advancing to the "
                        "next provider",
                        provider, status,
                    )
                    return None
                if 400 <= status < 500:
                    raise DoNotFallBack(
                        f"{provider} rejected the request with {status}"
                    ) from exc
                last_error = exc
                if attempt < self.settings.llm_max_retries:
                    await asyncio.sleep(min(8.0, 2**attempt))
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt < self.settings.llm_max_retries:
                    await asyncio.sleep(min(8.0, 2**attempt))

        logger.warning(
            "%s exhausted after %d attempts: %s",
            provider, self.settings.llm_max_retries, last_error,
        )
        return None

    # ── main call ────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        system: str = SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: int = 700,
        use_cache: bool = True,
    ) -> LLMResult:
        temperature = (
            self.settings.llm_temperature if temperature is None else temperature
        )

        if self.settings.offline_llm:
            return LLMResult(text="", model="offline", degraded=True,
                             latency_ms=0.0)

        chain = self.settings.llm_provider_chain
        if not chain:
            logger.warning("no LLM provider configured; degrading")
            return LLMResult(text="", model="unconfigured", degraded=True,
                             provider="")

        # Must happen before any request is constructed, and it is done once
        # for the whole chain: the fallback must not become a hole in the
        # de-identification gate.
        self._guard({"content": prompt, "system": system})

        if self._client is None:
            async with self:
                return await self.complete(
                    prompt, system, temperature, max_tokens, use_cache
                )

        started = time.perf_counter()
        skipped: List[str] = []

        for index, (provider, base_url, model) in enumerate(chain):
            breaker = self._breakers.get(provider)
            is_primary = index == 0

            # The breaker is checked before the request, not after it.
            # Skipping the call is the entire point: a call to an exhausted
            # endpoint is a request that cannot succeed and cannot be
            # un-spent.
            if breaker is not None and not breaker.allows_request():
                skipped.append(provider)
                logger.info(
                    "circuit %s open for another %.0fs; not spending a request",
                    provider, breaker.seconds_until_probe(),
                )
                continue

            # Provider and model are both in the cache key. Different providers
            # return different text for the same prompt, so a cache entry
            # must never be replayed under another provider's name.
            cache_key = RedisCache.key_for(
                prompt, system, f"{provider}:{model}", temperature
            )
            if use_cache:
                hit = await self.cache.get(cache_key)
                if hit is not None:
                    try:
                        blob = json.loads(hit)
                        if breaker is not None:
                            breaker.record_success()
                        return LLMResult(
                            text=blob["text"],
                            model=blob.get("model", model),
                            cached=True,
                            latency_ms=0.4,
                            provider=provider,
                        )
                    except Exception:  # pragma: no cover
                        pass

            hedged = False
            try:
                if is_primary and self.settings.llm_first_byte_timeout_s > 0:
                    try:
                        result = await asyncio.wait_for(
                            self._attempt(
                                provider, base_url, model, prompt, system,
                                temperature, max_tokens, use_cache, cache_key,
                                last=index == len(chain) - 1,
                            ),
                            timeout=self.settings.llm_first_byte_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        # A stall, not a refusal. Give the turn to the next
                        # provider rather than hold a clinician for the full
                        # request timeout.
                        logger.warning(
                            "primary %s did not answer in %.0fs; handing over",
                            provider, self.settings.llm_first_byte_timeout_s,
                        )
                        hedged = True
                        result = None
                else:
                    result = await self._attempt(
                        provider, base_url, model, prompt, system, temperature,
                        max_tokens, use_cache, cache_key,
                        last=index == len(chain) - 1,
                    )
            except DoNotFallBack as exc:
                # Not an outage, so the breaker stays closed: the endpoint is
                # healthy, the request is wrong.
                logger.warning("%s; not trying another provider", exc)
                return LLMResult(
                    text="",
                    model="rejected",
                    degraded=True,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    attempts=1,
                    provider=provider,
                )

            if result is not None:
                if breaker is not None:
                    breaker.record_success()
                if not is_primary:
                    logger.warning(
                        "primary %s unavailable (%s); answered by fallback %r",
                        chain[0][0], "timed out" if hedged else "exhausted",
                        provider,
                    )
                return result

            if breaker is not None:
                breaker.record_failure()

        logger.warning(
            "every configured LLM provider failed (skipped: %s); degrading",
            ", ".join(skipped) or "none",
        )
        return LLMResult(
            text="",
            model="degraded",
            degraded=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=self.settings.llm_max_retries,
            provider="",
        )

    async def stream(self, prompt: str, system: str = SYSTEM_PROMPT):
        """Yield chunks. Used by the SSE endpoint for bounded perceived latency."""
        result = await self.complete(prompt, system)
        if result.text:
            for i in range(0, len(result.text), 48):
                yield result.text[i : i + 48]
                await asyncio.sleep(0.01)


def parse_claims(raw: str) -> List[Dict[str, Any]]:
    """Extract the claims array from a model response.

    Tolerates fenced code blocks and surrounding prose, because free-tier
    models frequently wrap JSON in markdown despite being asked not to.
    A parse failure returns an empty list, which the linker then treats as
    "nothing to cite" — a safe failure, since an uncited answer is dropped.
    """
    if not raw:
        return []
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
            candidate = candidate.strip()
            if candidate.startswith("{"):
                text = candidate
                break
    start = text.find("{")
    if start == -1:
        return []
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                text = text[start : index + 1]
                break
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    claims = data.get("claims")
    if not isinstance(claims, list):
        return []
    out: List[Dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text_value = claim.get("text")
        ids = claim.get("event_ids")
        if not isinstance(text_value, str) or not text_value.strip():
            continue
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list):
            ids = []
        out.append(
            {
                "text": text_value.strip(),
                "event_ids": [str(i) for i in ids],
            }
        )
    return out
