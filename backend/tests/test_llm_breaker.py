"""Circuit breaker and hang-hedge for the two-provider chain.

The property that matters is not "it falls back" - that is already covered
in test_llm_fallback.py. It is that the breaker *stops calling* an exhausted
provider, because every call to an endpoint whose daily quota is spent is a
request that cannot succeed and cannot be un-spent. Without the breaker, N
questions cost 2N requests against a 50/day budget; with it, they cost N+1.

The clock is injected so cooldown expiry is tested without sleeping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import httpx
import pytest

from app.core.config import Settings
from app.services.llm import SYSTEM_PROMPT, CircuitBreaker, LLMClient, RedisCache


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_settings(**over) -> Settings:
    base = dict(
        openrouter_api_key="sk-or-test",
        llm_model="primary-model",
        llm_base_url="https://primary.example/v1",
        opencode_api_key="oc-test",
        opencode_model="fallback-model",
        opencode_base_url="https://fallback.example/v1",
        groq_api_key="gsk-test",
        groq_model="third-model",
        groq_base_url="https://third.example/v1",
        llm_max_retries=2,
        llm_cache_ttl_s=60,
        enforce_deidentification=True,
        offline_llm=False,
        llm_fallback_enabled=True,
        llm_breaker_enabled=True,
        llm_breaker_cooldown_s=900.0,
        llm_breaker_max_cooldown_s=3600.0,
        llm_first_byte_timeout_s=12.0,
        # Pinned so these tests keep asserting breaker behaviour against a
        # *known* primary. Shipping default is groq-first; see
        # test_groq_is_the_default_primary.
        llm_primary="openrouter",
    )
    base.update(over)
    return Settings(**base)


def ok(text: str = '{"claims":[]}') -> Dict[str, Any]:
    return {"model": "served", "choices": [{"message": {"content": text}}]}


class Recorder:
    """MockTransport handler that records URLs and replays scripted statuses.

    An exhausted script raises rather than repeating the last entry, so a
    test cannot accidentally pass by calling a provider more times than it
    meant to.
    """

    def __init__(self, script: Dict[str, List[Tuple[int, Any]]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: List[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        queue = self.script.get(url)
        if not queue:
            raise AssertionError(f"unexpected call to {url}")
        status, payload = queue.pop(0)
        if isinstance(status, Exception):
            raise status
        if status >= 400:
            return httpx.Response(status, json={"e": 1}, request=request)
        return httpx.Response(status, json=payload, request=request)

    def count(self, fragment: str) -> int:
        return sum(1 for u in self.calls if fragment in u)


def build(script, **over):
    rec = Recorder(script)
    client = LLMClient(settings=make_settings(**over))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    return client, rec


PRIMARY = "https://primary.example/v1/chat/completions"
FALLBACK = "https://fallback.example/v1/chat/completions"


async def call(client, prompt="p"):
    return await client.complete(prompt, use_cache=False)


# â”€â”€ breaker state machine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_starts_closed():
    b = CircuitBreaker("x", clock=FakeClock())
    assert b.state() == CircuitBreaker.CLOSED
    assert b.allows_request() is True


def test_failure_opens_it():
    b = CircuitBreaker("x", clock=FakeClock())
    b.record_failure()
    assert b.state() == CircuitBreaker.OPEN
    assert b.allows_request() is False


def test_success_closes_it():
    b = CircuitBreaker("x", clock=FakeClock())
    b.record_failure()
    b.record_success()
    assert b.state() == CircuitBreaker.CLOSED
    assert b.failures == 0


def test_cooldown_doubles_and_is_capped():
    clock = FakeClock()
    b = CircuitBreaker("x", base_cooldown=100.0, max_cooldown=400.0, clock=clock)
    b.record_failure()
    assert b.cooldown == 100.0
    b.record_failure()
    assert b.cooldown == 200.0
    b.record_failure()
    assert b.cooldown == 400.0
    b.record_failure()
    assert b.cooldown == 400.0, "must not exceed the ceiling"


def test_half_open_after_cooldown():
    clock = FakeClock()
    b = CircuitBreaker("x", base_cooldown=100.0, clock=clock)
    b.record_failure()
    clock.advance(99)
    assert b.allows_request() is False
    clock.advance(2)
    assert b.state() == CircuitBreaker.HALF_OPEN
    assert b.allows_request() is True, "one trial request must be admitted"


def test_half_open_success_closes():
    clock = FakeClock()
    b = CircuitBreaker("x", base_cooldown=100.0, clock=clock)
    b.record_failure()
    clock.advance(101)
    b.record_success()
    assert b.state() == CircuitBreaker.CLOSED


def test_half_open_failure_reopens_longer():
    clock = FakeClock()
    b = CircuitBreaker("x", base_cooldown=100.0, clock=clock)
    b.record_failure()
    clock.advance(101)
    b.record_failure()
    assert b.state() == CircuitBreaker.OPEN
    assert b.cooldown == 200.0


def test_disabled_breaker_never_opens():
    b = CircuitBreaker("x", enabled=False, clock=FakeClock())
    b.record_failure()
    assert b.allows_request() is True


# â”€â”€ the reason the breaker exists â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def test_second_request_does_not_touch_exhausted_primary():
    """The whole point: the primary is not called again while it is open.

    Without this, question two pays the same two failed primary calls as
    question one, and a 50/day budget lasts 25 questions instead of 50.
    """
    client, rec = build(
        {PRIMARY: [(429, None), (429, None)], FALLBACK: [(200, ok()), (200, ok())]}
    )
    r1 = await call(client, "q1")
    r2 = await call(client, "q2")

    assert r1.provider == "opencode" and r2.provider == "opencode"
    assert rec.count("primary.example") == 2, (
        "primary must be called exactly twice total (one exhausted turn), "
        f"not once per question; saw {rec.count('primary.example')}"
    )
    assert rec.count("fallback.example") == 2


async def test_breaker_state_is_reported():
    client, _ = build({PRIMARY: [(429, None), (429, None)], FALLBACK: [(200, ok())]})
    await call(client)
    states = {b["provider"]: b for b in client.breaker_state()}
    assert states["openrouter"]["state"] == CircuitBreaker.OPEN
    assert states["openrouter"]["failures"] == 1
    assert states["openrouter"]["retry_in_s"] > 0
    assert states["opencode"]["state"] == CircuitBreaker.CLOSED


async def test_primary_recovers_after_cooldown_and_is_preferred_again():
    clock = FakeClock()
    breakers = {
        name: CircuitBreaker(
            name, base_cooldown=100.0, max_cooldown=400.0, clock=clock
        )
        for name in ("openrouter", "opencode")
    }
    rec = Recorder({
        PRIMARY: [(429, None), (429, None), (200, ok())],
        FALLBACK: [(200, ok())],
    })
    client = LLMClient(settings=make_settings(), breakers=breakers)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec))

    assert (await call(client, "q1")).provider == "opencode"
    assert breakers["openrouter"].state() == CircuitBreaker.OPEN

    clock.advance(101)
    r = await call(client, "q2")
    assert r.provider == "openrouter", "a recovered primary must be preferred"
    assert breakers["openrouter"].state() == CircuitBreaker.CLOSED


async def test_all_providers_open_degrades_without_spending_anything():
    clock = FakeClock()
    breakers = {
        name: CircuitBreaker(name, base_cooldown=100.0, clock=clock)
        for name in ("openrouter", "opencode", "groq")
    }
    for b in breakers.values():
        b.record_failure()
    rec = Recorder({})  # any call fails the test
    client = LLMClient(settings=make_settings(), breakers=breakers)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec))

    r = await call(client)
    assert r.degraded is True and r.text == ""
    assert rec.calls == [], "an open circuit must not spend a request"


async def test_disabled_breaker_keeps_calling_primary():
    """The kill switch restores the old cost profile deliberately."""
    client, rec = build(
        {
            PRIMARY: [(429, None), (429, None)] * 2,
            FALLBACK: [(200, ok()), (200, ok())],
        },
        llm_breaker_enabled=False,
    )
    await call(client, "q1")
    await call(client, "q2")
    assert rec.count("primary.example") == 4


# â”€â”€ the hang hedge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def test_primary_timeout_hands_over_to_fallback():
    class Slow:
        def __call__(self, request):
            import time
            time.sleep(5)
            raise AssertionError("should have been cancelled")

    rec = Recorder({FALLBACK: [(200, ok())]})
    client = LLMClient(settings=make_settings(llm_first_byte_timeout_s=0.25))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec))

    r = await call(client)
    assert r.provider == "opencode"
    assert rec.count("fallback.example") == 1


async def test_timeout_opens_the_breaker():
    """A stall is a failure: the next request should not stall again."""
    rec = Recorder({FALLBACK: [(200, ok()), (200, ok())]})
    client = LLMClient(settings=make_settings(llm_first_byte_timeout_s=0.25))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec))

    await call(client, "q1")
    states = {b["provider"]: b for b in client.breaker_state()}
    assert states["openrouter"]["state"] == CircuitBreaker.OPEN

    await call(client, "q2")
    assert rec.count("fallback.example") == 2


async def test_hedge_disabled_when_zero():
    rec = Recorder({PRIMARY: [(200, ok())], FALLBACK: [(200, ok())]})
    client = LLMClient(settings=make_settings(llm_first_byte_timeout_s=0))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    r = await call(client)
    assert r.provider == "openrouter"


async def test_fast_primary_is_never_hedged_away():
    rec = Recorder({PRIMARY: [(200, ok())], FALLBACK: []})
    client = LLMClient(settings=make_settings(llm_first_byte_timeout_s=12.0))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    r = await call(client)
    assert r.provider == "openrouter"
    assert rec.count("fallback.example") == 0


# â”€â”€ cache still beats an open circuit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def test_cache_hit_served_even_when_primary_open():
    """A cached answer must not be withheld because a breaker is open.

    The breaker exists to save requests, not to withhold a result we already
    paid for.
    """

    class MemCache(RedisCache):
        def __init__(self):
            super().__init__(enabled=True, client=None)
            self.store: Dict[str, str] = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, ttl):
            self.store[key] = value

    cache = MemCache()
    breakers = {
        n: CircuitBreaker(n, base_cooldown=100.0, clock=FakeClock())
        for n in ("openrouter", "opencode")
    }
    breakers["openrouter"].record_failure()   # already exhausted

    from app.services.llm import RedisCache as RC
    key = RC.key_for(
        "q", SYSTEM_PROMPT, "opencode:fallback-model",
        make_settings().llm_temperature,
    )
    cache.store[key] = '{"text":"cached answer","model":"fallback-model"}'

    rec = Recorder({})
    client = LLMClient(settings=make_settings(), cache=cache, breakers=breakers)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec))

    r = await client.complete("q", use_cache=True)
    assert r.cached is True and r.text == "cached answer"
    assert rec.calls == []

