"""Provider fallback: OpenCode primary, Groq second, OpenRouter last.

All three are free tiers. The order exists for measured reasons: OpenCode
leads because it is the configured primary; Groq answers in ~0.1 s against
the primary's seconds, which is what keeps the demo responsive; OpenRouter
is last because its free allowance is a hard 50 requests/day for the whole
key and has been observed returning 429. These tests pin the three
properties that make the chain safe:

1. The primary is tried first and each fallback only on genuine exhaustion.
2. A payload blocked by the de-identification gate is never handed to any
   vendor. Three providers must not mean three chances to leak.
3. The cache key includes the provider, so an answer from one is never
   replayed under another's name.

The HTTP layer is stubbed. These assert routing and safety, not the network.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

from app.core.config import Settings
from app.services.llm import (
    SYSTEM_PROMPT,
    DeidentificationBlocked,
    LLMClient,
    LLMResult,
    RedisCache,
)


def make_settings(**over) -> Settings:
    # openrouter is threaded through `primary.example` and opencode through
    # `fallback.example` so the routing tests below read naturally; groq is
    # the third link at `third.example`. The urls are fixtures only - what
    # the chain tests assert is the order of provider names.
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
        llm_primary="openrouter",
    )
    base.update(over)
    return Settings(**base)


class StubTransport:
    """A MockTransport handler that records calls and replays scripted statuses.

    `httpx.MockTransport` invokes the handler with a single `Request`, so the
    URL, body and headers are read off the request rather than passed in.
    """

    def __init__(self, script: Dict[str, List[Tuple[int, Any]]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content) if request.content else None
        self.calls.append(
            # Headers are kept as an httpx object, not a plain dict: httpx
            # normalises names to lower case, so a dict would silently break
            # every `["Authorization"]` lookup below.
            {"url": url, "body": body, "headers": request.headers}
        )
        queue = self.script.get(url)
        if not queue:
            raise AssertionError(f"unexpected call to {url}")
        status, payload = queue.pop(0)
        if isinstance(status, Exception):
            raise status
        if status >= 400:
            return httpx.Response(
                status, json={"error": "scripted"}, request=request
            )
        return httpx.Response(status, json=payload, request=request)


def ok_body(text: str = '{"claims":[]}') -> Dict[str, Any]:
    return {"model": "served-model", "choices": [{"message": {"content": text}}]}


async def run(client: LLMClient, prompt: str = "p", **kw) -> LLMResult:
    """Drive a client whose mock transport is already installed.

    Deliberately *not* `async with client:` — entering the context manager
    assigns a real `httpx.AsyncClient` over the mock, and the test then
    performs genuine DNS lookups instead of exercising the stub.
    """
    return await client.complete(prompt, use_cache=False, **kw)


def attach(client: LLMClient, transport: StubTransport) -> None:
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(transport))


# â”€â”€ the chain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_chain_is_ordered_primary_then_fallback():
    chain = make_settings().llm_provider_chain
    assert [c[0] for c in chain] == ["openrouter", "opencode", "groq"]


def test_fallback_absent_when_no_key():
    chain = make_settings(opencode_api_key=None).llm_provider_chain
    assert [c[0] for c in chain] == ["openrouter", "groq"]


def test_fallback_absent_when_disabled():
    """The kill switch must actually remove it, not just be advisory."""
    chain = make_settings(llm_fallback_enabled=False).llm_provider_chain
    assert [c[0] for c in chain] == ["openrouter"]


def test_fallback_alone_when_primary_unset():
    """A deployment with no OpenRouter key still gets prose."""
    chain = make_settings(openrouter_api_key=None).llm_provider_chain
    assert [c[0] for c in chain] == ["opencode", "groq"]


def test_chain_without_groq():
    """Only OpenCode and OpenRouter configured: the chain is the old two."""
    chain = make_settings(groq_api_key=None).llm_provider_chain
    assert [c[0] for c in chain] == ["openrouter", "opencode"]


def test_offline_yields_no_chain():
    chain = make_settings(offline_llm=True).llm_provider_chain
    assert chain == []


def test_groq_is_the_default_primary():
    """Shipping order, asserted so a silent reversal cannot slip through.

    OpenCode was the default until its free tier began refusing all
    external calls (403 "can only be used from within OpenCode"). Leading
    with it would burn a rejection plus a breaker cycle on every fresh
    cooldown window. Groq answers in ~0.1 s and is the fastest working
    link, so it leads; OpenCode stays configured as a fallback in case
    the policy reverts.
    """
    assert Settings().llm_primary == "groq"
    chain = make_settings(llm_primary="groq").llm_provider_chain
    assert [c[0] for c in chain] == ["groq", "opencode", "openrouter"]


def test_primary_order_is_configurable_in_all_directions():
    """The swap must be a setting, not an edit buried in the chain logic."""
    assert [c[0] for c in make_settings(llm_primary="openrouter").llm_provider_chain] == [
        "openrouter",
        "opencode",
        "groq",
    ]
    assert [c[0] for c in make_settings(llm_primary="opencode").llm_provider_chain] == [
        "opencode",
        "groq",
        "openrouter",
    ]
    assert [c[0] for c in make_settings(llm_primary="groq").llm_provider_chain] == [
        "groq",
        "opencode",
        "openrouter",
    ]


def test_unknown_provider_name_raises_instead_of_leaking_a_key():
    """The header branch is exhaustive on purpose.

    A mis-set provider must raise rather than fall through to some other
    vendor's key - falling through would authenticate as whoever happens
    to be in the `else`.
    """
    with pytest.raises(ValueError, match="unknown LLM provider"):
        LLMClient._headers("opencai", make_settings())


def test_unknown_primary_name_does_not_take_chat_down():
    """A typo in an env var must not raise.

    Raising here would 500 the chat endpoint on boot of a bad config, which is
    a strictly worse outcome than attempting a provider that does work.
    """
    chain = make_settings(llm_primary="opencai").llm_provider_chain
    assert [c[0] for c in chain] == ["opencode", "groq", "openrouter"]


def test_fallback_still_runs_when_primary_is_opencode():
    """The breaker/quota argument is symmetric; only the order changed.

    With OpenCode first and OpenRouter second, a 429 from OpenCode must still
    cost exactly one request rather than being retried.
    """
    chain = make_settings(llm_primary="opencode").llm_provider_chain
    assert chain[0][2] == "fallback-model"  # opencode's model, attempted first
    assert chain[1][0] == "groq"  # second link, not the last
    assert chain[2][2] == "primary-model"  # openrouter last


# â”€â”€ routing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def test_primary_used_when_healthy():
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(200, ok_body())],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    r = await run(c)
    assert r.provider == "openrouter"
    assert r.degraded is False
    assert len(t.calls) == 1, "fallback must not be contacted on success"


async def test_fallback_used_after_primary_rate_limited():
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(429, None), (429, None)],
        "https://fallback.example/v1/chat/completions": [(200, ok_body('{"claims":[]}'))],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    r = await run(c)
    assert r.provider == "opencode"
    assert r.degraded is False
    assert r.text


async def test_fallback_used_after_primary_5xx():
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(500, None), (500, None)],
        "https://fallback.example/v1/chat/completions": [(200, ok_body())],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    r = await run(c)
    assert r.provider == "opencode"


async def test_fallback_used_after_primary_network_error():
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [
            (httpx.ConnectError("refused"), None),
            (httpx.ConnectError("refused"), None),
        ],
        "https://fallback.example/v1/chat/completions": [(200, ok_body())],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    r = await run(c)
    assert r.provider == "opencode"


async def test_degrades_when_all_three_fail():
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(429, None), (429, None)],
        "https://fallback.example/v1/chat/completions": [(429, None), (429, None)],
        "https://third.example/v1/chat/completions": [(429, None), (429, None)],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    r = await run(c)
    assert r.degraded is True
    assert r.text == ""
    assert r.provider == ""


async def test_third_provider_serves_when_first_two_are_exhausted():
    """The chain is three deep: two 429s must still end in prose."""
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(429, None), (429, None)],
        "https://fallback.example/v1/chat/completions": [(429, None), (429, None)],
        "https://third.example/v1/chat/completions": [(200, ok_body())],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    r = await run(c)
    assert r.degraded is False
    assert r.provider == "groq"
    assert r.text


async def test_client_4xx_does_not_fall_back():
    """A malformed request fails identically on both vendors.

    Falling back would multiply the latency to reach the same 400, and would
    make a request bug look like an availability problem.
    """
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(400, None)],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    r = await run(c)
    assert r.degraded is True
    assert r.model == "rejected"
    assert not any("fallback.example" in call["url"] for call in t.calls)


async def test_4xx_does_not_open_the_breaker():
    """A rejected request is not an outage, so the endpoint stays in rotation."""
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(400, None)] * 4,
        "https://fallback.example/v1/chat/completions": [(200, ok_body())] * 4,
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    await run(c, "q1")
    states = {b["provider"]: b for b in c.breaker_state()}
    assert states["openrouter"]["state"] == "closed", (
        "a 400 must not be treated as the endpoint being down"
    )


async def test_403_advances_to_next_provider():
    """A 401/403 rejects our credentials at one vendor, not the request.

    Seen live: OpenCode's free tier began returning 403 "can only be used
    from within OpenCode" on every external call. That is a dead link, and
    the chain must walk past it - unlike a 400, which fails identically
    everywhere and must not fall back.
    """
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(403, None), (403, None)],
        "https://fallback.example/v1/chat/completions": [(200, ok_body())],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    r = await run(c)
    assert r.degraded is False
    assert r.provider == "opencode"


# â”€â”€ credentials must not leak between providers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def test_each_provider_gets_only_its_own_key():
    t = StubTransport({
        "https://primary.example/v1/chat/completions": [(429, None), (429, None)],
        "https://fallback.example/v1/chat/completions": [(429, None), (429, None)],
        "https://third.example/v1/chat/completions": [(200, ok_body())],
    })
    c = LLMClient(settings=make_settings())
    attach(c, t)
    await run(c)
    by_url = {call["url"]: call["headers"] for call in t.calls}
    own_key = {
        "https://primary.example/v1/chat/completions": "sk-or-test",
        "https://fallback.example/v1/chat/completions": "oc-test",
        "https://third.example/v1/chat/completions": "gsk-test",
    }
    all_keys = set(own_key.values())
    # Each vendor's request carries its own key and no other vendor's.
    for url, headers in by_url.items():
        blob = " ".join(f"{k}: {v}" for k, v in headers.multi_items())
        assert own_key[url] in blob
        for key in all_keys - {own_key[url]}:
            assert key not in blob


async def test_openrouter_attribution_not_sent_to_opencode():
    t = StubTransport({
        "https://fallback.example/v1/chat/completions": [(200, ok_body())],
    })
    c = LLMClient(settings=make_settings(openrouter_api_key=None))
    attach(c, t)
    await run(c)
    headers = t.calls[0]["headers"]
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


# â”€â”€ the safety gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def test_blocked_payload_never_reaches_any_provider():
    """The gate is checked once for the whole chain.

    This is the test that matters most. With three vendors, a guard placed
    inside the per-provider loop would still be correct, but a guard placed
    only around the primary would send PHI to a fallback unchecked.
    """
    t = StubTransport({})  # any outbound call fails the test
    c = LLMClient(settings=make_settings())
    attach(c, t)
    with pytest.raises(DeidentificationBlocked):
        await run(c, prompt="Patient Rakesh Mehta, DOB 1972-04-11, phone 9876543210")
    assert t.calls == [], "a blocked payload must not be sent anywhere"


async def test_blocked_payload_stops_even_when_fallback_would_have_worked():
    t = StubTransport({
        "https://fallback.example/v1/chat/completions": [(200, ok_body())],
    })
    c = LLMClient(settings=make_settings(openrouter_api_key=None))
    attach(c, t)
    with pytest.raises(DeidentificationBlocked):
        await run(c, prompt="Mr Rakesh Mehta 9876543210")
    assert t.calls == []


async def test_guard_disabled_allows_traffic():
    t = StubTransport({"https://primary.example/v1/chat/completions": [(200, ok_body())]})
    c = LLMClient(settings=make_settings(enforce_deidentification=False))
    attach(c, t)
    r = await run(c, prompt="Rakesh Mehta 9876543210")
    assert r.provider == "openrouter"


# â”€â”€ cache isolation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def test_cache_key_includes_provider():
    """A cached primary answer must not be served as the fallback's."""
    a = RedisCache.key_for("p", "s", "openrouter:m1", 0.0)
    b = RedisCache.key_for("p", "s", "opencode:m2", 0.0)
    assert a != b


async def test_cache_hit_reports_the_right_provider():
    class MemCache(RedisCache):
        def __init__(self):
            super().__init__(enabled=True, client=None)
            self.store: Dict[str, str] = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, ttl):
            self.store[key] = value

    settings = make_settings(openrouter_api_key=None)
    cache = MemCache()
    c = LLMClient(settings=settings, cache=cache)
    t = StubTransport({})
    attach(c, t)

    # No primary configured, so opencode is the only provider in the chain
    # and the seeded key is the one that gets looked up. Must be the real
    # SYSTEM_PROMPT: `complete()` keys the cache on the system message it
    # actually sends, not on whatever a test guesses.
    key = RedisCache.key_for(
        "p", SYSTEM_PROMPT, "opencode:fallback-model", settings.llm_temperature
    )
    cache.store[key] = json.dumps({"text": "cached!", "model": "fallback-model"})

    r = await c.complete("p", use_cache=True)
    assert r.cached is True
    assert r.provider == "opencode"
    assert r.model == "fallback-model"
    assert t.calls == [], "a cache hit must not make a network call"


# â”€â”€ reported state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def test_no_provider_configured_degrades():
    c = LLMClient(settings=make_settings(
        openrouter_api_key=None, opencode_api_key=None, groq_api_key=None))
    r = await run(c)
    assert r.degraded is True
    assert r.model == "unconfigured"


async def test_offline_short_circuits_before_any_provider():
    t = StubTransport({})
    c = LLMClient(settings=make_settings(offline_llm=True))
    attach(c, t)
    r = await run(c)
    assert r.degraded is True and r.model == "offline"
    assert t.calls == []


async def test_as_dict_exposes_provider():
    d = LLMResult(text="x", model="m", provider="opencode").as_dict()
    assert d["provider"] == "opencode"


