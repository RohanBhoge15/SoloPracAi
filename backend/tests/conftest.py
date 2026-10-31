"""
Shared test fixtures.

The one thing worth understanding here is why the engine is disposed
between tests.

`app/db/session.py` caches the async engine in a module global, which is
right for the application - one pool, reused across requests. It is wrong
for a test process: pytest-asyncio gives every test a fresh event loop, and
a pooled asyncpg connection is bound to the loop that created it. Handing
that connection to a different loop raises "Event loop is closed", or more
insidiously, "AttributeError: 'NoneType' object has no attribute 'send'"
from deep inside the transport.

The symptom is a test suite that passes one test at a time and fails when
run as a file. That is the worst failure mode to hand someone, because it
looks like a product bug. Disposing the pool after each test makes the
lifecycle explicit and the suite order-independent.

Every test in this directory talks to a real PostgreSQL instance, not a
sqlite file. The models use PostgreSQL-specific features - the validity
views, the self-referential ON DELETE RESTRICT constraint, JSONB payload
columns - so a different backend would be testing something else.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db import session as db_session
from app.main import app


@pytest_asyncio.fixture
async def engine_disposed():
    """Guarantee no pooled connection outlives the test that opened it."""
    yield
    # The loop pytest is about to tear down is still alive here, which is
    # exactly what dispose() needs in order to close sockets politely
    # rather than leave the proactor transport half-written.
    engine = db_session._engine
    if engine is not None:
        await engine.dispose()


@pytest_asyncio.fixture
async def client(engine_disposed):
    """An HTTP client wired straight to the ASGI app.

    No network, no running server: the request never leaves the process, so
    these tests test the application and not the deployment around it. The
    live-stack scripts under `solum/_e2e/` cover the other half.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
