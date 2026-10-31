"""Async database session management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base

_settings = get_settings()

#: Created lazily. Importing a pure function (a parser, the validity core)
#: must not require a database driver to be installed, and must not open a
#: connection pool as a side effect of import.
_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            _settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False, autoflush=False
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Standalone use outside the request cycle."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_models() -> None:
    """Create tables and the derived-validity views.

    Used for local development and tests. Production uses Alembic.

    The self-referencing constraint is re-applied rather than only declared
    on the model: an existing database keeps the original ON DELETE RESTRICT,
    which makes deleting a patient with corrections fail.
    """
    from sqlalchemy import text

    from app.db.models import AS_OF_VIEW_SQL, SUPERSESSION_FK_SQL, VALIDITY_VIEW_SQL

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for statement in SUPERSESSION_FK_SQL:
            await conn.execute(text(statement))
        await conn.execute(text(VALIDITY_VIEW_SQL))
        await conn.execute(text(AS_OF_VIEW_SQL))
