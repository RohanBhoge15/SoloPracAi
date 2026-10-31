"""Create tables and derived-validity views.

Idempotent. Safe to run repeatedly; used by the demo bootstrap and by CI.
"""

from __future__ import annotations

import asyncio
import sys


async def main() -> int:
    from sqlalchemy import text

    from app.db.models import AS_OF_VIEW_SQL, VALIDITY_VIEW_SQL
    from app.db.session import get_engine
    from app.db.models import Base

    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text(VALIDITY_VIEW_SQL))
            await conn.execute(text(AS_OF_VIEW_SQL))
            print("schema created")
    except Exception as exc:
        print(f"schema creation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
