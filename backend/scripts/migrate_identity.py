"""Identity schema bring-up for an existing database.

The Patient table grew columns (user_id, doctor_id, email, ...) after the
first version shipped. `create_all` will not add a column to a table that
already exists, so this issues the ALTERs explicitly. Idempotent.
"""

from __future__ import annotations

import asyncio
import sys

# (table, column, type)
PATIENT_COLUMNS = [
    ("user_id", "UUID"),
    ("doctor_id", "UUID"),
    ("email", "VARCHAR(320)"),
    ("photo_path", "TEXT"),
    ("origin", "VARCHAR(24) NOT NULL DEFAULT 'doctor_created'"),
    ("identity_verified", "BOOLEAN NOT NULL DEFAULT FALSE"),
]

INDEXES = [
    (
        "ix_patients_email_lower",
        "CREATE INDEX IF NOT EXISTS ix_patients_email_lower "
        "ON patients (LOWER(email))",
    ),
    (
        "ix_patients_doctor_account",
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_patients_doctor_account "
        "ON patients (doctor_id, user_id) WHERE user_id IS NOT NULL",
    ),
]

FOREIGN_KEYS = [
    (
        "fk_patients_user",
        "ALTER TABLE patients ADD CONSTRAINT fk_patients_user "
        "FOREIGN KEY (user_id) REFERENCES accounts(id) ON DELETE SET NULL",
    ),
    (
        "fk_patients_doctor",
        "ALTER TABLE patients ADD CONSTRAINT fk_patients_doctor "
        "FOREIGN KEY (doctor_id) REFERENCES doctors(id) ON DELETE CASCADE",
    ),
]


async def migrate() -> int:
    from sqlalchemy import text

    # Both modules must be imported: create_all only sees tables whose
    # models have actually been loaded onto Base.metadata.
    from app.db import identity as _identity  # noqa: F401
    from app.db.models import Base
    from app.db.session import get_engine

    engine = get_engine()
    try:
        async with engine.begin() as conn:
            # accounts and doctors must exist before patients can
            # reference them.
            await conn.run_sync(Base.metadata.create_all)
            print(f"  tables: {sorted(Base.metadata.tables)}")

            # Order matters: columns must exist before a constraint can
            # reference them, and constraints before the index that
            # depends on them.
            for column, ddl in PATIENT_COLUMNS:
                await conn.execute(
                    text(
                        f"ALTER TABLE patients ADD COLUMN IF NOT EXISTS "
                        f"{column} {ddl}"
                    )
                )
                print(f"  + patients.{column}")

            for name, ddl in FOREIGN_KEYS:
                exists = await conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.table_constraints "
                        "WHERE constraint_name = :n"
                    ),
                    {"n": name},
                )
                if exists.first() is None:
                    await conn.execute(text(ddl))
                    print(f"  + constraint {name}")

            for name, ddl in INDEXES:
                await conn.execute(text(ddl))
                print(f"  + index {name}")

        print("identity schema ready")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"migration failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(migrate()))
