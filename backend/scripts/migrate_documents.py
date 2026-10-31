"""Bring the documents table up to the unified schema.

An earlier version had a `documents` table for uploaded scans only
(filename, storage_path, ocr_confidence, committed). It is now the single
table for every document a doctor produces or a patient uploads, with
more columns. `create_all` will not add columns to an existing table, so
they are issued explicitly. Idempotent.
"""

from __future__ import annotations

import asyncio
import sys

COLUMNS = [
    ("patient_id", "UUID"),
    ("doctor_id", "UUID"),
    ("kind", "VARCHAR(24) NOT NULL DEFAULT 'scan'"),
    ("title", "VARCHAR(200)"),
    ("caption", "TEXT"),
    ("description", "TEXT"),
    ("object_key", "TEXT"),
    ("content_type", "VARCHAR(120)"),
    ("size_bytes", "INTEGER"),
    ("content", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
    ("extracted", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
    ("issued_to_patient", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("issued_at", "TIMESTAMPTZ"),
    ("read_at", "TIMESTAMPTZ"),
    ("supersedes_id", "UUID"),
    ("superseded_by_id", "UUID"),
    ("source_event_id", "UUID"),
    ("version", "INTEGER NOT NULL DEFAULT 1"),
    # kept from the scan-only schema
    ("filename", "VARCHAR(300)"),
    ("raw_text", "TEXT"),
]

#: The old storage_path column is superseded by object_key.
RENAMES = [("storage_path", "object_key")]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_documents_patient_created "
    "ON documents (patient_id, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_documents_doctor ON documents (doctor_id)",
    "CREATE INDEX IF NOT EXISTS ix_documents_kind ON documents (kind)",
]

#: The scan-only schema made `filename` and `title` NOT NULL. A generated
#: prescription has no source filename, so both must be nullable now.
RELAX_NOT_NULL = [
    "ALTER TABLE documents ALTER COLUMN filename DROP NOT NULL",
    "ALTER TABLE documents ALTER COLUMN title DROP NOT NULL",
    "ALTER TABLE documents ALTER COLUMN patient_id DROP NOT NULL",
]

FOREIGN_KEYS = [
    (
        "fk_documents_patient",
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS fk_documents_patient",
    ),
    (
        "fk_documents_patient",
        "ALTER TABLE documents ADD CONSTRAINT fk_documents_patient FOREIGN KEY "
        "(patient_id) REFERENCES patients(id) ON DELETE CASCADE",
    ),
    (
        "fk_documents_doctor",
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS fk_documents_doctor",
    ),
    (
        "fk_documents_doctor",
        "ALTER TABLE documents ADD CONSTRAINT fk_documents_doctor FOREIGN KEY "
        "(doctor_id) REFERENCES doctors(id) ON DELETE SET NULL",
    ),
]


async def migrate() -> int:
    from sqlalchemy import text

    from app.db import documents as _documents  # noqa: F401
    from app.db import identity as _identity  # noqa: F401
    from app.db.models import Base
    from app.db.session import get_engine

    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            for old, new in RENAMES:
                exists = await conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name='documents' AND column_name=:n"
                    ),
                    {"n": old},
                )
                if exists.first() is not None:
                    new_exists = await conn.execute(
                        text(
                            "SELECT 1 FROM information_schema.columns "
                            "WHERE table_name='documents' AND column_name=:n"
                        ),
                        {"n": new},
                    )
                    if new_exists.first() is None:
                        await conn.execute(
                            text(f"ALTER TABLE documents RENAME {old} TO {new}")
                        )
                        print(f"  renamed {old} -> {new}")

            for column, ddl in COLUMNS:
                await conn.execute(
                    text(
                        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS "
                        f"{column} {ddl}"
                    )
                )
                print(f"  + documents.{column}")

            for name, ddl in FOREIGN_KEYS:
                if ddl.startswith("ALTER TABLE documents DROP"):
                    await conn.execute(text(ddl))
                else:
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

            for ddl in RELAX_NOT_NULL:
                try:
                    await conn.execute(text(ddl))
                except Exception:  # noqa: BLE001
                    # Already nullable, or the column does not exist yet.
                    pass

            for ddl in INDEXES:
                await conn.execute(text(ddl))

        print("documents schema ready")
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
