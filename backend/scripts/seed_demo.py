"""Seed a demonstration patient with a genuine supersession history.

Every record here is a real clinical shape — an amended potassium, a
stopped medication, a revised diagnosis — not placeholder text. The
point is that the corrections are *authentic to how records actually go
wrong*, so a demo shows the behaviour that matters.

All names are fictional. Nothing here is real patient data.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

SEED = 42


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def seed(patient_count: int = 1, verbose: bool = True) -> int:
    from sqlalchemy import select

    from app.core.validity import Event as CoreEvent
    from app.core.validity import EventType, SourceType
    from app.core.validity import content_hash
    from app.db.models import Event, Patient
    from app.db.session import get_engine, get_session_factory

    rng = random.Random(SEED)
    created_patients = 0
    engine = get_engine()

    try:
        async with get_session_factory()() as session:
            existing = (
                await session.execute(select(Patient).limit(1))
            ).scalars().all()
            if existing:
                print("database already seeded; skipping")
                return 0

            for index in range(patient_count):
                patient = Patient(
                    display_name="Rakesh Mehta" if index == 0 else f"Demo Patient {index}",
                    age=54 if index == 0 else rng.randint(28, 76),
                    phone="9876543210" if index == 0 else f"98{rng.randint(10**7, 10**8 - 1)}",
                    email=(
                        "rakesh.mehta@example.com"
                        if index == 0
                        else f"patient{index}@example.com"
                    ),
                    address="42 MG Road, Pune" if index == 0 else "Pune",
                )
                session.add(patient)
                await session.flush()
                created_patients += 1

                base = _now() - timedelta(days=400)
                rows: list[CoreEvent] = []

                def add(
                    etype: EventType,
                    payload: dict,
                    days_ago: float,
                    source: SourceType = SourceType.MANUAL,
                    target: str | None = None,
                ) -> str:
                    # Real UUIDs, matching the Postgres UUID column type.
                    eid = str(uuid.uuid4())
                    rows.append(
                        CoreEvent(
                            event_id=eid,
                            patient_id=str(patient.id),
                            event_type=etype,
                            valid_time=base + timedelta(days=days_ago),
                            payload=payload,
                            source_type=source,
                            supersedes_id=target,
                        )
                    )
                    return eid

                # ── the amended potassium: the core demo scenario ──────
                k_original = add(
                    EventType.LAB,
                    {
                        "test": "Potassium",
                        "value": 2.1,
                        "unit": "mEq/L",
                        "ref_low": 3.5,
                        "ref_high": 5.1,
                        "flag": "low",
                        "lab": "City Hospital Laboratory",
                    },
                    days_ago=200,
                )
                add(
                    EventType.LAB,
                    {
                        "test": "Potassium",
                        "value": 4.8,
                        "unit": "mEq/L",
                        "ref_low": 3.5,
                        "ref_high": 5.1,
                        "flag": "normal",
                        "lab": "City Hospital Laboratory",
                        "amendment": "Specimen re-run; original transcription error",
                    },
                    days_ago=197,
                    source=SourceType.CORRECTION,
                    target=k_original,
                )

                # ── metformin: started, titrated up, then stopped ──────
                # Days-ago values decrease, so the order is chronological.
                add(
                    EventType.MEDICATION,
                    {
                        "drug": "Metformin",
                        "dose": "500 mg",
                        "frequency": "twice daily",
                        "status": "stopped",
                        "reason": "superseded by dose titration",
                    },
                    days_ago=320,
                )
                add(
                    EventType.MEDICATION,
                    {
                        "drug": "Metformin",
                        "dose": "1000 mg",
                        "frequency": "twice daily",
                        "status": "stopped",
                        "reason": "eGFR 24, switched to insulin",
                    },
                    days_ago=45,
                )

                # ── telmisartan: ongoing ───────────────────────────────
                add(
                    EventType.MEDICATION,
                    {
                        "drug": "Telmisartan",
                        "dose": "40 mg",
                        "frequency": "once daily",
                        "status": "active",
                    },
                    days_ago=280,
                )

                # ── revised diagnosis ─────────────────────────────────
                dx_original = add(
                    EventType.DIAGNOSIS,
                    {"diagnosis": "Type 2 Diabetes Mellitus", "icd10": "E11.9"},
                    days_ago=350,
                )
                add(
                    EventType.DIAGNOSIS,
                    {
                        "diagnosis": "Type 2 Diabetes Mellitus with chronic kidney disease",
                        "icd10": "E11.22",
                        "revised_from": "E11.9",
                    },
                    days_ago=90,
                    source=SourceType.CORRECTION,
                    target=dx_original,
                )

                # ── an amended HbA1c ───────────────────────────────────
                hba1c_original = add(
                    EventType.LAB,
                    {
                        "test": "HbA1c",
                        "value": 9.4,
                        "unit": "%",
                        "ref_low": 4.0,
                        "ref_high": 5.6,
                        "flag": "high",
                    },
                    days_ago=150,
                )
                add(
                    EventType.LAB,
                    {
                        "test": "HbA1c",
                        "value": 7.2,
                        "unit": "%",
                        "ref_low": 4.0,
                        "ref_high": 5.6,
                        "flag": "high",
                        "amendment": "Assay recalibrated; prior run not valid",
                    },
                    days_ago=147,
                    source=SourceType.CORRECTION,
                    target=hba1c_original,
                )

                # ── a distractor correction (not queried often) ───────
                creat_original = add(
                    EventType.LAB,
                    {
                        "test": "Creatinine",
                        "value": 1.4,
                        "unit": "mg/dL",
                        "ref_low": 0.6,
                        "ref_high": 1.3,
                        "flag": "high",
                    },
                    days_ago=100,
                )
                add(
                    EventType.LAB,
                    {
                        "test": "Creatinine",
                        "value": 1.1,
                        "unit": "mg/dL",
                        "ref_low": 0.6,
                        "ref_high": 1.3,
                        "flag": "normal",
                    },
                    days_ago=98,
                    source=SourceType.CORRECTION,
                    target=creat_original,
                )

                # ── unchanged records: no correction, must stay clean ──
                add(
                    EventType.LAB,
                    {
                        "test": "Sodium",
                        "value": 140,
                        "unit": "mEq/L",
                        "ref_low": 135,
                        "ref_high": 145,
                        "flag": "normal",
                    },
                    days_ago=100,
                )
                add(
                    EventType.VITAL,
                    {"measurement": "Blood pressure", "value": "128/82", "unit": "mmHg"},
                    days_ago=30,
                )
                add(
                    EventType.VITAL,
                    {"measurement": "Weight", "value": 74, "unit": "kg"},
                    days_ago=30,
                )
                add(
                    EventType.NOTE,
                    {
                        "note": "Reports occasional dizziness on standing. "
                        "No syncope. Advised hydration.",
                        "author": "Dr. Anita Desai",
                    },
                    days_ago=30,
                )
                add(
                    EventType.NOTE,
                    {
                        "note": "Renal function declining. Started insulin. "
                        "Metformin discontinued.",
                        "author": "Dr. Anita Desai",
                    },
                    days_ago=45,
                )
                add(
                    EventType.IMAGING,
                    {
                        "study": "Chest X-ray",
                        "findings": "Mild cardiomegaly. No focal consolidation.",
                        "impression": "Cardiomegaly",
                    },
                    days_ago=60,
                )

                for core in rows:
                    row = Event.from_core(core)
                    row.content_hash = content_hash(core)
                    session.add(row)

                await session.flush()
                if verbose:
                    print(f"seeded {patient.display_name}: {len(rows)} events")

            await session.commit()
    except Exception as exc:
        print(f"seed failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await engine.dispose()

    print(f"done: {created_patients} patient(s)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(seed(args.patients)))
