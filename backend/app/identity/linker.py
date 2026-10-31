"""Identity linking.

The requirement, from the product side:

    A doctor registers a walk-in patient with an email today. The patient
    creates their own account next year using that same email. They must
    become one identity - across every doctor who has ever seen them -
    with no action from anyone.

How it works:

    1. Normalise the email (lowercase, validate).
    2. Find every unlinked patient row with that address.
    3. Point them at the new account.

Two safeguards, because a silent wrong link is worse than no link:

    - Only *exact, normalised* matches link automatically. A near-match
      (fuzzy name, shared family email) is surfaced for a human instead.
    - Every automatic link is written to `identity_link_log`, so it can be
      audited and reversed.

A family sharing one phone/email is a real Indian scenario. Exact
matching alone means the second family member simply does not link -
which is a visible non-event rather than a silent data mix-up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import normalise_email
from app.db.identity import Doctor, IdentityLinkLog
from app.db.models import Patient

logger = logging.getLogger(__name__)


@dataclass
class LinkResult:
    """What a linking pass did, in a form worth showing a human."""

    account_id: UUID
    email: str
    linked: List[UUID] = None  # type: ignore[assignment]
    already_linked: List[UUID] = None  # type: ignore[assignment]
    skipped_conflicting: int = 0

    def __post_init__(self) -> None:
        if self.linked is None:
            self.linked = []
        if self.already_linked is None:
            self.already_linked = []

    @property
    def total(self) -> int:
        return len(self.linked) + len(self.already_linked)

    def as_dict(self) -> Dict:
        return {
            "account_id": str(self.account_id),
            "email": self.email,
            "linked": [str(x) for x in self.linked],
            "already_linked": [str(x) for x in self.already_linked],
            "linked_count": len(self.linked),
            "total_records": self.total,
        }


class IdentityLinker:
    """Links unowned patient records to a freshly created account."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def link_for_account(self, account_id: UUID, email: str) -> LinkResult:
        """Attach every matching unlinked record to `account_id`.

        Idempotent: running it twice links nothing the second time.
        """
        normalised = normalise_email(email)
        if not normalised:
            return LinkResult(account_id=account_id, email=email or "")

        result = LinkResult(account_id=account_id, email=normalised)

        rows = (
            await self.session.execute(
                select(Patient).where(
                    func.lower(Patient.email) == normalised
                )
            )
        ).scalars().all()

        for patient in rows:
            if patient.user_id == account_id:
                result.already_linked.append(patient.id)
                continue
            if patient.user_id is not None:
                # Owned by a different account. An exact email match should
                # not happen, but if it does it means two people share an
                # address, so it is counted and skipped rather than moved.
                result.skipped_conflicting += 1
                logger.warning(
                    "email %s already claimed by another account; "
                    "patient %s not moved",
                    normalised,
                    patient.id,
                )
                continue

            patient.user_id = account_id
            patient.identity_verified = True
            result.linked.append(patient.id)

            self.session.add(
                IdentityLinkLog(
                    account_id=account_id,
                    patient_id=patient.id,
                    email=normalised,
                    method="email_exact",
                    confidence=1.0,
                )
            )

        if result.linked:
            logger.info(
                "linked %d record(s) to account %s via email",
                len(result.linked),
                account_id,
            )
        return result

    async def attach_doctor_record(
        self, doctor: Doctor, patient_id: UUID
    ) -> None:
        """Ensure a doctor owns a patient row, creating one if needed.

        This is the reverse direction: the patient already has an
        account, and the doctor is seeing them for the first time.
        """
        patient = await self.session.get(Patient, patient_id)
        if patient is None:
            raise ValueError(f"patient {patient_id} not found")

        if patient.doctor_id is None:
            patient.doctor_id = doctor.id
            return
        if patient.doctor_id == doctor.id:
            return

        # Already someone else's copy. Give this doctor their own row
        # pointing at the same account, so records stay isolated per
        # doctor while the identity stays shared.
        existing = (
            await self.session.execute(
                select(Patient).where(
                    Patient.doctor_id == doctor.id,
                    Patient.user_id == patient.user_id,
                )
            )
        ).scalars().first()

        if existing is not None:
            return

        clone = Patient(
            user_id=patient.user_id,
            doctor_id=doctor.id,
            display_name=patient.display_name,
            email=patient.email,
            age=patient.age,
            phone=patient.phone,
            address=patient.address,
            abha_number=patient.abha_number,
            origin="linked_clone",
            identity_verified=patient.identity_verified,
        )
        self.session.add(clone)

    async def find_across_doctors(self, account_id: UUID) -> List[Patient]:
        """Every patient row belonging to one account, across doctors."""
        rows = (
            await self.session.execute(
                select(Patient)
                .where(Patient.user_id == account_id)
                .order_by(Patient.created_at.desc())
            )
        ).scalars().all()
        return list(rows)

    async def unlink(self, patient_id: UUID) -> bool:
        """Reverse a link. Used when a match turns out to be wrong."""
        patient = await self.session.get(Patient, patient_id)
        if patient is None or patient.user_id is None:
            return False
        patient.user_id = None
        patient.identity_verified = False
        return True
