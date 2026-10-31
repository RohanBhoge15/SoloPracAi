"""Drug-name suggestions for the prescription composer.

Reference data, not patient data: the FDA NDC directory is US federal
public domain, so there is nothing to de-identify and nothing per-patient
to scope. The route still requires a signed-in doctor, because the only
caller is the prescription composer and an unauthenticated suggestion box
is a bot faucet for no benefit.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.routes.auth import current_doctor
from app.services.drugs import DEFAULT_LIMIT, MAX_LIMIT, MIN_QUERY_CHARS, count, suggest

router = APIRouter(prefix="/drugs", tags=["drugs"])


class DrugSuggestOut(BaseModel):
    query: str
    suggestions: List[str]
    total_names: int


@router.get("/suggest", response_model=DrugSuggestOut)
async def suggest_drugs(
    q: str = Query(min_length=MIN_QUERY_CHARS, max_length=60),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    _doctor=Depends(current_doctor),
):
    """Up to `limit` drug names starting with (or containing a word
    starting with) `q`, fastest matches first."""
    return DrugSuggestOut(
        query=q.strip(), suggestions=suggest(q, limit), total_names=count()
    )
