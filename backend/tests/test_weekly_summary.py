"""Weekly narrative must reach the LLM with no identifiers.

Regression: draft_summary built its prompt with absolute calendar dates,
so the PII gate refused EVERY report and only the extract fallback ever
shipped (seen live Oct 2026). The prompt is now scrubbed exactly like
chat prompts. Fully stubbed — no network, no quota.
"""

import asyncio
from types import SimpleNamespace

from app.documents import weekly
from app.services.pii import assert_payload_safe


class FakeLLM:
    def __init__(self):
        self.seen = None

    async def complete(self, prompt, system=None):
        self.seen = prompt
        # The real gate runs inside the client; mirror it here so the test
        # fails the same way production did.
        assert_payload_safe(prompt)
        return SimpleNamespace(text="Steady week. No changes.", degraded=False)


def test_weekly_prompt_is_gate_clean():
    record_text = (
        "CURRENT RECORDS (true as of now):\n"
        "  [abc123] 2026-10-11 lab: HbA1c = 8.9 %\n"
        "  [def456] 2026-10-04 medication: Metformin 500 mg (active)\n"
    )
    llm = FakeLLM()
    out = asyncio.run(
        weekly.draft_summary(
            llm,
            patient={"id": "pid-1", "display_name": "Ranveer Thakur", "age": 47, "phone": "+919876543210"},
            doctor={"account": {"display_name": "Dr Rohan Bhoge"}},
            period={"days": 7, "period_from": "2026-10-04T00:00:00", "period_to": "2026-10-11T00:00:00", "findings": [], "medications": [], "corrections": []},
            record_text=record_text,
            cited=["abc123", "def456"],
        )
    )
    assert out == "Steady week. No changes."
    assert "2026-10-11" not in llm.seen
    assert "Ranveer Thakur" not in llm.seen
    assert "+919876543210" not in llm.seen
