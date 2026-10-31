"""The vision draft reader, tested without loading any weights.

The bug guarded here was silent and total. `_parse` took a third required
parameter that no caller passed and no code read, so every transcription
raised `TypeError` *inside* a broad `except`, and was returned as an
`error` string. MedGemma could therefore never produce a reading, on any
image, ever -- while reporting itself `AVAILABLE` and burning a full
generation pass first. It looked like a model-quality problem and was a
one-argument call-signature problem.
"""

from __future__ import annotations

import inspect

import pytest

from app.core.config import Settings
from app.services.vision_llm import (
    SYSTEM,
    USER,
    _MedGemma,
    _parse,
)


def make_settings(**over) -> Settings:
    """Settings with the vision reader switched on.

    Built explicitly rather than read from the environment, so the tests do
    not inherit whatever `ENABLE_VISION_LLM` happens to be on the developer's
    machine -- the whole point of the flag is that it is off by default.
    """
    base = dict(
        enable_vision_llm=True,
        medgemma_gguf="/nonexistent/medgemma",
        medgemma_file="absent.gguf",
        offline_llm=True,
    )
    base.update(over)
    return Settings(**base)


class TestParseSignature:
    def test_parse_is_callable_with_exactly_what_transcribe_passes(self):
        """The contract `transcribe` actually uses.

        Asserted against the caller rather than against a literal, so adding
        a required parameter to `_parse` without updating `transcribe` fails
        here instead of failing 140 seconds into a real inference.
        """
        from app.services.vision_llm import _MedGemma as M

        source = inspect.getsource(M.transcribe)
        # Every _parse(...) call in transcribe must be satisfiable.
        params = list(inspect.signature(_parse).parameters.values())
        required = [
            p
            for p in params
            if p.default is inspect.Parameter.empty
            and p.kind
            in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]

        import re

        calls = re.findall(r"_parse\(([^)]*)\)", source)
        assert calls, "transcribe no longer calls _parse; update this test"

        for call in calls:
            # Count top-level commas only; default args may contain none here
            # but a nested call would.
            depth = 0
            args = 1
            for ch in call:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == "," and depth == 0:
                    args += 1
            assert args >= len(required), (
                f"_parse is called with {args} argument(s) but requires "
                f"{len(required)}: {[p.name for p in required]}"
            )

    def test_parse_has_no_unused_required_parameter(self):
        """A required parameter nothing reads and nothing passes is a trap."""
        source = inspect.getsource(_parse)
        signature = inspect.signature(_parse)
        for name, param in signature.parameters.items():
            if param.default is not inspect.Parameter.empty:
                continue
            if name in ("self",):
                continue
            # The body must actually use it.
            assert source.count(name) > 1, (
                f"_parse requires '{name}' but never uses it. Either pass it "
                f"or give it a default; a required-and-unused parameter makes "
                f"the function impossible to call correctly."
            )


class TestParseBehaviour:
    def test_reads_a_well_formed_draft(self):
        raw = (
            '{"document_type": "lab report", "patient_name": "UNCLEAR", '
            '"sample_date": "2026-02-14", '
            '"lines": [{"test": "Haemoglobin", "value": "13.4", '
            '"unit": "g/dL", "ref": "13.0 - 17.0"}]}'
        )
        result = _parse(raw, 12.5)

        assert result.readable is True
        assert result.document_type == "lab report"
        assert result.sample_date == "2026-02-14"
        assert len(result.lines) == 1
        assert result.lines[0]["test"] == "Haemoglobin"
        assert result.seconds == 12.5
        assert result.error is None

    def test_keeps_the_raw_output_verbatim(self):
        """A doctor must see what the model actually claimed.

        The tidy parse is a convenience; the raw text is the evidence. If the
        raw field were dropped, a wrong reading could not be diagnosed
        against what the model actually emitted.
        """
        raw = '{"document_type": "lab report", "lines": []}'
        assert _parse(raw, 1.0).raw == raw

    def test_unparseable_output_is_not_reported_as_readable(self):
        result = _parse("I am a language model and cannot read images.", 3.0)
        assert result.readable is False
        # And it is kept, rather than discarded, so the failure is legible.
        assert result.raw

    def test_prose_with_an_embedded_object_still_parses(self):
        """Real models wrap JSON in preamble. Being strict here would throw
        away a usable draft and report a failure the model did not make."""
        raw = (
            'Sure! Here is the reading:\n'
            '{"document_type": "lab report", "lines": [{"test": "TSH", '
            '"value": "5.31", "unit": "mIU/L"}]}\n'
            "Let me know if you need anything else."
        )
        result = _parse(raw, 9.0)
        assert result.document_type == "lab report"
        assert result.lines[0]["test"] == "TSH"

    def test_rows_exposes_only_readable_values(self):
        raw = (
            '{"lines": ['
            '{"test": "TSH", "value": "5.31", "unit": "mIU/L"}, '
            '{"test": "Blah", "value": "", "unit": ""}]}'
        )
        rows = _parse(raw, 1.0).rows
        assert len(rows) == 1
        assert rows[0]["test"] == "TSH"


class TestAvailability:
    def test_reports_unavailable_when_the_flag_is_off(self):
        assert _MedGemma().available(make_settings(enable_vision_llm=False)) is False

    def test_missing_weights_are_reported_not_raised(self, tmp_path):
        """An absent model is a deployment fact, not an exception.

        It must not mark the singleton permanently failed either: weights
        mounted after start-up have to become usable without a restart.
        """
        med = _MedGemma()
        assert med.available(make_settings(medgemma_gguf=str(tmp_path))) is False

    def test_prompts_do_not_leak_the_answer_shape(self):
        """The draft reader asks for a reading. It must not be told what a
        correction is, or it will start 'correcting' values on its own --
        which is the one thing a 4B model must never do to a medical record.
        """
        assert "supersede" not in (SYSTEM + USER).lower()
        assert "correct" not in (SYSTEM + USER).lower()
