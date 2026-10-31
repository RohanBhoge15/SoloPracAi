"""Tests for placeholder re-hydration.

The model only ever sees placeholders. What comes back must be readable:
a raw `T_MINUS_203D` in a clinical answer is worse than no answer.
"""

from __future__ import annotations

import pytest

from app.services.pipeline import AnswerPipeline, PatientContext, build_context


def ctx(display_name="Rakesh Mehta", age=54) -> PatientContext:
    return build_context(
        "p1", [], display_name=display_name, age=age
    )


@pytest.fixture
def pipeline() -> AnswerPipeline:
    return AnswerPipeline(llm=None)


def test_name_placeholder_restored(pipeline):
    out = pipeline._rehydrate("The current dose for [PT_01] is unchanged.", ctx())
    assert "Rakesh Mehta" in out
    assert "[PT_01]" not in out


def test_name_placeholder_restored_without_brackets(pipeline):
    """Models frequently drop the brackets when quoting a token."""
    out = pipeline._rehydrate("Dose for PT_01 is unchanged.", ctx())
    assert "Rakesh Mehta" in out
    assert "PT_01" not in out


def test_age_placeholder_restored(pipeline):
    out = pipeline._rehydrate("A [AGE_54] year old patient.", ctx())
    assert "54" in out
    assert "AGE_" not in out


def test_relative_token_becomes_plain_language(pipeline):
    out = pipeline._rehydrate("Recorded [T_MINUS_14D].", ctx())
    assert "T_MINUS" not in out
    assert "14 days ago" in out


def test_relative_token_without_brackets(pipeline):
    out = pipeline._rehydrate("Recorded as of T_MINUS_203D.", ctx())
    assert "T_MINUS" not in out
    assert "203 days ago" in out


def test_today_token_rendered(pipeline):
    out = pipeline._rehydrate("Recorded [T_TODAY].", ctx())
    assert "T_TODAY" not in out
    assert "today" in out.lower()


def test_long_offset_reads_better_as_approximate(pipeline):
    out = pipeline._rehydrate("From [T_MINUS_365D].", ctx())
    assert "year ago" in out


def test_no_placeholder_left_behind(pipeline):
    text = (
        "For [PT_01], age [AGE_54], the [T_MINUS_90D] result was abnormal "
        "and [T_MINUS_7D] it improved."
    )
    out = pipeline._rehydrate(text, ctx())
    for token in ("[PT_", "AGE_", "T_MINUS_", "T_TODAY"):
        assert token not in out


def test_empty_input_returns_empty(pipeline):
    assert pipeline._rehydrate("", ctx()) == ""


def test_plain_text_untouched(pipeline):
    text = "Potassium is 4.8 mEq/L, which is within the reference range."
    assert pipeline._rehydrate(text, ctx()) == text
