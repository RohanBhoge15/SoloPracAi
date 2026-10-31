"""Tests for the generation arm's scoring.

The scorer is the one piece of this experiment that can be wrong *quietly*.
A retrieval metric computed from ranks cannot silently invert; a regex that
matches "9.4" inside "9.45" will happily report the model as correct on every
trial and produce a clean, publishable, meaningless number. So the boundaries
are pinned here explicitly.
"""

from __future__ import annotations

from math import comb

import pytest

from tapqa.generation import (
    ArmOutcome,
    _mentions,
    _mcnemar_exact,
    _rate,
    _summarise,
)


class TestMentions:
    @pytest.mark.parametrize(
        "answer,value,expected",
        [
            # The positive cases.
            ("HbA1c is 9.4%", "9.4", True),
            ("her TSH was 5.31 mIU/L", "5.31", True),
            ("TSH: 5.31", 5.31, True),
            ("platelet count 2.4 lakh", "2.4", True),
            ("a 0.5 reading", "0.5", True),
            ("she takes metformin 500mg", "Metformin", True),
            # Decimal formatting of an integral value. This is the case the
            # previous textual matcher got wrong, and it broke in both
            # directions at once: a correct answer scored as a miss, and a
            # quoted retracted value scored as absent, so genuine staleness
            # was reported as clean.
            ("52.0 mg/dL", "52.0", True),
            ("52.0 mg/dL", "52", True),
            ("52 mg/dL", "52.0", True),
            ("46.0 mg/dL", 46.0, True),
            ("2.0 mg/dL.", 2.0, True),
            ("6.09 lakhs.", "6.09", True),
            # Genuinely different numbers must not match.
            ("HbA1c is 9.45%", "9.4", False),
            ("HbA1c is 94%", "9.4", False),
            ("HbA1c is 7.25", "7.2", False),
            ("count 1165", "165", False),
            ("81.0 mg/dL", "52.0", False),
            ("52.1 mg/dL", "52.0", False),
            # Thousands separators.
            ("the value is 1,234", "1234", True),
            # Free text boundaries.
            ("she takes MetFORMINOL", "Metformin", False),
            # Nothing to match.
            ("HbA1c is 9.4%", None, False),
            ("HbA1c is 9.4%", "", False),
            ("HbA1c is 9.4%", "   ", False),
            ("no numbers at all here", "9.4", False),
        ],
    )
    def test_boundaries(self, answer, value, expected):
        assert _mentions(answer, value) is expected

    def test_a_value_written_with_a_unit_suffix_still_matches(self):
        assert _mentions("HbA1c 9.4 %", "9.4") is True

    def test_a_retracted_value_in_decimal_form_is_detected(self):
        """The regression that mattered, stated directly.

        A model quoting the retracted value in decimal form must be scored
        stale. When this returned False the arm reported 0% staleness while
        the underlying behaviour was present.
        """
        assert _mentions("TSH: 6.61 mIU/L", "6.61") is True
        assert _mentions("TSH: 52.0 mg/dL", "52.0") is True


class TestMcNemar:
    def test_balanced_discordance_is_not_significant(self):
        assert _mcnemar_exact(3, 3)["p"] == 1.0

    def test_is_two_sided(self):
        """A one-sided test would halve the p-value and overstate the result.

        The direction of the effect was not predicted in advance, so the
        factor of two is not optional.
        """
        one_sided = sum(comb(3, i) for i in range(1)) / 8
        assert _mcnemar_exact(3, 0)["p"] == pytest.approx(min(1.0, 2 * one_sided))

    def test_uniform_discordance_is_significant(self):
        assert _mcnemar_exact(10, 0)["p"] < 0.01

    def test_no_discordance_reports_no_p_rather_than_one(self):
        """`p = None` and `p = 1.0` mean different things.

        One is "the arms never differed"; the other is "they differed and the
        difference was noise". Reporting 1.0 for zero discordant pairs lets a
        completely uninformative run be quoted as a negative result.
        """
        result = _mcnemar_exact(0, 0)
        assert result["p"] is None
        assert result["discordant"] == 0
        assert "never differed" in result["note"]

    def test_never_exceeds_one(self):
        for b in range(0, 8):
            for c in range(0, 8):
                p = _mcnemar_exact(b, c)["p"]
                # None is the honest answer at zero discordance; every other
                # cell must be a real probability.
                assert p is None or 0.0 <= p <= 1.0


class TestRate:
    def test_none_values_are_excluded_not_counted_as_false(self):
        """A trial with no verdict must not be scored as a clean answer."""
        assert _rate([True, None, False]) == pytest.approx(0.5)

    def test_all_none_is_none_not_zero(self):
        assert _rate([None, None]) is None

    def test_empty_is_none(self):
        assert _rate([]) is None


def _outcome(**kw) -> ArmOutcome:
    base = dict(
        query_id="q",
        kind="corrected",
        question="current tsh",
        ours_answer="TSH 5.31",
        baseline_answer="TSH 6.61",
        ours_stale=False,
        baseline_stale=True,
        ours_correct=True,
        baseline_correct=False,
    )
    base.update(kw)
    return ArmOutcome(**base)


class TestSummarise:
    def test_an_errored_trial_is_excluded_from_the_pair(self):
        report = _summarise([_outcome(), _outcome(error="boom")], k_context=5)
        assert report["corrected"]["trials"] == 2
        assert report["corrected"]["paired"] == 1
        assert len(report["errors"]) == 1

    def test_unchanged_control_pairs_without_a_staleness_verdict(self):
        """Staleness is undefined where nothing was corrected.

        A staleness-based definition of `paired` would score every control
        trial unpaired and silently delete the specificity check.
        """
        control = _outcome(
            kind="unchanged",
            ours_stale=None,
            baseline_stale=None,
            ours_correct=True,
            baseline_correct=True,
        )
        report = _summarise([control], k_context=5)
        assert report["unchanged_control"]["paired"] == 1
        assert report["unchanged_control"]["ours_accuracy"] == 1.0

    def test_records_the_absence_of_annotation(self):
        """The report must be self-describing about its own main variable."""
        assert _summarise([], k_context=5)["annotated"] is False
