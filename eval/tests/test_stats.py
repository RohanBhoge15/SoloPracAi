"""Tests for the paired significance tests.

The bug these exist to prevent: `mcnemar_exact` passed `min(b, c)` to the
binomial tail where the two-sided p-value needs `max(b, c)`. With b=97 and
c=3 that computes P(X >= 3) ~ 1.0, so the test returned exactly p = 1.0
for a result that should be indistinguishable from zero. Nothing raised,
the CIs were fine, and every one of the eight comparisons in the first
full TAP-QA run came back as 1.0000 - which is what gave it away. A real
null does not land on the same number eight times running.

A test that returns the maximum possible p-value for a lopsided result is
worse than no test, because it reads as "no difference found".
"""

from __future__ import annotations

import pytest

from tapqa.stats import (
    _binomial_tail_at_least,
    mcnemar_exact,
    paired_bootstrap,
    wilson,
)


class TestBinomialTail:
    def test_at_least_is_a_real_upper_tail(self):
        # P(X >= 0) is 1 by definition, whatever n is.
        assert _binomial_tail_at_least(0, 10, 0.5) == 1.0
        # P(X >= n) is one outcome over 2**n.
        assert _binomial_tail_at_least(10, 10, 0.5) == 1.0 / 1024
        # P(X >= 5) for n=10 is 0.5 + 0.5 * P(X=5) = 0.5 + 0.5 * 252/1024
        assert abs(_binomial_tail_at_least(5, 10, 0.5) - 0.623046875) < 1e-12

    def test_tail_is_monotone_decreasing(self):
        values = [_binomial_tail_at_least(k, 20, 0.5) for k in range(21)]
        assert all(a >= b for a, b in zip(values, values[1:]))


class TestMcNemar:
    def test_lopsided_result_is_significant(self):
        """The regression. 97 discordant pairs, all favouring method a."""
        a = [True] * 97 + [False] * 3
        b = [False] * 97 + [True] * 3
        result = mcnemar_exact(a, b)
        assert result["b"] == 97
        assert result["c"] == 3
        assert result["p_value"] < 1e-20
        assert result["significant"] is True

    def test_the_pre_fix_direction_would_have_been_exactly_one(self):
        """Guard the specific arithmetic error, not just the outcome.

        P(X >= min) rather than P(X >= max) is what produced 1.0. If this
        assertion ever fails, the tail is being read the wrong way round
        again and the symptom will be a flat column of 1.0000.
        """
        n, smaller = 100, 3
        wrong = min(1.0, 2 * _binomial_tail_at_least(smaller, n, 0.5))
        assert wrong == 1.0, (
            "if this is no longer 1.0 the guard below is no longer testing "
            "the historical bug - update it deliberately"
        )
        right = min(1.0, 2 * _binomial_tail_at_least(max(97, 3), n, 0.5))
        assert right < 1e-20

    def test_perfect_agreement_is_not_significant(self):
        shared = [True] * 50 + [False] * 50
        result = mcnemar_exact(shared, list(shared))
        assert result["p_value"] == 1.0
        assert result["significant"] is False
        assert result.get("note") == "no discordant pairs"

    def test_balanced_discordance_is_not_significant(self):
        # 5 each way: the textbook "no evidence" case.
        a = [True] * 5 + [False] * 5
        b = [False] * 5 + [True] * 5
        result = mcnemar_exact(a, b)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_mild_imbalance_is_not_significant(self):
        # 6 vs 4 out of 10 discordant: not enough to reject.
        a = [True] * 6 + [False] * 4
        b = [False] * 6 + [True] * 4
        assert mcnemar_exact(a, b)["significant"] is False

    def test_direction_is_symmetric(self):
        """Swapping the arms must not change the p-value."""
        a = [True] * 30 + [False] * 10
        b = [False] * 30 + [True] * 10
        forward = mcnemar_exact(a, b)
        reverse = mcnemar_exact(b, a)
        assert forward["p_value"] == reverse["p_value"]
        assert (forward["b"], forward["c"]) == (reverse["c"], reverse["b"])

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            mcnemar_exact([True], [True, False])


class TestPairedBootstrap:
    def test_identical_inputs_give_a_zero_difference(self):
        values = [1.0, 0.0] * 50
        result = paired_bootstrap(values, list(values))
        assert abs(result["point"]) < 1e-9
        assert result["low"] <= 0.0 <= result["high"]

    def test_a_large_gap_excludes_zero(self):
        a = [1.0] * 95 + [0.0] * 5
        b = [0.0] * 95 + [1.0] * 5
        result = paired_bootstrap(a, b)
        assert result["point"] > 0.8
        assert result["low"] > 0.0
        assert result["p_value"] < 0.05


class TestWilson:
    def test_zero_successes_is_an_interval_not_a_point(self):
        interval = wilson(0, 100)
        assert interval.point == 0.0
        assert interval.low == 0.0
        assert 0.0 < interval.high < 0.05

    def test_all_successes_is_symmetric(self):
        interval = wilson(100, 100)
        assert interval.point == 1.0
        assert interval.high == 1.0
        assert 0.95 < interval.low < 1.0

    def test_interval_narrows_as_n_grows(self):
        narrow = wilson(50, 1000)
        wide = wilson(50, 100)
        assert (narrow.high - narrow.low) < (wide.high - wide.low)
