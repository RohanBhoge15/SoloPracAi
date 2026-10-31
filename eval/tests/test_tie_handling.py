"""Tests for tie-aware rank accounting.

The bug these exist to prevent: a ranker returned a plain ordered list and
the harness read gold's position off it, so whichever way a tied group
happened to be sorted *was* the result. `rank_lexical` broke ties on
`event_id`, which encodes insertion order, which encodes chronology, so
the original of every corrected test won automatically. Token overlap
scored 99.8% historical accuracy with no temporal reasoning in it at all.

The fix is that rank comes from the method's own scores, averaged over
ties. These tests pin that behaviour down, including the case that is
easiest to get wrong.
"""

from __future__ import annotations

import pytest

from tapqa.ranking import Ranking, from_order, from_scores, gold_rank, tie_count


def test_untied_gold_keeps_its_position():
    r = from_scores([(0.9, "a"), (0.5, "b"), (0.1, "c")])
    assert gold_rank(r, ["a"]) == 1.0
    assert gold_rank(r, ["b"]) == 2.0
    assert gold_rank(r, ["c"]) == 3.0


def test_absent_gold_is_zero():
    r = from_scores([(0.9, "a"), (0.5, "b")])
    assert gold_rank(r, ["zzz"]) == 0.0


def test_tie_at_the_top_denies_credit():
    """The case that mattered.

    Two records score identically, so the method has not separated them.
    Rounding 1.5 down to 1 would hand back exactly the credit the tie
    handling exists to withhold.
    """
    r = from_scores([(1.0, "original"), (1.0, "correction")])
    assert gold_rank(r, ["original"]) == 1.5
    assert gold_rank(r, ["correction"]) == 1.5
    # The property the metrics actually depend on.
    assert (gold_rank(r, ["original"]) == 1) is False


def test_tied_gold_is_symmetric_regardless_of_input_order():
    """Which of the tied pair is listed first must not change the rank.

    This is the regression test for the chronology artefact: the two
    versions of a corrected test are interchangeable to a lexical matcher,
    so swapping them may not move the number.
    """
    a = from_scores([(1.0, "original"), (1.0, "correction")])
    b = from_scores([(1.0, "correction"), (1.0, "original")])
    assert gold_rank(a, ["original"]) == gold_rank(b, ["original"])
    assert gold_rank(a, ["correction"]) == gold_rank(b, ["correction"])


def test_tie_group_rank_is_the_midpoint():
    r = from_scores([(0.1, "d"), (0.9, "a"), (0.9, "b"), (0.9, "c")])
    # Tied group occupies 1-3, so the midpoint is 2.
    assert gold_rank(r, ["a"]) == 2.0
    assert gold_rank(r, ["b"]) == 2.0
    assert gold_rank(r, ["c"]) == 2.0
    assert gold_rank(r, ["d"]) == 4.0


def test_larger_tie_group():
    r = from_scores([(1.0, f"e{i}") for i in range(5)])
    # Group occupies 1-5, midpoint 3.
    assert gold_rank(r, ["e0"]) == 3.0


def test_one_of_several_acceptable_answers_may_win():
    r = from_scores([(0.9, "a"), (0.4, "b")])
    assert gold_rank(r, ["b", "a"]) == 1.0


def test_from_order_reports_a_strict_order():
    r = from_order(["x", "y", "z"])
    assert gold_rank(r, ["x"]) == 1.0
    assert gold_rank(r, ["z"]) == 3.0
    assert tie_count(r) == 0


def test_tie_count():
    r = from_scores([(1.0, "a"), (1.0, "b"), (0.5, "c")])
    assert tie_count(r) == 1
    assert tie_count(from_order(["a", "b"])) == 0


def test_empty_ranking():
    r = Ranking(ids=(), scores=())
    assert gold_rank(r, ["a"]) == 0.0
    assert r.top is None


def test_ranking_is_iterable_as_ids():
    """Call sites index and iterate a ranking; it must still behave."""
    r = from_scores([(0.9, "a"), (0.5, "b")])
    assert list(r) == ["a", "b"]
    assert r[0] == "a"
    assert r.top == "a"
    assert len(r) == 2
    assert r.contains_all(["a", "b"])
    assert not r.contains_all(["a", "zzz"])


def test_lexical_baseline_ties_original_and_correction():
    """The specific regression, end to end through the real baseline.

    `rank_lexical` cannot tell a value from the value that corrected it -
    that is why it is in the benchmark as a floor. If this test ever fails
    because token overlap got "better", the floor has started answering
    questions it has no information about.
    """
    from tapqa.baselines import rank_lexical
    from tapqa.dataset import Query, Record

    records = [
        Record(
            event_id="p1-e1",
            patient_id="p1",
            event_type="lab_result",
            valid_time="2026-01-01T00:00:00+00:00",
            recorded_at="2026-01-01T00:00:00+00:00",
            payload={"test": "Glucose", "value": "186 mg/dL"},
        ),
        Record(
            event_id="p1-e2",
            patient_id="p1",
            event_type="lab_result",
            valid_time="2026-01-01T00:00:00+00:00",
            recorded_at="2026-01-03T00:00:00+00:00",
            payload={"test": "Glucose", "value": "142 mg/dL"},
            supersedes_id="p1-e1",
        ),
    ]
    query = Query(
        query_id="q1",
        patient_id="p1",
        text="what was her glucose as of 2026-01-01",
        kind="historical",
        gold_event_id="p1-e1",
        test="Glucose",
    )

    ranking = rank_lexical(query, records)
    assert tie_count(ranking) == 1, "the two versions should score equally"
    # Neither version is preferred, so neither is credited.
    assert gold_rank(ranking, ["p1-e1"]) == 1.5
    assert gold_rank(ranking, ["p1-e2"]) == 1.5
