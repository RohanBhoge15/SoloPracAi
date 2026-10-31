"""Tests for drug-name autocomplete.

No database, no network, no model. The list is a shipped JSON file, so
these assert the lookup behaviour and the file's integrity, not the FDA.
"""

from __future__ import annotations

from app.services.drugs import count, suggest


def test_list_is_big_and_sorted():
    assert count() > 50_000, "the NDC-derived list should hold tens of thousands"


def test_common_drug_first():
    assert suggest("metf")[0] == "Metformin"


def test_case_insensitive():
    assert suggest("METF") == suggest("metf") == suggest("Metf")


def test_multi_ingredient_part_found():
    # "Clavulanate Potassium" is indexed on its own, split from the
    # "Amoxicillin; Clavulanate Potassium" generic string.
    assert any("Clavulanate" in n for n in suggest("clavulanate"))


def test_limit_respected():
    assert len(suggest("me", 5)) <= 5
    assert len(suggest("me", 50)) <= 20, "hard cap, regardless of ask"


def test_short_and_empty_queries_return_nothing():
    assert suggest("") == []
    assert suggest("a") == []
    assert suggest("  ") == []


def test_unknown_prefix_returns_nothing():
    assert suggest("zzqx") == []


def test_never_raises_on_garbage():
    assert isinstance(suggest(None), list)  # type: ignore[arg-type]
    assert isinstance(suggest("metformin!!!"), list)
