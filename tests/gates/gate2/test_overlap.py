"""Tests for gate 2's cross-PM word n-gram containment."""

from pm_traitbench.gates.gate2.overlap import containment_by_pm, ngrams, summarise


def test_ngrams_lowercases_and_splits_on_non_letters() -> None:
    assert ngrams("Buy 10Y, sell 2Y", 2) == frozenset({("buy", "y"), ("y", "sell"), ("sell", "y")})


def test_ngrams_empty_below_n() -> None:
    assert ngrams("ab cd", 5) == frozenset()


def test_identical_pms_give_one_and_disjoint_give_zero() -> None:
    identical = {"pm_001": "the quick brown fox", "pm_002": "the quick brown fox"}
    assert containment_by_pm(identical, 2) == {"pm_001": 1.0, "pm_002": 1.0}

    disjoint = {"pm_001": "aa bb cc", "pm_002": "dd ee ff"}
    assert containment_by_pm(disjoint, 2) == {"pm_001": 0.0, "pm_002": 0.0}


def test_containment_is_against_the_nearest_pm_not_the_union() -> None:
    texts = {
        "pm_a": "aa bb cc dd ee ff gg hh",
        "pm_b": "aa bb cc dd ee ff",
        "pm_c": "cc dd ee ff gg hh",
    }
    result = containment_by_pm(texts, 5)
    assert result["pm_a"] == 0.5


def test_single_pm_gives_zero() -> None:
    assert containment_by_pm({"pm_001": "aa bb cc"}, 2) == {"pm_001": 0.0}


def test_summarise_median_p90_max_and_empty() -> None:
    values = [float(v) for v in range(1, 11)]

    result = summarise(values)

    assert result == {"median": 5.5, "p90": 9.0, "max": 10.0}
    assert summarise([]) == {"median": 0.0, "p90": 0.0, "max": 0.0}
