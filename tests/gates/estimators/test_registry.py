"""Tests for the estimator registry."""

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.gates.gate1.estimators import ESTIMATORS


def test_registry_keys_equal_bias_params():
    assert set(ESTIMATORS) == set(BIAS_PARAMS)


def test_only_overconfidence_has_higher_is_stronger_false():
    weaker_on_higher = {param for param, spec in ESTIMATORS.items() if not spec.higher_is_stronger}
    assert weaker_on_higher == {"overconfidence_coverage"}
