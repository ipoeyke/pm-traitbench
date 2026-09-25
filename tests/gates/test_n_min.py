"""Tests for gate 1's opportunity-count minimums."""

import math

from pm_traitbench.gates.gate1.n_min import N_MIN

# (param, p0, p1): the neutral and active opportunity-share centres each n_min's margin is set from.
_PAIRS = (
    ("exit_deficiency", 0.06, 0.44),
    ("loss_aversion_lambda", 0.10, 0.40),
    ("herding_weight", 0.17, 0.58),
    ("anchoring_rho", 0.20, 0.50),
)


def test_n_min_values_are_the_ceiling_of_the_standard_error_formula() -> None:
    for param, p0, p1 in _PAIRS:
        expected = math.ceil(16 * p0 * (1 - p0) / (p1 - p0) ** 2)
        assert N_MIN[param] == expected


def test_n_min_keys_match_the_four_params_with_a_minimum() -> None:
    assert set(N_MIN) == {name for name, _, _ in _PAIRS}
