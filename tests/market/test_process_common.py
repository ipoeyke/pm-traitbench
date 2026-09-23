"""Tests for process-shared types and helpers: noise draws, grids and merging."""

import numpy as np
import pytest

from pm_traitbench.market.processes.common import (
    ProcessOutput,
    log_grid_step,
    nearest_level,
    round_log_gap,
    unit_student_t,
)


def test_unit_student_t_sample_variance_near_one() -> None:
    rng = np.random.default_rng(0)
    draws = unit_student_t(rng, df=5, size=200_000)
    assert abs(draws.var() - 1.0) < 0.05


@pytest.mark.parametrize(
    ("price", "step"),
    [(72, 5), (7.2, 0.5), (250, 50)],
)
def test_log_grid_step_halves_the_leading_decade(price: float, step: float) -> None:
    assert log_grid_step(price) == step


def test_nearest_level_rounds_to_the_nearest_grid_multiple() -> None:
    assert nearest_level(73, 5) == 75


def test_round_log_gap_is_zero_at_a_round_level() -> None:
    assert round_log_gap(75, log_grid_step(75)) == pytest.approx(0.0)


def test_merge_combines_disjoint_outputs() -> None:
    a = ProcessOutput(prices={"EQ-0001": np.array([1.0])})
    b = ProcessOutput(prices={"EQ-0002": np.array([2.0])})
    merged = a.merge(b)
    assert set(merged.prices) == {"EQ-0001", "EQ-0002"}


def test_merge_raises_on_a_duplicate_key() -> None:
    a = ProcessOutput(prices={"EQ-0001": np.array([1.0])})
    b = ProcessOutput(prices={"EQ-0001": np.array([2.0])})
    with pytest.raises(ValueError):
        a.merge(b)
