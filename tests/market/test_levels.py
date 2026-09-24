"""Tests for the shared round-level grid helpers."""

import pytest

from pm_traitbench.market.levels import (
    SPREAD_STEP_BP,
    YIELD_STEP_PCT,
    log_grid_step,
    nearest_level,
    round_log_gap,
)


def test_log_grid_step_halves_the_leading_decade() -> None:
    assert log_grid_step(72) == 5


def test_nearest_level_rounds_to_the_nearest_grid_multiple() -> None:
    assert nearest_level(4.13, 0.25) == 4.25


def test_round_log_gap_is_zero_at_a_round_level() -> None:
    assert round_log_gap(75, log_grid_step(75)) == pytest.approx(0.0)


def test_yield_and_spread_step_constants() -> None:
    assert YIELD_STEP_PCT == 0.25
    assert SPREAD_STEP_BP == 10.0


def test_processes_common_still_exposes_the_three_helpers() -> None:
    from pm_traitbench.market.synthetic.processes.common import (
        log_grid_step as common_log_grid_step,
    )
    from pm_traitbench.market.synthetic.processes.common import (
        nearest_level as common_nearest_level,
    )
    from pm_traitbench.market.synthetic.processes.common import (
        round_log_gap as common_round_log_gap,
    )

    assert common_log_grid_step is log_grid_step
    assert common_nearest_level is nearest_level
    assert common_round_log_gap is round_log_gap
