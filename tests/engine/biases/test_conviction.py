"""Tests for the conviction-mixture sizing rank."""

import math

import numpy as np
import pytest
from scipy.stats import pearsonr

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.biases.conviction import size_rank
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.rng import stream

_CONVICTION = 3


def _params(m: float, *, active: bool = True) -> EffectiveParams:
    """An EffectiveParams over all eight BIAS_PARAMS, miscalibration set and the rest neutral."""
    values = {param: 0.5 for param in BIAS_PARAMS}
    values["conviction_size_miscalibration"] = m
    active_map = {param: False for param in BIAS_PARAMS}
    active_map["conviction_size_miscalibration"] = active
    return EffectiveParams(values=values, active=active_map)


def _se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def test_m_zero_always_returns_the_stated_conviction() -> None:
    n = 20_000
    params = _params(0.0)
    rng = stream(1, "conviction-m-zero")
    for _ in range(n):
        rank, flag = size_rank(_CONVICTION, params, rng)
        assert rank == _CONVICTION
        assert flag is None


def test_m_one_spreads_ranks_uniformly_over_one_to_five() -> None:
    n = 20_000
    params = _params(1.0)
    rng = stream(1, "conviction-m-one")
    ranks = [size_rank(_CONVICTION, params, rng)[0] for _ in range(n)]
    for r in range(1, 6):
        share = sum(1 for rank in ranks if rank == r) / n
        assert share == pytest.approx(0.2, abs=4 * _se(0.2, n))


def test_m_half_keeps_conviction_on_the_mixture_share() -> None:
    n = 20_000
    params = _params(0.5)
    rng = stream(1, "conviction-m-half")
    ranks = [size_rank(_CONVICTION, params, rng)[0] for _ in range(n)]
    share = sum(1 for rank in ranks if rank == _CONVICTION) / n
    expected = 0.5 + 0.5 * (1 / 5)
    assert share == pytest.approx(expected, abs=4 * _se(expected, n))


def test_flag_set_only_when_active_and_rank_differs() -> None:
    params_active = _params(1.0, active=True)
    rng = stream(1, "conviction-flags-active")
    saw_mismatch = False
    for _ in range(500):
        rank, flag = size_rank(_CONVICTION, params_active, rng)
        if rank != _CONVICTION:
            saw_mismatch = True
            assert flag == "conviction:mis_sized"
        else:
            assert flag is None
    assert saw_mismatch


def test_no_flag_when_inactive() -> None:
    params_inactive = _params(1.0, active=False)
    rng = stream(1, "conviction-flags-inactive")
    for _ in range(500):
        _, flag = size_rank(_CONVICTION, params_inactive, rng)
        assert flag is None


def test_rank_uncorrelated_with_conviction_at_m_one() -> None:
    # m=1 draws the rank independently of conviction, so the sample correlation
    # over 2,000 draws should sit near zero; 0.15 gives ample margin over the
    # ~0.022 standard error implied by the sample size, with no seed hunting.
    params = _params(1.0)
    rng = stream(1, "conviction-corr")
    convictions = rng.integers(1, 6, size=2000)
    ranks = np.array([size_rank(int(c), params, rng)[0] for c in convictions])
    corr, _ = pearsonr(convictions, ranks)
    assert abs(corr) < 0.15
