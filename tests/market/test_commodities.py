"""Tests for the commodity process: curve shape, group vol, correlation and pull."""

import dataclasses
import functools
from datetime import date

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import FUTURES_TENORS, CommodityGroup, Family, Regime
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.calendar import EventJumps
from pm_traitbench.market.processes.commodities import simulate
from pm_traitbench.market.processes.common import ProcessInputs, log_grid_step, round_log_gap
from pm_traitbench.market.regimes import constant_path
from pm_traitbench.market.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.timeline import Timeline

# An ordinary root, not tuned to pass; verified locally for roots 0-9.
_ROOT = 0
_Z_SEED = 7
_GROUP_SEED = 11


def _axis():
    return build_axis(Timeline(date(2026, 1, 5), 1040), 0)


def _universe(config, root=_ROOT):
    return tuple(build_universe(config, stream(root, "market", "universe")))


def _commodities(instruments):
    return [i for i in instruments if i.family == Family.COMMODITIES]


def _in_group(instruments, group):
    return [c for c in _commodities(instruments) if c.commodity_group == group]


def _driver_shock(n_days, seed=_Z_SEED):
    return np.random.default_rng(seed).standard_normal(n_days)


def _group_shocks(n_days, seed=_GROUP_SEED):
    return {
        group: np.random.default_rng(seed + i).standard_normal(n_days)
        for i, group in enumerate(CommodityGroup)
    }


def _z_for(path, u):
    return u + path.driver_mean


def _zero_jumps(axis):
    return EventJumps(by_instrument={}, macro=np.zeros(axis.n_days))


def _inputs(config, instruments, axis, path, z, group_shocks, jumps, root=_ROOT):
    return ProcessInputs(
        instruments=instruments,
        axis=axis,
        path=path,
        z=z,
        group_shocks=group_shocks,
        jumps=jumps,
        market=config.market,
        rng_for=functools.partial(stream, root, "market"),
    )


def _group_index_returns(output, commodities):
    prices = np.array([output.prices[c.instrument_id] for c in commodities])
    return np.diff(np.log(prices), axis=1).mean(axis=0)


def test_curve_has_twelve_positive_finite_rows_equal_to_m1_at_the_front() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    group_shocks = _group_shocks(axis.n_days)
    output = simulate(_inputs(config, instruments, axis, path, z, group_shocks, _zero_jumps(axis)))

    commodities = _commodities(instruments)
    assert set(output.prices) == {c.instrument_id for c in commodities}
    for c in commodities:
        rows = np.array([output.curves[(c.instrument_id, tenor)] for tenor in FUTURES_TENORS])
        assert rows.shape == (len(FUTURES_TENORS), axis.n_days)
        assert np.all(np.isfinite(rows))
        assert np.all(rows > 0)
        assert np.array_equal(output.prices[c.instrument_id], rows[0])


@pytest.mark.parametrize("regime", list(Regime))
def test_curve_sign_follows_the_group_slope_on_nearly_every_day(regime: Regime) -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(regime, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    group_shocks = _group_shocks(axis.n_days)
    output = simulate(_inputs(config, instruments, axis, path, z, group_shocks, _zero_jumps(axis)))

    slope = config.market.families.commodity.curve_slope
    for group in CommodityGroup:
        commodities = _in_group(instruments, group)
        m1 = np.array([output.prices[c.instrument_id] for c in commodities])
        m12 = np.array([output.curves[(c.instrument_id, FUTURES_TENORS[-1])] for c in commodities])
        share_correct = np.mean(m12 > m1) if slope[group] > 0 else np.mean(m12 < m1)
        assert share_correct >= 0.99


@pytest.mark.parametrize("regime", list(Regime))
def test_group_index_vol_matches_the_model_implied_target(regime: Regime) -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(regime, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    group_shocks = _group_shocks(axis.n_days)
    output = simulate(_inputs(config, instruments, axis, path, z, group_shocks, _zero_jumps(axis)))

    commodity_cfg = config.market.families.commodity
    mult = config.market.regimes.vol_multiplier[regime]
    share = commodity_cfg.group_share
    for group in CommodityGroup:
        commodities = _in_group(instruments, group)
        n = len(commodities)
        realised_vol = _group_index_returns(output, commodities).std() * np.sqrt(252)

        v = commodity_cfg.group_vol[group]
        c = commodity_cfg.driver_corr[group]
        target = mult * v * np.sqrt(c**2 + (1 - c**2) * (share + (1 - share) / n))

        assert abs(realised_vol - target) / target < 0.10


def test_group_index_correlation_sign_matches_the_group_driver_corr() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    u = _driver_shock(axis.n_days)
    z = _z_for(path, u)
    group_shocks = _group_shocks(axis.n_days)
    output = simulate(_inputs(config, instruments, axis, path, z, group_shocks, _zero_jumps(axis)))

    driver_corr = config.market.families.commodity.driver_corr
    for group in CommodityGroup:
        c = driver_corr[group]
        if c == 0:
            continue
        commodities = _in_group(instruments, group)
        index_returns = _group_index_returns(output, commodities)
        corr = np.corrcoef(index_returns, z[1:])[0, 1]
        assert (corr > 0) == (c > 0)


def test_range_pull_reduces_the_mean_m1_round_level_gap() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    group_shocks = _group_shocks(axis.n_days)
    jumps = _zero_jumps(axis)
    commodities = _commodities(instruments)

    with_pull = simulate(_inputs(config, instruments, axis, path, z, group_shocks, jumps))
    no_pull_path = dataclasses.replace(path, kappa=np.zeros_like(path.kappa))
    without_pull = simulate(
        _inputs(config, instruments, axis, no_pull_path, z, group_shocks, jumps)
    )

    def _mean_abs_gap(output) -> float:
        prices = np.array([output.prices[c.instrument_id] for c in commodities])[:, :-1]
        gap = round_log_gap(prices, log_grid_step(prices))
        return float(np.mean(np.abs(gap)))

    assert _mean_abs_gap(with_pull) < _mean_abs_gap(without_pull)
