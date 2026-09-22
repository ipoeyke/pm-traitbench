"""Tests for the equity price process: levels, vol, correlation, pulls and jumps."""

import dataclasses
import functools
from datetime import date

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import Family, InstrumentKind, Regime
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.calendar import EventJumps
from pm_traitbench.market.processes.common import ProcessInputs, log_grid_step, round_log_gap
from pm_traitbench.market.processes.equities import simulate
from pm_traitbench.market.regimes import constant_path
from pm_traitbench.market.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import Instrument
from pm_traitbench.timeline import Timeline

# An ordinary root, not tuned to pass; verified locally for roots 0-9.
_ROOT = 0
_Z_SEED = 7


def _axis():
    return build_axis(Timeline(date(2026, 1, 5), 1040), 0)


def _universe(config, root=_ROOT):
    return tuple(build_universe(config, stream(root, "market", "universe")))


def _equities(instruments):
    return [i for i in instruments if i.family == Family.EQUITIES]


def _driver_shock(n_days, seed=_Z_SEED):
    return np.random.default_rng(seed).standard_normal(n_days)


def _z_for(path, u):
    return u + path.driver_mean


def _zero_jumps(axis):
    return EventJumps(by_instrument={}, macro=np.zeros(axis.n_days))


def _inputs(config, instruments, axis, path, z, jumps, root=_ROOT):
    return ProcessInputs(
        instruments=instruments,
        axis=axis,
        path=path,
        z=z,
        group_shocks={},
        jumps=jumps,
        market=config.market,
        rng_for=functools.partial(stream, root, "market"),
    )


def _index_returns(output, equities):
    prices = np.array([output.prices[e.instrument_id] for e in equities])
    return np.diff(np.log(prices), axis=1).mean(axis=0)


def test_prices_are_positive_finite_and_one_series_per_equity() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    equities = _equities(instruments)
    assert set(output.prices) == {e.instrument_id for e in equities}
    for series in output.prices.values():
        assert series.shape == (axis.n_days,)
        assert np.all(np.isfinite(series))
        assert np.all(series > 0)


@pytest.mark.parametrize("regime", list(Regime))
def test_index_realised_vol_matches_the_model_implied_target(regime: Regime) -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(regime, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    equities = _equities(instruments)
    realised_vol = _index_returns(output, equities).std() * np.sqrt(252)

    n = len(equities)
    mean_beta = np.mean([e.beta for e in equities])
    eq = config.market.families.equity
    mult = config.market.regimes.vol_multiplier[regime]
    target = mult * np.sqrt(mean_beta**2 * eq.market_vol**2 + eq.idio_vol**2 / n)

    assert abs(realised_vol - target) / target < 0.10


def test_index_returns_correlate_strongly_with_the_driver() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    u = _driver_shock(axis.n_days)
    z = _z_for(path, u)
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    equities = _equities(instruments)
    index_returns = _index_returns(output, equities)
    corr = np.corrcoef(index_returns, z[1:])[0, 1]
    assert corr > 0.9


def test_mean_reversion_pull_shrinks_the_round_level_gap() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    jumps = _zero_jumps(axis)
    equities = _equities(instruments)

    with_pull = simulate(_inputs(config, instruments, axis, path, z, jumps))
    no_pull_path = dataclasses.replace(path, kappa=np.zeros_like(path.kappa))
    without_pull = simulate(_inputs(config, instruments, axis, no_pull_path, z, jumps))

    def _mean_abs_gap(output) -> float:
        prices = np.array([output.prices[e.instrument_id] for e in equities])[:, :-1]
        gap = round_log_gap(prices, log_grid_step(prices))
        return float(np.mean(np.abs(gap)))

    assert _mean_abs_gap(with_pull) < _mean_abs_gap(without_pull)


def test_an_earnings_jump_raises_the_log_return_by_exactly_its_size() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))

    target = _equities(instruments)[0].instrument_id
    jump_day = axis.n_days // 2
    jump_size = 0.05
    jump_arr = np.zeros(axis.n_days)
    jump_arr[jump_day] = jump_size
    jumps = EventJumps(by_instrument={target: jump_arr}, macro=np.zeros(axis.n_days))

    with_jump = simulate(_inputs(config, instruments, axis, path, z, jumps))
    without_jump = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    r_with = np.log(with_jump.prices[target][jump_day] / with_jump.prices[target][jump_day - 1])
    r_without = np.log(
        without_jump.prices[target][jump_day] / without_jump.prices[target][jump_day - 1]
    )
    assert r_with - r_without == pytest.approx(jump_size)


def test_adding_an_equity_leaves_the_others_paths_unchanged() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    jumps = _zero_jumps(axis)

    base_output = simulate(_inputs(config, instruments, axis, path, z, jumps))

    extra = Instrument(
        instrument_id="EQ-9999",
        family=Family.EQUITIES,
        kind=InstrumentKind.EQUITY,
        name="Equity 9999",
        currency="USD",
        sector="sector_01",
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=1.0,
        expiry_rule=None,
    )
    extended_output = simulate(_inputs(config, (*instruments, extra), axis, path, z, jumps))

    for instrument in _equities(instruments):
        assert np.array_equal(
            base_output.prices[instrument.instrument_id],
            extended_output.prices[instrument.instrument_id],
        )
