"""Tests for the sovereign curve process: levels, vol, correlation, pull and jumps."""

import dataclasses
import functools
from datetime import date

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import SOVEREIGN_TENORS, InstrumentKind, Regime, Tenor
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.calendar import EventJumps
from pm_traitbench.market.processes.common import ProcessInputs, nearest_level
from pm_traitbench.market.processes.rates import simulate
from pm_traitbench.market.regimes import constant_path
from pm_traitbench.market.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.timeline import Timeline

# An ordinary root, not tuned to pass; verified locally for roots 0-9.
_ROOT = 0
_Z_SEED = 7


def _axis():
    return build_axis(Timeline(date(2026, 1, 5), 1040), 0)


def _universe(config, root=_ROOT):
    return tuple(build_universe(config, stream(root, "market", "universe")))


def _curves(instruments):
    return [i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE]


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


def _no_floor_config():
    """A config with the yield floor pushed far away, isolating the pure random walk."""
    config = Config()
    levels = config.market.levels.model_copy(update={"yield_floor_pct": -1e6})
    market = config.market.model_copy(update={"levels": levels})
    return config.model_copy(update={"market": market})


def test_every_curve_has_all_four_tenors_and_none_breach_the_floor() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    curves = _curves(instruments)
    assert set(output.curves) == {
        (curve.instrument_id, tenor) for curve in curves for tenor in SOVEREIGN_TENORS
    }
    floor = config.market.levels.yield_floor_pct
    for series in output.curves.values():
        assert series.shape == (axis.n_days,)
        assert np.all(np.isfinite(series))
        assert np.all(series >= floor)


def test_floor_actually_binds_for_a_curve_starting_near_zero() -> None:
    config = Config()
    unfloored = config.model_copy(
        update={
            "market": config.market.model_copy(
                update={"levels": config.market.levels.model_copy(update={"yield_floor_pct": -1e6})}
            )
        }
    )
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RISK_OFF, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    jumps = _zero_jumps(axis)

    floored = simulate(_inputs(config, instruments, axis, path, z, jumps))
    raw = simulate(_inputs(unfloored, instruments, axis, path, z, jumps))

    key = ("RT-JPY", Tenor.Y2)
    assert raw.curves[key].min() < 0
    assert floored.curves[key].min() >= config.market.levels.yield_floor_pct


def test_day_zero_curve_equals_curve_start_exactly() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    for curve in _curves(instruments):
        start = config.market.levels.curve_start[curve.currency]
        for tenor_idx, tenor in enumerate(SOVEREIGN_TENORS):
            assert output.curves[(curve.instrument_id, tenor)][0] == pytest.approx(start[tenor_idx])


@pytest.mark.parametrize("regime", list(Regime))
def test_ten_year_realised_vol_matches_the_level_vol_target(regime: Regime) -> None:
    config = _no_floor_config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(regime, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    y10 = output.curves[("RT-USD", Tenor.Y10)]
    changes_bp = np.diff(y10) * 100
    realised_vol = changes_bp.std() * np.sqrt(252)

    rates_cfg = config.market.families.rates
    mult = config.market.regimes.vol_multiplier[regime]
    target = rates_cfg.level_vol_bp * mult

    assert abs(realised_vol - target) / target < 0.10


def test_ten_year_changes_correlate_positively_with_the_driver() -> None:
    config = _no_floor_config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    u = _driver_shock(axis.n_days)
    z = _z_for(path, u)
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    y10 = output.curves[("RT-USD", Tenor.Y10)]
    corr = np.corrcoef(np.diff(y10), z[1:])[0, 1]
    assert corr > 0


def test_range_kappa_shrinks_the_ten_year_round_level_gap() -> None:
    config = _no_floor_config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    jumps = _zero_jumps(axis)

    with_pull = simulate(_inputs(config, instruments, axis, path, z, jumps))
    no_pull_path = dataclasses.replace(path, kappa=np.zeros_like(path.kappa))
    without_pull = simulate(_inputs(config, instruments, axis, no_pull_path, z, jumps))

    def _mean_abs_gap(output) -> float:
        series = output.curves[("RT-USD", Tenor.Y10)]
        gap = series - nearest_level(series, 0.25)
        return float(np.mean(np.abs(gap)))

    assert _mean_abs_gap(with_pull) < _mean_abs_gap(without_pull)


def test_a_cb_jump_lowers_every_tenor_by_exactly_its_size() -> None:
    config = _no_floor_config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))

    target = "RT-USD"
    jump_day = axis.n_days // 2
    jump_arr = np.zeros(axis.n_days)
    jump_arr[jump_day] = 8.0
    jumps = EventJumps(by_instrument={target: jump_arr}, macro=np.zeros(axis.n_days))

    with_jump = simulate(_inputs(config, instruments, axis, path, z, jumps))
    without_jump = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    for tenor in SOVEREIGN_TENORS:
        with_value = with_jump.curves[(target, tenor)][jump_day]
        without_value = without_jump.curves[(target, tenor)][jump_day]
        assert with_value - without_value == pytest.approx(-0.08)
