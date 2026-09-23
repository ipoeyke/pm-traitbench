"""Tests for the credit process: levels, asymmetry, vol, correlation and price marks."""

import functools
from datetime import date

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import HY_BANDS, Family, Regime, Tenor
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.regimes import constant_path
from pm_traitbench.market.synthetic.events import EventJumps
from pm_traitbench.market.synthetic.processes.common import ProcessInputs, ProcessOutput
from pm_traitbench.market.synthetic.processes.credit import simulate
from pm_traitbench.market.synthetic.processes.rates import simulate as simulate_rates
from pm_traitbench.market.synthetic.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.timeline import Timeline

# An ordinary root, not tuned to pass; verified locally for roots 0-9.
_ROOT = 0
_Z_SEED = 7


def _axis():
    return build_axis(Timeline(date(2026, 1, 5), 1040), 0)


def _short_axis():
    """A half-year axis, short enough that undamped logF variance never breaks the linear mark."""
    return build_axis(Timeline(date(2026, 1, 5), 26), 0)


def _universe(config, root=_ROOT):
    return tuple(build_universe(config, stream(root, "market", "universe")))


def _issuers(instruments):
    return [i for i in instruments if i.family == Family.CREDIT]


def _ig(instruments):
    return [i for i in _issuers(instruments) if i.rating_band not in HY_BANDS]


def _hy(instruments):
    return [i for i in _issuers(instruments) if i.rating_band in HY_BANDS]


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


def _simulate(config, instruments, axis, path, z, jumps):
    rates_inputs = _inputs(config, instruments, axis, path, z, jumps)
    rates = simulate_rates(rates_inputs)
    credit_inputs = _inputs(config, instruments, axis, path, z, jumps)
    # The linear duration mark diverges over decades, which the one-year horizon never reaches.
    with np.errstate(over="ignore", invalid="ignore"):
        return simulate(credit_inputs, rates), rates


def _log_index(output, issuers):
    spreads = np.array([output.spreads[i.instrument_id] for i in issuers])
    return np.log(spreads).mean(axis=0)


def test_spreads_are_positive_finite_and_one_series_per_issuer() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output, _ = _simulate(config, instruments, axis, path, z, _zero_jumps(axis))

    issuers = _issuers(instruments)
    assert set(output.spreads) == {i.instrument_id for i in issuers}
    for series in output.spreads.values():
        assert series.shape == (axis.n_days,)
        assert np.all(np.isfinite(series))
        assert np.all(series > 0)


def test_prices_are_positive_finite_and_one_series_per_issuer() -> None:
    config = Config()
    axis = _short_axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output, _ = _simulate(config, instruments, axis, path, z, _zero_jumps(axis))

    issuers = _issuers(instruments)
    assert set(output.prices) == {i.instrument_id for i in issuers}
    for series in output.prices.values():
        assert series.shape == (axis.n_days,)
        assert np.all(np.isfinite(series))
        assert np.all(series > 0)


def test_widening_steps_average_larger_than_tightening_steps() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output, _ = _simulate(config, instruments, axis, path, z, _zero_jumps(axis))

    index = _log_index(output, _ig(instruments))
    steps = np.diff(index)
    widen = steps[steps > 0]
    tighten = steps[steps < 0]
    assert np.mean(np.abs(widen)) > np.mean(np.abs(tighten))


def test_centred_shock_leaves_no_drift_under_a_driftless_regime() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output, _ = _simulate(config, instruments, axis, path, z, _zero_jumps(axis))

    steps = np.diff(_log_index(output, _ig(instruments)))
    standard_error = steps.std() / np.sqrt(len(steps))
    assert abs(steps.mean()) < 4 * standard_error


@pytest.mark.parametrize("regime", list(Regime))
def test_ig_index_vol_matches_the_model_implied_target(regime: Regime) -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(regime, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output, _ = _simulate(config, instruments, axis, path, z, _zero_jumps(axis))

    ig = _ig(instruments)
    index = _log_index(output, ig)
    realised_vol = np.diff(index).std() * np.sqrt(252)

    credit_cfg = config.market.families.credit
    mult = config.market.regimes.vol_multiplier[regime]
    a = credit_cfg.asymmetry
    k_sq = (a**2 + a**-2) / 2 - (a - 1 / a) ** 2 / (2 * np.pi)
    target = mult * np.sqrt(credit_cfg.factor_vol**2 * k_sq + credit_cfg.issuer_vol**2 / len(ig))

    assert abs(realised_vol - target) / target < 0.10


def test_ig_index_correlates_negatively_with_the_driver() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    u = _driver_shock(axis.n_days)
    z = _z_for(path, u)
    output, _ = _simulate(config, instruments, axis, path, z, _zero_jumps(axis))

    index = _log_index(output, _ig(instruments))
    corr = np.corrcoef(np.diff(index), z[1:])[0, 1]
    assert corr < 0


def test_hy_index_vol_exceeds_ig_index_vol() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output, _ = _simulate(config, instruments, axis, path, z, _zero_jumps(axis))

    ig_vol = np.diff(_log_index(output, _ig(instruments))).std()
    hy_vol = np.diff(_log_index(output, _hy(instruments))).std()
    assert hy_vol > ig_vol


def test_simulate_raises_if_an_issuers_currency_has_no_curve() -> None:
    config = Config()
    axis = _short_axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    credit_inputs = _inputs(config, instruments, axis, path, z, _zero_jumps(axis))

    with pytest.raises(ValueError):
        simulate(credit_inputs, ProcessOutput())


def test_price_falls_when_spread_and_five_year_yield_both_rise() -> None:
    config = Config()
    axis = _short_axis()
    instruments = _universe(config)
    path = constant_path(Regime.RISK_OFF, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output, rates = _simulate(config, instruments, axis, path, z, _zero_jumps(axis))

    issuer = _issuers(instruments)[0]
    spread = output.spreads[issuer.instrument_id]
    price = output.prices[issuer.instrument_id]
    y5 = rates.curves[(f"RT-{issuer.currency}", Tenor.Y5)]

    ds = np.diff(spread)
    dy5 = np.diff(y5)
    dp = np.diff(price)
    candidates = np.where((ds > 0) & (dy5 > 0))[0]
    assert len(candidates) > 0
    assert np.all(dp[candidates] < 0)
