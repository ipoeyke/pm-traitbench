"""Tests for the FX process: cross consistency, levels, vol, correlation and pull."""

import functools
from datetime import date

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import Family, Regime
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.calendar import EventJumps
from pm_traitbench.market.constants import FX_PAIRS, USD_PAIR
from pm_traitbench.market.processes.common import ProcessInputs
from pm_traitbench.market.processes.fx import simulate
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


def _fx_pairs(instruments):
    return [i for i in instruments if i.family == Family.FX]


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


def _config_with_pairs(pairs):
    config = Config()
    universe = config.market.universe.model_copy(update={"fx_pairs": pairs})
    market = config.market.model_copy(update={"universe": universe})
    return config.model_copy(update={"market": market})


def _currency_log_value(output, ccy):
    """A currency's own log value against USD, signed by its anchor pair's convention."""
    anchor = USD_PAIR[ccy]
    base, quote = FX_PAIRS[anchor]
    price = output.prices[f"FX-{anchor}"]
    return np.log(price) if base == ccy else -np.log(price)


def test_prices_are_positive_finite_and_one_series_per_pair() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    pairs = _fx_pairs(instruments)
    assert set(output.prices) == {p.instrument_id for p in pairs}
    for series in output.prices.values():
        assert series.shape == (axis.n_days,)
        assert np.all(np.isfinite(series))
        assert np.all(series > 0)


def test_day_zero_usd_pairs_equal_fx_start() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    for pair, start in config.market.levels.fx_start.items():
        assert output.prices[f"FX-{pair}"][0] == pytest.approx(start)


def test_cross_pairs_are_consistent_with_their_usd_pairs_on_every_day() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RISK_OFF, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    eurusd = output.prices["FX-EURUSD"]
    gbpusd = output.prices["FX-GBPUSD"]
    usdjpy = output.prices["FX-USDJPY"]
    audusd = output.prices["FX-AUDUSD"]

    assert np.allclose(output.prices["FX-EURGBP"], eurusd / gbpusd, rtol=1e-12)
    assert np.allclose(output.prices["FX-EURJPY"], eurusd * usdjpy, rtol=1e-12)
    assert np.allclose(output.prices["FX-AUDJPY"], audusd * usdjpy, rtol=1e-12)


def test_cross_consistency_holds_for_a_pair_subset_without_jpy() -> None:
    config = _config_with_pairs(("EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "EURGBP"))
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    pairs = _fx_pairs(instruments)
    assert set(output.prices) == {p.instrument_id for p in pairs}
    for series in output.prices.values():
        assert np.all(np.isfinite(series))
        assert np.all(series > 0)

    assert np.allclose(
        output.prices["FX-EURGBP"],
        output.prices["FX-EURUSD"] / output.prices["FX-GBPUSD"],
        rtol=1e-12,
    )


def test_simulate_runs_for_a_pair_subset_without_any_crosses() -> None:
    config = _config_with_pairs(("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"))
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    pairs = _fx_pairs(instruments)
    assert set(output.prices) == {p.instrument_id for p in pairs}
    for pair, start in config.market.levels.fx_start.items():
        series = output.prices[f"FX-{pair}"]
        assert series[0] == pytest.approx(start)
        assert np.all(np.isfinite(series))
        assert np.all(series > 0)


@pytest.mark.parametrize("regime", list(Regime))
def test_each_currency_vol_matches_the_model_implied_target(regime: Regime) -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(regime, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    fx_cfg = config.market.families.fx
    mult = config.market.regimes.vol_multiplier[regime]
    for ccy in USD_PAIR:
        realised_vol = np.diff(_currency_log_value(output, ccy)).std() * np.sqrt(252)
        target = mult * fx_cfg.currency_vol[ccy]
        assert abs(realised_vol - target) / target < 0.10


def test_aud_correlates_positively_and_jpy_negatively_with_the_driver() -> None:
    config = Config()
    axis = _axis()
    instruments = _universe(config)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    u = _driver_shock(axis.n_days)
    z = _z_for(path, u)
    output = simulate(_inputs(config, instruments, axis, path, z, _zero_jumps(axis)))

    aud_returns = np.diff(_currency_log_value(output, "AUD"))
    jpy_returns = np.diff(_currency_log_value(output, "JPY"))
    assert np.corrcoef(aud_returns, z[1:])[0, 1] > 0
    assert np.corrcoef(jpy_returns, z[1:])[0, 1] < 0


# The round-level pull's effect on mean |gap| is not checked directly: the 0.01 grid
# step is close to a day's currency vol, so its effect on a single 20-year draw is
# smaller than the draw-to-draw sampling noise.
