"""Tests for per-seed market generation and row conversion."""

import functools
from datetime import date

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import (
    HY_BANDS,
    CommodityGroup,
    Family,
    InstrumentKind,
    Regime,
    Tenor,
)
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.calendar import EventJumps
from pm_traitbench.market.constants import FX_PAIRS, USD_PAIR
from pm_traitbench.market.drivers import draw_shocks
from pm_traitbench.market.generate import MarketRows, SeedMarket, generate_seed, market_rng, to_rows
from pm_traitbench.market.processes import commodities, credit, equities, fx, rates
from pm_traitbench.market.processes.common import ProcessInputs
from pm_traitbench.market.regimes import constant_path
from pm_traitbench.market.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.timeline import Timeline

# An ordinary root, not tuned to pass; verified locally for roots 0-9.
_ROOT = 0
_Z_SEED = 7
_GROUP_SEED = 11
_DEFAULT_ORDER = (Regime.RANGE, Regime.RISK_OFF, Regime.RISK_ON)
_OTHER_ORDER = (Regime.RISK_ON, Regime.RANGE, Regime.RISK_OFF)


def _demo_config() -> Config:
    """A demo-sized universe: cheap enough for row-count and equality tests."""
    config = Config()
    universe = config.market.universe.model_copy(
        update={"n_equities": 20, "n_sectors": 5, "n_credit_issuers": 12}
    )
    market = config.market.model_copy(update={"universe": universe})
    return config.model_copy(update={"market": market})


def _zero_jump_zero_flip_config(seeds: dict[str, tuple[Regime, Regime, Regime]]) -> Config:
    """Seeds A and B with the given regime orders, every seed-specific draw made inert."""
    config = _demo_config()
    events = {
        event: spec.model_copy(update={"jump_size": 0.0})
        for event, spec in config.market.events.items()
    }
    consensus = config.market.consensus.model_copy(update={"flips_per_instrument_year": 0.0})
    market = config.market.model_copy(
        update={"events": events, "consensus": consensus, "seeds": seeds}
    )
    population = config.population.model_copy(update={"market_seeds": tuple(seeds)})
    return config.model_copy(update={"market": market, "population": population})


def _market_inputs(config: Config):
    """The axis, universe and driver shocks shared by every seed generated from `config`."""
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = build_universe(config, stream(config.seed.root, "market", "universe"))
    shocks = draw_shocks(config.seed.root, axis.n_days)
    return axis, instruments, shocks


def _generate(config: Config, seed: str):
    axis, instruments, shocks = _market_inputs(config)
    return axis, instruments, generate_seed(config, seed, instruments, shocks, axis)


def _generate_pair(config: Config, seed_a: str, seed_b: str):
    axis, instruments, shocks = _market_inputs(config)
    market_a = generate_seed(config, seed_a, instruments, shocks, axis)
    market_b = generate_seed(config, seed_b, instruments, shocks, axis)
    return axis, instruments, market_a, market_b


def test_market_rng_matches_the_stream_helper() -> None:
    rng_for = market_rng(123)
    drawn = rng_for("noise", "EQ-0001", "idio").standard_normal(5)
    expected = stream(123, "market", "noise", "EQ-0001", "idio").standard_normal(5)
    np.testing.assert_array_equal(drawn, expected)


def test_row_counts_and_dates_fall_within_the_horizon() -> None:
    config = _demo_config()
    axis, instruments, market = _generate(config, "A")
    rows = to_rows(market)

    assert isinstance(market, SeedMarket)
    assert isinstance(rows, MarketRows)

    n_instruments = len(instruments)
    n_curves = len(config.market.universe.curves)
    n_commodities = sum(config.market.universe.commodities.values())
    horizon_days = axis.n_days - axis.n_burn

    assert len(rows.prices) == (n_instruments - n_curves) * horizon_days
    assert len(rows.curves) == (4 * n_curves + 12 * n_commodities) * horizon_days
    assert len(rows.consensus) == n_instruments * horizon_days
    assert len(rows.regimes) == 3

    horizon_dates = set(axis.dates[axis.horizon])
    for row in (*rows.prices, *rows.curves, *rows.consensus, *rows.calendar):
        assert row.date in horizon_dates

    first_horizon, last_horizon = axis.dates[axis.n_burn], axis.dates[-1]
    for span in rows.regimes:
        assert first_horizon <= span.date_start <= span.date_end <= last_horizon


def test_seeds_with_the_same_regime_order_and_no_seed_specific_draws_match() -> None:
    config = _zero_jump_zero_flip_config({"A": _DEFAULT_ORDER, "B": _DEFAULT_ORDER})
    _, _, market_a, market_b = _generate_pair(config, "A", "B")
    rows_a, rows_b = to_rows(market_a), to_rows(market_b)

    def _without_seed(rows):
        return [row.model_dump(exclude={"seed"}) for row in rows]

    assert _without_seed(rows_a.prices) == _without_seed(rows_b.prices)
    assert _without_seed(rows_a.curves) == _without_seed(rows_b.curves)
    assert _without_seed(rows_a.consensus) == _without_seed(rows_b.consensus)


def test_seeds_with_different_regime_orders_differ_in_prices() -> None:
    # Same zero-jump, zero-flip config as the equality test above, so regime order
    # is the only thing that can make the two seeds' prices differ.
    config = _zero_jump_zero_flip_config({"A": _DEFAULT_ORDER, "B": _OTHER_ORDER})
    _, _, market_a, market_b = _generate_pair(config, "A", "B")
    rows_a, rows_b = to_rows(market_a), to_rows(market_b)

    prices_a = {(r.instrument_id, r.date): r.price for r in rows_a.prices}
    prices_b = {(r.instrument_id, r.date): r.price for r in rows_b.prices}
    assert prices_a != prices_b


def test_generate_seed_is_deterministic_across_calls() -> None:
    config = _demo_config()
    axis, instruments, shocks = _market_inputs(config)

    rows_1 = to_rows(generate_seed(config, "A", instruments, shocks, axis))
    rows_2 = to_rows(generate_seed(config, "A", instruments, shocks, axis))

    def _dump(rows):
        return [row.model_dump() for row in rows]

    assert _dump(rows_1.prices) == _dump(rows_2.prices)
    assert _dump(rows_1.curves) == _dump(rows_2.curves)
    assert _dump(rows_1.consensus) == _dump(rows_2.consensus)
    assert _dump(rows_1.calendar) == _dump(rows_2.calendar)
    assert _dump(rows_1.regimes) == _dump(rows_2.regimes)


# --- Drift-sign test: every family's index drift against the model-implied mean. ---


def _long_axis():
    return build_axis(Timeline(date(2026, 1, 5), 1040), 0)  # 20 years


def _no_floor_config() -> Config:
    """A config with the yield floor pushed away, isolating the unfloored random walk
    the drift formula assumes; the default floor binds for JPY over 20 years.
    """
    config = Config()
    levels = config.market.levels.model_copy(update={"yield_floor_pct": -1e6})
    market = config.market.model_copy(update={"levels": levels})
    return config.model_copy(update={"market": market})


def _universe(config: Config, root: int = _ROOT):
    return tuple(build_universe(config, stream(root, "market", "universe")))


def _driver_shock(n_days: int, seed: int = _Z_SEED):
    return np.random.default_rng(seed).standard_normal(n_days)


def _z_for(path, u):
    return u + path.driver_mean


def _group_shocks(n_days: int, seed: int = _GROUP_SEED):
    return {
        group: np.random.default_rng(seed + i).standard_normal(n_days)
        for i, group in enumerate(CommodityGroup)
    }


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


def _simulate_every_family(config, instruments, axis, path, z, group_shocks, jumps):
    inputs = _inputs(config, instruments, axis, path, z, group_shocks, jumps)
    equities_out = equities.simulate(inputs)
    rates_out = rates.simulate(inputs)
    # The linear duration mark diverges over a 20-year axis; only spreads are asserted on.
    with np.errstate(over="ignore", invalid="ignore"):
        credit_out = credit.simulate(inputs, rates_out)
    commodities_out = commodities.simulate(inputs)
    fx_out = fx.simulate(inputs)
    return equities_out, rates_out, credit_out, commodities_out, fx_out


def _currency_log_value(output, ccy: str) -> np.ndarray:
    """A currency's own log value against USD, signed by its anchor pair's convention."""
    anchor = USD_PAIR[ccy]
    base, quote = FX_PAIRS[anchor]
    price = output.prices[f"FX-{anchor}"]
    return np.log(price) if base == ccy else -np.log(price)


def _assert_drift_matches_model(label: str, diffs: np.ndarray, implied_mean: float) -> None:
    """Realised mean must sit within 4 standard errors of the model-implied mean; the
    sign is checked only where the implied mean clears 5 standard errors, since a
    smaller drift is not reliably detectable on a 20-year span.
    """
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    realised_mean = diffs.mean()
    assert abs(realised_mean - implied_mean) <= 4 * se, (
        f"{label}: realised {realised_mean:.3g} vs implied {implied_mean:.3g}, se {se:.3g}"
    )
    if abs(implied_mean) >= 5 * se:
        assert (realised_mean > 0) == (implied_mean > 0), (
            f"{label}: sign mismatch, realised {realised_mean:.3g} vs implied {implied_mean:.3g}"
        )


@pytest.mark.parametrize("regime", [Regime.RISK_OFF, Regime.RISK_ON])
def test_family_index_drift_matches_the_model_implied_mean(regime: Regime) -> None:
    config = _no_floor_config()
    axis = _long_axis()
    instruments = _universe(config)
    path = constant_path(regime, axis.n_days, config)
    z = _z_for(path, _driver_shock(axis.n_days))
    group_shocks = _group_shocks(axis.n_days)
    jumps = _zero_jumps(axis)

    equities_out, rates_out, credit_out, commodities_out, fx_out = _simulate_every_family(
        config, instruments, axis, path, z, group_shocks, jumps
    )

    mult = config.market.regimes.vol_multiplier[regime]
    driver_mean = config.market.regimes.driver_mean[regime]

    equity_list = [i for i in instruments if i.family == Family.EQUITIES]
    equity_index = np.log(
        np.array([equities_out.prices[e.instrument_id] for e in equity_list])
    ).mean(axis=0)
    eq_cfg = config.market.families.equity
    eq_implied = (
        np.mean([e.beta for e in equity_list]) * eq_cfg.market_vol * mult / np.sqrt(252)
    ) * driver_mean
    _assert_drift_matches_model("equities", np.diff(equity_index), eq_implied)

    curve_list = [i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE]
    y10_index = np.array([rates_out.curves[(c.instrument_id, Tenor.Y10)] for c in curve_list]).mean(
        axis=0
    )
    rates_cfg = config.market.families.rates
    mean_level_vol = np.mean([rates_cfg.level_vol_bp[c.currency] for c in curve_list]) / 100
    rates_implied = (mean_level_vol * mult / np.sqrt(252) * rates_cfg.driver_corr) * driver_mean
    _assert_drift_matches_model("rates_10y", np.diff(y10_index), rates_implied)

    ig_issuers = [
        i for i in instruments if i.family == Family.CREDIT and i.rating_band not in HY_BANDS
    ]
    credit_index = np.log(np.array([credit_out.spreads[i.instrument_id] for i in ig_issuers])).mean(
        axis=0
    )
    credit_cfg = config.market.families.credit
    a = credit_cfg.asymmetry
    credit_implied = (
        credit_cfg.factor_vol * mult / np.sqrt(252) * credit_cfg.driver_corr * (a + 1 / a) / 2
    ) * driver_mean
    _assert_drift_matches_model("credit_ig_spread", np.diff(credit_index), credit_implied)

    commodity_list = [i for i in instruments if i.family == Family.COMMODITIES]
    commodity_index = np.log(
        np.array([commodities_out.prices[c.instrument_id] for c in commodity_list])
    ).mean(axis=0)
    commodity_cfg = config.market.families.commodity
    commodity_loadings = [
        commodity_cfg.group_vol[c.commodity_group]
        * mult
        / np.sqrt(252)
        * commodity_cfg.driver_corr[c.commodity_group]
        for c in commodity_list
    ]
    commodity_implied = np.mean(commodity_loadings) * driver_mean
    _assert_drift_matches_model("commodities", np.diff(commodity_index), commodity_implied)

    fx_index = np.array([_currency_log_value(fx_out, ccy) for ccy in USD_PAIR]).mean(axis=0)
    fx_cfg = config.market.families.fx
    fx_loadings = [
        fx_cfg.currency_vol[ccy] * mult / np.sqrt(252) * fx_cfg.driver_corr[ccy] for ccy in USD_PAIR
    ]
    fx_implied = np.mean(fx_loadings) * driver_mean
    _assert_drift_matches_model("fx", np.diff(fx_index), fx_implied)
