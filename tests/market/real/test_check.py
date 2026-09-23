"""Tests for the real-market data-quality check."""

import dataclasses
import json
from datetime import timedelta

import numpy as np
import pytest

from pm_traitbench.config import REAL_FILL_LIMIT, Config
from pm_traitbench.enums import EventType, Family, Regime
from pm_traitbench.errors import MarketCheckError
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.real.build import build_seed
from pm_traitbench.market.real.check import check_real_market
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.universe import build_real_universe


def _build(config: Config, result, seed: str = "R1"):
    cache = RawCache.open(result.data_dir)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = build_real_universe(config, cache, axis)
    market = build_seed(config, seed, instruments, cache, axis)
    return instruments, market


def test_check_real_market_passes_on_the_fixture_build(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    report = check_real_market(market, instruments, config)

    assert report.seed == "R1"
    failed = [m for m in report.metrics if not m.passed]
    assert not failed, failed


def test_nan_price_fails_naming_finite_and_the_series(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    equity = next(i for i in instruments if i.family == Family.EQUITIES)
    prices = dict(market.output.prices)
    series = prices[equity.instrument_id].copy()
    series[5] = np.nan
    prices[equity.instrument_id] = series
    output = dataclasses.replace(market.output, prices=prices)
    modified = dataclasses.replace(market, output=output)

    with pytest.raises(MarketCheckError) as excinfo:
        check_real_market(modified, instruments, config)
    assert "finite" in str(excinfo.value)
    assert equity.instrument_id in str(excinfo.value)


def test_fill_above_the_limit_fails_naming_fill_run(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    fills = dict(market.fills)
    name = next(iter(fills))
    fills[name] = REAL_FILL_LIMIT + 1
    modified = dataclasses.replace(market, fills=fills)

    with pytest.raises(MarketCheckError) as excinfo:
        check_real_market(modified, instruments, config)
    assert "fill_run" in str(excinfo.value)


def test_moved_expiry_date_fails(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    calendar = list(market.calendar)
    idx = next(i for i, row in enumerate(calendar) if row.event == EventType.CONTRACT_EXPIRY)
    moved = calendar[idx].model_copy(update={"date": calendar[idx].date + timedelta(days=7)})
    calendar[idx] = moved
    modified = dataclasses.replace(market, calendar=calendar)

    with pytest.raises(MarketCheckError) as excinfo:
        check_real_market(modified, instruments, config)
    assert "count:contract_expiry" in str(excinfo.value)


def test_earnings_rows_five_axis_days_apart_fails_naming_earnings_spacing(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    calendar = list(market.calendar)
    earnings = [(i, row) for i, row in enumerate(calendar) if row.event == EventType.EARNINGS]
    anchor_row = earnings[0][1]
    anchor_instrument = anchor_row.instrument_id
    anchor_t = market.axis.index(anchor_row.date)
    close_t = anchor_t + 5 if anchor_t + 5 < market.axis.n_days else anchor_t - 5
    close_date = market.axis.dates[close_t]

    other_idx, other_row = next(
        (i, row) for i, row in earnings if row.instrument_id != anchor_instrument
    )
    calendar[other_idx] = other_row.model_copy(
        update={"instrument_id": anchor_instrument, "date": close_date}
    )
    modified = dataclasses.replace(market, calendar=calendar)

    with pytest.raises(MarketCheckError) as excinfo:
        check_real_market(modified, instruments, config)
    assert "earnings_spacing" in str(excinfo.value)
    assert anchor_instrument in str(excinfo.value)


def test_report_has_vol_and_corr_for_every_regime_and_family(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    report = check_real_market(market, instruments, config)

    regimes = {span.regime for span in market.schedule}
    assert regimes == set(Regime)
    for metric_name in ("vol", "corr"):
        entries = [m for m in report.metrics if m.metric == metric_name]
        assert {(m.regime, m.family) for m in entries} == {
            (regime, family) for regime in regimes for family in Family
        }
        assert all(m.passed for m in entries)
        assert all(m.target == m.realised for m in entries)


def test_report_to_dict_round_trips_through_json_dumps(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    report = check_real_market(market, instruments, config)
    payload = json.dumps(report.to_dict())
    decoded = json.loads(payload)

    assert decoded["seed"] == "R1"
    assert len(decoded["metrics"]) == len(report.metrics)
