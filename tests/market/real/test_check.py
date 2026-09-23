"""Tests for the real-market data-quality check."""

import dataclasses
import json
from bisect import bisect_right
from datetime import timedelta

import numpy as np
import pytest

from pm_traitbench.config import REAL_FILL_LIMIT, Config
from pm_traitbench.enums import EventType, Family, InstrumentKind, Regime, Tenor
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


def test_negative_equity_price_fails_naming_positive_and_the_series(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    equity = next(i for i in instruments if i.instrument_id == "EQ-R001")
    prices = dict(market.output.prices)
    series = prices[equity.instrument_id].copy()
    series[5] = -1.0
    prices[equity.instrument_id] = series
    output = dataclasses.replace(market.output, prices=prices)
    modified = dataclasses.replace(market, output=output)

    # Structural metrics (positive included) are checked, and raise, before
    # any realised moment is computed, so this never reaches the vol/corr
    # log() calls that a bad price would otherwise poison.
    with pytest.raises(MarketCheckError) as excinfo:
        check_real_market(modified, instruments, config)
    message = str(excinfo.value)
    assert "positive:EQ-R001" in message
    assert "1 miss" in message


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


def test_no_earnings_rows_and_zero_earnings_count(fake_cache) -> None:
    """A real seed has no earnings feed, so its calendar draws no EARNINGS row."""
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    assert not any(row.event == EventType.EARNINGS for row in market.calendar)
    assert market.drawn_events[EventType.EARNINGS] == 0


def test_constant_rates_curve_over_a_span_fails_naming_flat_rates(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    instruments, market = _build(config, result)

    curve_inst = next(i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE)
    span = market.schedule[0]
    a = market.axis.index(span.date_start)
    b = bisect_right(market.axis.dates, span.date_end) - 1

    curves = dict(market.output.curves)
    key = (curve_inst.instrument_id, Tenor.Y10)
    series = curves[key].copy()
    # Flatten from the day before the span too, so the first change inside
    # the span (level[a] - level[a - 1]) is also zero, not just the rest.
    series[a - 1 : b + 1] = series[a - 1]
    curves[key] = series
    output = dataclasses.replace(market.output, curves=curves)
    modified = dataclasses.replace(market, output=output)

    with pytest.raises(MarketCheckError) as excinfo:
        check_real_market(modified, instruments, config)
    assert "flat:rates" in str(excinfo.value)


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
