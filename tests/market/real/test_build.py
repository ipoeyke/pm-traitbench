"""Tests for the real-market seed builder."""

from datetime import date, timedelta

import numpy as np
import pytest

from pm_traitbench.config import Config, RealSeedSpec
from pm_traitbench.enums import EventType, Family, InstrumentKind, Regime, Tenor
from pm_traitbench.market.axis import SimAxis, build_axis
from pm_traitbench.market.real.align import aligned_series
from pm_traitbench.market.real.build import build_seed, real_schedule
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.sources import REAL_INSTRUMENTS
from pm_traitbench.market.real.universe import build_real_universe, real_axis_dates
from pm_traitbench.market.seed import to_rows

_REGISTRY = {inst.instrument_id: inst for inst in REAL_INSTRUMENTS}


def _build(config: Config, result, seed: str = "R1"):
    cache = RawCache.open(result.data_dir)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = build_real_universe(config, cache, axis)
    market = build_seed(config, seed, instruments, cache, axis)
    return axis, instruments, market


def _same_window_second_seed_config() -> Config:
    """R1 plus a second real seed, R2, sharing R1's exact window and regime dates."""
    config = Config()
    r1 = config.market.real.seeds["R1"]
    r2 = r1.model_copy(update={"note": "same window as R1, for flip-only difference testing"})
    real = config.market.real.model_copy(update={"seeds": {"R1": r1, "R2": r2}})
    market = config.market.model_copy(update={"real": real})
    population = config.population.model_copy(update={"pilot_market_seeds": ("R1", "R2")})
    return config.model_copy(update={"market": market, "population": population})


def test_row_counts_match_the_registry_and_default_horizon(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, market = _build(config, result)
    rows = to_rows(market)

    assert len(rows.prices) == 66 * 260
    assert len(rows.curves) == (4 + 12 * 15) * 260
    assert len(rows.consensus) == 67 * 260
    assert len(rows.regimes) == 3


def test_equity_log_changes_match_the_real_series_exactly(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    axis, instruments, market = _build(config, result)
    dates = real_axis_dates(config.market.real.seeds["R1"], axis, config.calendar.start)

    for inst in instruments:
        if inst.family != Family.EQUITIES:
            continue
        reg = _REGISTRY[inst.instrument_id]
        raw, _ = aligned_series(cache.yahoo(reg.series), dates, name=inst.instrument_id)
        rebased = market.output.prices[inst.instrument_id]
        np.testing.assert_allclose(np.diff(np.log(rebased)), np.diff(np.log(raw)), atol=1e-12)


def test_commodity_m1_log_changes_match_the_real_series_exactly(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    axis, instruments, market = _build(config, result)
    dates = real_axis_dates(config.market.real.seeds["R1"], axis, config.calendar.start)

    for inst in instruments:
        if inst.family != Family.COMMODITIES:
            continue
        reg = _REGISTRY[inst.instrument_id]
        raw, _ = aligned_series(cache.yahoo(reg.series), dates, name=inst.instrument_id)
        rebased = market.output.prices[inst.instrument_id]
        np.testing.assert_allclose(np.diff(np.log(rebased)), np.diff(np.log(raw)), atol=1e-12)


def test_fx_usd_pair_log_changes_match_the_real_series_exactly(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    axis, instruments, market = _build(config, result)
    dates = real_axis_dates(config.market.real.seeds["R1"], axis, config.calendar.start)

    for reg in REAL_INSTRUMENTS:
        if reg.family != Family.FX:
            continue
        raw, _ = aligned_series(cache.fred(reg.series), dates, name=reg.instrument_id)
        rebased = market.output.prices[reg.instrument_id]
        np.testing.assert_allclose(np.diff(np.log(rebased)), np.diff(np.log(raw)), atol=1e-12)


def test_credit_spread_rebasing_is_a_constant_multiple_of_the_real_spread(fake_cache) -> None:
    # The fixture's independent random-walk yields can make the real AA/BBB-over-
    # Treasury difference cross zero, unlike genuine corporate-over-sovereign data,
    # so a scalar-multiple check is used instead of a log-difference one.
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    axis, instruments, market = _build(config, result)
    dates = real_axis_dates(config.market.real.seeds["R1"], axis, config.calendar.start)

    dgs20, _ = aligned_series(cache.fred("DGS20"), dates, name="DGS20")
    for inst in instruments:
        if inst.family != Family.CREDIT:
            continue
        reg = _REGISTRY[inst.instrument_id]
        yld, _ = aligned_series(cache.fred(reg.series), dates, name=inst.instrument_id)
        s_real_bp = 100 * (yld - dgs20)
        base = config.market.levels.credit_base_spread_bp[inst.rating_band]
        expected = s_real_bp * base / s_real_bp[0]
        np.testing.assert_allclose(market.output.spreads[inst.instrument_id], expected, atol=1e-9)


def test_day_zero_levels_match_configuration(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments, market = _build(config, result)

    lo, hi = config.market.levels.equity_price_range
    for inst in instruments:
        if inst.family == Family.EQUITIES:
            assert lo <= market.output.prices[inst.instrument_id][0] <= hi

    assert market.output.prices["CM-CRD"][0] == pytest.approx(72.0)
    assert market.output.prices["FX-EURUSD"][0] == pytest.approx(1.10)

    curve_inst = next(i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE)
    assert market.output.curves[(curve_inst.instrument_id, Tenor.Y10)][0] == pytest.approx(4.1)

    assert market.output.spreads["CR-R-IG-001"][0] == pytest.approx(50.0)


def test_fx_crosses_are_consistent_with_their_usd_pairs(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, market = _build(config, result)

    eurusd = market.output.prices["FX-EURUSD"]
    gbpusd = market.output.prices["FX-GBPUSD"]
    eurgbp = market.output.prices["FX-EURGBP"]
    np.testing.assert_allclose(eurgbp, eurusd / gbpusd, atol=1e-12)

    audusd = market.output.prices["FX-AUDUSD"]
    usdjpy = market.output.prices["FX-USDJPY"]
    audjpy = market.output.prices["FX-AUDJPY"]
    np.testing.assert_allclose(audjpy, audusd * usdjpy, atol=1e-12)


def test_curve_shape_preserves_real_tenor_differences_where_unfloored(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    axis, instruments, market = _build(config, result)
    dates = real_axis_dates(config.market.real.seeds["R1"], axis, config.calendar.start)

    curve_inst = next(i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE)
    y2_real, _ = aligned_series(cache.fred("DGS2"), dates, name="DGS2")
    y10_real, _ = aligned_series(cache.fred("DGS10"), dates, name="DGS10")
    y2 = market.output.curves[(curve_inst.instrument_id, Tenor.Y2)]
    y10 = market.output.curves[(curve_inst.instrument_id, Tenor.Y10)]

    floor = config.market.levels.yield_floor_pct
    unfloored = (y2 > floor) & (y10 > floor)
    assert unfloored.any()
    np.testing.assert_allclose((y2 - y10)[unfloored], (y2_real - y10_real)[unfloored], atol=1e-9)


def test_credit_price_falls_when_spread_and_five_year_yield_both_rise(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments, market = _build(config, result)

    curve_inst = next(i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE)
    y5 = market.output.curves[(curve_inst.instrument_id, Tenor.Y5)]

    for inst in instruments:
        if inst.family != Family.CREDIT:
            continue
        s = market.output.spreads[inst.instrument_id]
        p = market.output.prices[inst.instrument_id]
        found = False
        for t in range(1, len(s)):
            if s[t] > s[t - 1] and y5[t] > y5[t - 1]:
                assert p[t] < p[t - 1]
                found = True
                break
        assert found, f"no day with both spread and 5Y yield rising for {inst.instrument_id}"


def test_driver_has_unit_sd_and_is_zero_on_day_zero(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, market = _build(config, result)

    assert market.z[0] == 0.0
    assert np.std(market.z[1:]) == pytest.approx(1.0)


def test_real_schedule_maps_regime_starts_and_tiles_a_small_axis() -> None:
    axis = SimAxis(dates=tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(10)), n_burn=2)
    calendar_start = date(2020, 1, 3)
    spec = RealSeedSpec(
        window_start=date(2018, 6, 4),
        regime_starts=(
            (Regime.RANGE, date(2018, 6, 4)),
            (Regime.RISK_OFF, date(2018, 6, 6)),
            (Regime.RISK_ON, date(2018, 6, 8)),
        ),
        basis="design",
        note="unit test spec",
    )

    schedule = real_schedule(spec, axis, calendar_start, "S")

    assert [span.regime for span in schedule] == [Regime.RANGE, Regime.RISK_OFF, Regime.RISK_ON]
    assert schedule[0].date_start == calendar_start
    assert schedule[0].date_end == calendar_start + timedelta(days=1)
    assert schedule[1].date_start == calendar_start + timedelta(days=2)
    assert schedule[1].date_end == calendar_start + timedelta(days=3)
    assert schedule[2].date_start == calendar_start + timedelta(days=4)
    assert schedule[2].date_end == axis.dates[-1]


def test_schedule_dates_equal_mapped_regime_starts_and_tile_the_horizon(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    axis, _, market = _build(config, result)

    spec = config.market.real.seeds["R1"]
    offset = spec.window_start - config.calendar.start
    expected_starts = [regime_start - offset for _, regime_start in spec.regime_starts]

    schedule = market.schedule
    assert [span.date_start for span in schedule] == expected_starts
    assert schedule[0].date_start == axis.dates[axis.n_burn]
    for prev, nxt in zip(schedule, schedule[1:], strict=False):
        assert nxt.date_start == prev.date_end + timedelta(days=1)
    assert schedule[-1].date_end == axis.dates[-1]


def test_burn_in_maps_to_the_first_regime(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, market = _build(config, result)

    spec = config.market.real.seeds["R1"]
    first_regime = spec.regime_starts[0][0]
    assert market.path.regimes[0] == first_regime
    assert all(regime == first_regime for regime in market.path.regimes[: market.axis.n_burn])


def test_fills_reports_the_holiday_gap_for_fred_sourced_series(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, market = _build(config, result)

    assert market.fills["DGS20"] == 1
    assert market.fills["DGS10"] == 1
    assert market.fills["CR-R-IG-001"] == 1


def test_every_calendar_event_date_is_a_horizon_date(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    axis, _, market = _build(config, result)

    horizon_dates = set(axis.dates[axis.horizon])
    assert all(row.date in horizon_dates for row in market.calendar)


def test_build_seed_is_deterministic(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = build_real_universe(config, cache, axis)

    market1 = build_seed(config, "R1", instruments, cache, axis)
    market2 = build_seed(config, "R1", instruments, cache, axis)

    assert to_rows(market1) == to_rows(market2)


def test_two_real_seeds_with_the_same_window_differ_only_in_flip_rows(fake_cache) -> None:
    config = _same_window_second_seed_config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = build_real_universe(config, cache, axis)

    market_r1 = build_seed(config, "R1", instruments, cache, axis)
    market_r2 = build_seed(config, "R2", instruments, cache, axis)

    def _non_flip_rows(market):
        return sorted(
            (row.date, row.instrument_id or "", row.event, row.surprise, row.affected)
            for row in market.calendar
            if row.event != EventType.CONSENSUS_FLIP
        )

    assert _non_flip_rows(market_r1) == _non_flip_rows(market_r2)

    def _flip_rows(market):
        return {
            (row.date, row.instrument_id)
            for row in market.calendar
            if row.event == EventType.CONSENSUS_FLIP
        }

    assert _flip_rows(market_r1) != _flip_rows(market_r2)
