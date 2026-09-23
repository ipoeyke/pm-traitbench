"""Tests for the real-market instrument universe and its alignment helper."""

from datetime import date, timedelta

import pytest

from pm_traitbench.config import REAL_FILL_LIMIT, Config
from pm_traitbench.enums import Family, InstrumentKind
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.real.align import aligned_series
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.universe import build_real_universe, real_axis_dates

_FAMILY_KIND = {
    Family.EQUITIES: InstrumentKind.EQUITY,
    Family.CREDIT: InstrumentKind.CREDIT_ISSUER,
    Family.RATES: InstrumentKind.SOVEREIGN_CURVE,
    Family.COMMODITIES: InstrumentKind.COMMODITY,
    Family.FX: InstrumentKind.FX_PAIR,
}


def _build(config: Config, result):
    cache = RawCache.open(result.data_dir)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    return axis, build_real_universe(config, cache, axis)


def test_build_real_universe_returns_67_instruments_in_family_order(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments = _build(config, result)

    assert len(instruments) == 67
    families = [i.family for i in instruments]
    assert families[:40] == [Family.EQUITIES] * 40
    assert families[40:42] == [Family.CREDIT] * 2
    assert families[42:43] == [Family.RATES]
    assert families[43:58] == [Family.COMMODITIES] * 15
    assert families[58:67] == [Family.FX] * 9

    ids = [i.instrument_id for i in instruments]
    assert len(set(ids)) == 67
    assert ids[:40] == [f"EQ-R{i:03d}" for i in range(1, 41)]


def test_instrument_kinds_match_their_family(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments = _build(config, result)

    for inst in instruments:
        assert inst.kind == _FAMILY_KIND[inst.family]


def test_equity_betas_are_near_the_fixtures_known_slope(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments = _build(config, result)

    equities = {i.instrument_id: i for i in instruments if i.family == Family.EQUITIES}
    assert set(equities) == set(result.betas)
    for instrument_id, known_beta in result.betas.items():
        assert equities[instrument_id].beta == pytest.approx(known_beta, abs=0.05)


def test_credit_instruments_have_the_registry_duration_and_rating(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments = _build(config, result)

    credit = {i.instrument_id: i for i in instruments if i.family == Family.CREDIT}
    assert set(credit) == {"CR-R-IG-001", "CR-R-IG-002"}
    for inst in credit.values():
        assert inst.duration_years == 13.0


def test_commodities_use_the_configured_expiry_rule(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments = _build(config, result)

    commodities = [i for i in instruments if i.family == Family.COMMODITIES]
    assert len(commodities) == 15
    for inst in commodities:
        assert inst.expiry_rule == config.market.universe.expiry_rule


def test_derived_fx_pairs_are_present_with_the_quote_currency(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments = _build(config, result)

    fx = {i.instrument_id: i for i in instruments if i.family == Family.FX}
    usd_pairs = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"}
    assert set(fx) == {f"FX-{p}" for p in usd_pairs} | {"FX-EURGBP", "FX-EURJPY", "FX-AUDJPY"}
    assert fx["FX-EURGBP"].currency == "GBP"
    assert fx["FX-EURJPY"].currency == "JPY"
    assert fx["FX-AUDJPY"].currency == "JPY"


def test_real_axis_dates_maps_calendar_start_to_window_start_and_preserves_weekday() -> None:
    config = Config()
    spec = config.market.real.seeds["R1"]
    axis = build_axis(config.timeline(), config.market.burn_in_days)

    dates = real_axis_dates(spec, axis, config.calendar.start)

    assert len(dates) == axis.n_days
    idx = axis.index(config.calendar.start)
    assert dates[idx] == spec.window_start
    for axis_day, real_day in zip(axis.dates, dates, strict=True):
        assert axis_day.weekday() == real_day.weekday()


def test_aligned_series_forward_fills_a_gap_and_reports_the_longest_run() -> None:
    dates = [date(2018, 6, 4) + timedelta(days=i) for i in range(5)]
    values = {dates[0]: 1.0, dates[1]: 2.0, dates[3]: 4.0, dates[4]: 5.0}

    array, longest_run = aligned_series(values, dates, name="TEST")

    assert list(array) == [1.0, 2.0, 2.0, 4.0, 5.0]
    assert longest_run == 1


def test_aligned_series_fills_day_zero_from_a_value_before_the_window() -> None:
    dates = [date(2018, 6, 4) + timedelta(days=i) for i in range(3)]
    values = {date(2018, 6, 1): 9.0, dates[1]: 2.0, dates[2]: 3.0}

    array, longest_run = aligned_series(values, dates, name="TEST")

    assert list(array) == [9.0, 2.0, 3.0]
    assert longest_run == 1


def test_aligned_series_raises_when_the_fill_run_exceeds_the_limit() -> None:
    dates = [date(2018, 6, 4) + timedelta(days=i) for i in range(REAL_FILL_LIMIT + 2)]
    values = {dates[0]: 1.0}

    with pytest.raises(StageIOError, match="TICKER"):
        aligned_series(values, dates, name="TICKER")


def test_aligned_series_raises_when_day_zero_has_no_buffered_value() -> None:
    dates = [date(2018, 6, 4), date(2018, 6, 5)]
    values = {date(2018, 6, 6): 1.0}

    with pytest.raises(StageIOError, match="DGS2"):
        aligned_series(values, dates, name="DGS2")
