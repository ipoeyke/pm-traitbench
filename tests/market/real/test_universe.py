"""Tests for the real-market instrument universe and its alignment helper."""

from datetime import date, timedelta

import numpy as np
import pytest

from pm_traitbench.config import REAL_FILL_LIMIT, Config, RealSeedSpec
from pm_traitbench.enums import Family, InstrumentKind, RatingBand, Regime
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.axis import SimAxis, build_axis
from pm_traitbench.market.real.align import aligned_series
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.universe import (
    _pooled_log_returns,
    build_real_universe,
    real_axis_dates,
)

_FAMILY_KIND = {
    Family.EQUITIES: InstrumentKind.EQUITY,
    Family.CREDIT: InstrumentKind.CREDIT_ISSUER,
    Family.RATES: InstrumentKind.SOVEREIGN_CURVE,
    Family.COMMODITIES: InstrumentKind.COMMODITY,
    Family.FX: InstrumentKind.FX_PAIR,
}

_EXPECTED_ID_ORDER: tuple[str, ...] = (
    *(f"EQ-R{i:03d}" for i in range(1, 41)),
    "CR-R-IG-001",
    "CR-R-IG-002",
    "RT-USD",
    "CM-CRD", "CM-BRN", "CM-HOL", "CM-GSL",
    "CM-GLD", "CM-SLV", "CM-PLT", "CM-CPR",
    "CM-WHT", "CM-CRN", "CM-SOY", "CM-SGR", "CM-COF", "CM-CTN", "CM-CCO",
    "FX-EURUSD", "FX-GBPUSD", "FX-USDJPY", "FX-AUDUSD", "FX-USDCHF", "FX-USDCAD",
    "FX-EURGBP", "FX-EURJPY", "FX-AUDJPY",
)  # fmt: skip


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
    assert ids == list(_EXPECTED_ID_ORDER)


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


class _StubCache:
    """A minimal cache double exposing only `yahoo`, for testing the pooling
    helper directly without a fetched manifest.
    """

    def __init__(self, series: dict[date, float]) -> None:
        self._series = series

    def yahoo(self, ticker: str) -> dict[date, float]:
        return self._series


def _real_seed_spec(window_start: date, note: str) -> RealSeedSpec:
    return RealSeedSpec(
        window_start=window_start,
        regime_starts=(
            (Regime.RANGE, window_start),
            (Regime.RISK_OFF, window_start + timedelta(days=1)),
            (Regime.RISK_ON, window_start + timedelta(days=2)),
        ),
        basis="design",
        note=note,
    )


def test_pooled_log_returns_excludes_any_return_spanning_the_window_gap() -> None:
    axis = SimAxis(
        dates=(date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)), n_burn=1
    )
    calendar_start = date(2020, 1, 1)

    spec1 = _real_seed_spec(date(2018, 6, 4), "window one")
    spec2 = _real_seed_spec(spec1.window_start + timedelta(weeks=104), "window two")

    dates1 = real_axis_dates(spec1, axis, calendar_start)
    dates2 = real_axis_dates(spec2, axis, calendar_start)
    assert dates1[-1] < dates2[0]  # a genuine, non-adjacent gap between the windows

    # A deliberate, huge level jump between the windows: any return wrongly
    # spanning the gap would dwarf every legitimate within-window return.
    series1 = {d: 100.0 * 1.01**i for i, d in enumerate(dates1)}
    series2 = {d: 1_000_000.0 * 1.01**i for i, d in enumerate(dates2)}
    cache = _StubCache({**series1, **series2})

    returns = _pooled_log_returns(cache, "TICKER", {"S1": spec1, "S2": spec2}, axis, calendar_start)

    expected = np.concatenate(
        [
            np.diff(np.log(np.array(list(series1.values())))),
            np.diff(np.log(np.array(list(series2.values())))),
        ]
    )
    assert len(returns) == (len(dates1) - 1) + (len(dates2) - 1)
    np.testing.assert_allclose(returns, expected)


def _two_real_seed_config() -> Config:
    """A config referencing two real seeds: R1's default window plus a
    second, non-adjacent window 104 weeks later.
    """
    config = Config()
    r1 = config.market.real.seeds["R1"]
    r2 = r1.model_copy(
        update={
            "window_start": r1.window_start + timedelta(weeks=104),
            "regime_starts": tuple(
                (regime, start + timedelta(weeks=104)) for regime, start in r1.regime_starts
            ),
            "note": "second window for pooled-beta testing",
        }
    )
    return config.model_copy(
        update={
            "market": config.market.model_copy(
                update={
                    "real": config.market.real.model_copy(update={"seeds": {"R1": r1, "R2": r2}})
                }
            ),
            "population": config.population.model_copy(update={"pilot_market_seeds": ("R1", "R2")}),
        }
    )


def test_equity_beta_pools_across_two_non_adjacent_real_seeds(fake_cache) -> None:
    config = _two_real_seed_config()
    result = fake_cache(config)
    _, instruments = _build(config, result)

    equities = {i.instrument_id: i for i in instruments if i.family == Family.EQUITIES}
    assert set(equities) == set(result.betas)
    for instrument_id, known_beta in result.betas.items():
        assert equities[instrument_id].beta == pytest.approx(known_beta, abs=0.05)


def test_credit_instruments_have_the_registry_duration_rating_and_sector(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, instruments = _build(config, result)

    credit = {i.instrument_id: i for i in instruments if i.family == Family.CREDIT}
    assert set(credit) == {"CR-R-IG-001", "CR-R-IG-002"}
    for inst in credit.values():
        assert inst.duration_years == 13.0
        assert inst.sector == "sector_01"
    assert credit["CR-R-IG-001"].rating_band == RatingBand.AA
    assert credit["CR-R-IG-002"].rating_band == RatingBand.BBB


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
