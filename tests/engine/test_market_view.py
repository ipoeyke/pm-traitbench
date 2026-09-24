"""Tests for the engine's in-memory market view."""

import math
import re
from datetime import date

import numpy as np
import pytest

from pm_traitbench.engine.constants import MIN_SD_DAYS, SD_FLOOR, TRAILING_SD_DAYS
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.enums import EventType, Family, InstrumentKind, Regime, StreetView, Tenor
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Instrument, Price, RegimeSpan


def _outright(instrument_id: str, tenor: Tenor | None = None) -> Series:
    return Series(legs=(LegRef(instrument_id, tenor, 1.0),), bullish_sign=1, unit="pct")


def _price_row(fixture_market: dict, instrument_id: str, day: date) -> Price:
    return next(
        p for p in fixture_market["prices"] if p.instrument_id == instrument_id and p.date == day
    )


def _curve_level(fixture_market: dict, curve_id: str, tenor: Tenor, day: date) -> float:
    return next(
        c.level
        for c in fixture_market["curves"]
        if c.curve_id == curve_id and c.tenor == tenor and c.date == day
    )


def _oracle_levels(fixture_market: dict, instrument_id: str) -> np.ndarray:
    """Independent 100*ln(price) series for an instrument, built straight from the price rows."""
    by_date = {
        p.date: p.price for p in fixture_market["prices"] if p.instrument_id == instrument_id
    }
    return np.array([100.0 * math.log(by_date[day]) for day in fixture_market["dates"]])


def _oracle_sd_h(levels: np.ndarray, t: int, h: int) -> float:
    """Recompute sd_h's specified formula from a level series, independent of MarketView."""
    n_days = len(levels)
    lo = max(0, t - TRAILING_SD_DAYS)
    moves = np.diff(levels[lo : t + 1])
    if len(moves) < MIN_SD_DAYS:
        hi = min(TRAILING_SD_DAYS, n_days - 1)
        moves = np.diff(levels[: hi + 1])
    sd = float(np.std(moves, ddof=1)) if len(moves) >= 2 else 0.0
    return max(sd * math.sqrt(h), SD_FLOOR)


def test_raw_level_equity_is_100_times_log_price(fixture_view: MarketView, fixture_market: dict):
    day = fixture_market["dates"][5]
    price = _price_row(fixture_market, "EQ-0001", day).price
    assert fixture_view.raw_level("EQ-0001", None, 5) == pytest.approx(100 * math.log(price))


def test_raw_level_commodity_futures_tenor_is_100_times_log_curve_level(
    fixture_view: MarketView, fixture_market: dict
):
    day = fixture_market["dates"][7]
    level = _curve_level(fixture_market, "CM-CRD", Tenor.M1, day)
    assert fixture_view.raw_level("CM-CRD", Tenor.M1, 7) == pytest.approx(100 * math.log(level))


def test_raw_level_sovereign_tenor_is_100_times_yield(
    fixture_view: MarketView, fixture_market: dict
):
    day = fixture_market["dates"][12]
    level = _curve_level(fixture_market, "RT-USD", Tenor.Y10, day)
    assert fixture_view.raw_level("RT-USD", Tenor.Y10, 12) == pytest.approx(100 * level)


def test_raw_level_credit_is_spread_bp(fixture_view: MarketView, fixture_market: dict):
    day = fixture_market["dates"][9]
    spread = _price_row(fixture_market, "CR-IG-001", day).spread_bp
    assert fixture_view.raw_level("CR-IG-001", None, 9) == pytest.approx(spread)


def test_raw_level_rejects_sovereign_curve_with_no_tenor(fixture_view: MarketView):
    with pytest.raises(EngineError):
        fixture_view.raw_level("RT-USD", None, 0)


def test_raw_level_rejects_equity_with_a_tenor(fixture_view: MarketView):
    with pytest.raises(EngineError):
        fixture_view.raw_level("EQ-0001", Tenor.Y10, 0)


def test_level_of_steepener_series_is_100_times_yield_spread(fixture_view: MarketView):
    series = Series(
        legs=(LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0)),
        bullish_sign=1,
        unit="bp",
    )
    t = 14
    y10 = fixture_view.raw_level("RT-USD", Tenor.Y10, t) / 100
    y2 = fixture_view.raw_level("RT-USD", Tenor.Y2, t) / 100
    assert fixture_view.level(series, t) == pytest.approx(100 * (y10 - y2))


@pytest.mark.parametrize(
    "t,h",
    [
        pytest.param(0, 5, id="lookahead-capped-at-n-days-minus-1"),
        pytest.param(3, 5, id="fallback-3-moves-below-min-sd-days"),
        pytest.param(10, 5, id="trailing-10-moves"),
        pytest.param(59, 5, id="trailing-60-days-59-moves"),
    ],
)
def test_sd_h_matches_oracle_computed_from_price_rows(
    fixture_view: MarketView, fixture_market: dict, t: int, h: int
):
    levels = _oracle_levels(fixture_market, "EQ-0001")
    expected = _oracle_sd_h(levels, t, h)
    assert fixture_view.sd_h(_outright("EQ-0001"), t, h) == pytest.approx(expected)


def test_trailing_move_clamps_lookback_to_day_zero(fixture_view: MarketView):
    series = _outright("EQ-0001")
    expected = fixture_view.level(series, 3) - fixture_view.level(series, 0)
    assert fixture_view.trailing_move(series, 3, 20) == pytest.approx(expected)


def test_forward_move_clamps_lookahead_to_last_day(fixture_view: MarketView):
    series = _outright("EQ-0001")
    assert fixture_view.forward_days(55, 20) == 4
    expected = fixture_view.level(series, 59) - fixture_view.level(series, 55)
    assert fixture_view.forward_move(series, 55, 20) == pytest.approx(expected)


def test_trailing_high_and_low_match_oracle_over_price_rows(
    fixture_view: MarketView, fixture_market: dict
):
    levels = _oracle_levels(fixture_market, "EQ-0001")
    t, days = 45, 20
    lo = max(0, t - days)
    series = _outright("EQ-0001")
    assert fixture_view.trailing_high(series, t, days) == pytest.approx(
        float(np.max(levels[lo : t + 1]))
    )
    assert fixture_view.trailing_low(series, t, days) == pytest.approx(
        float(np.min(levels[lo : t + 1]))
    )


def test_street_view_is_none_for_instrument_with_no_consensus(fixture_view: MarketView):
    assert fixture_view.street_view("RT-USD", 10) is None
    assert fixture_view.positioning("RT-USD", 10) is None


def test_street_view_flips_sign_at_day_thirty(fixture_view: MarketView):
    # t=25 and t=45 share the same phase in the fixture's oscillating consensus
    # pattern; the day-30 consensus flip negates the sign for t=45 but not t=25.
    assert fixture_view.street_view("EQ-0001", 25) == StreetView.OVERWEIGHT
    assert fixture_view.street_view("EQ-0001", 45) == StreetView.UNDERWEIGHT


def test_events_on_day_includes_market_wide_event_for_every_instrument(
    fixture_view: MarketView, fixture_instruments: list[Instrument]
):
    for instrument in fixture_instruments:
        assert EventType.MACRO_PRINT in fixture_view.events_on(instrument.instrument_id, 25)


def test_event_types_for_eq_0001_is_earnings_only(fixture_view: MarketView):
    assert fixture_view.event_types("EQ-0001") == {EventType.EARNINGS}


def test_event_types_for_commodity_excludes_contract_expiry(fixture_view: MarketView):
    types = fixture_view.event_types("CM-CRD")
    assert EventType.CONTRACT_EXPIRY not in types
    assert EventType.INVENTORY_REPORT in types


def test_days_to_expiry_counts_down_to_zero_and_is_none_after_last_expiry(
    fixture_view: MarketView, fixture_market: dict
):
    expiry_dates = sorted(
        row.date
        for row in fixture_market["calendar"]
        if row.instrument_id == "CM-CRD" and row.event == EventType.CONTRACT_EXPIRY
    )
    dates = fixture_market["dates"]
    last_expiry_t = dates.index(expiry_dates[-1])

    assert fixture_view.days_to_expiry("CM-CRD", last_expiry_t) == 0
    assert fixture_view.is_expiry_day("CM-CRD", last_expiry_t)
    assert last_expiry_t + 1 < len(dates)
    assert fixture_view.days_to_expiry("CM-CRD", last_expiry_t + 1) is None

    first_expiry_t = dates.index(expiry_dates[0])
    assert fixture_view.days_to_expiry("CM-CRD", first_expiry_t - 1) == 1


def test_days_to_expiry_is_none_for_non_commodity(fixture_view: MarketView):
    assert fixture_view.days_to_expiry("EQ-0001", 0) is None


def test_regime_matches_the_fixture_spans(fixture_view: MarketView):
    assert fixture_view.regime(29) == Regime.RANGE
    assert fixture_view.regime(30) == Regime.RISK_OFF


def test_regime_raises_outside_every_span():
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    view = MarketView.build(
        seed="T",
        dates=dates,
        instruments=[],
        prices=[],
        curves=[],
        consensus=[],
        calendar=[],
        regimes=[RegimeSpan(seed="T", regime=Regime.RANGE, date_start=dates[0], date_end=dates[0])],
    )
    assert view.regime(0) == Regime.RANGE
    with pytest.raises(EngineError):
        view.regime(1)


def test_build_rejects_overlapping_regime_spans():
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    regimes = [
        RegimeSpan(seed="T", regime=Regime.RANGE, date_start=dates[0], date_end=dates[1]),
        RegimeSpan(seed="T", regime=Regime.RISK_OFF, date_start=dates[1], date_end=dates[2]),
    ]
    with pytest.raises(EngineError):
        MarketView.build(
            seed="T",
            dates=dates,
            instruments=[],
            prices=[],
            curves=[],
            consensus=[],
            calendar=[],
            regimes=regimes,
        )


def test_missing_price_for_listed_instrument_raises_engine_error():
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    instruments = [
        Instrument(
            instrument_id="EQ-0001",
            family=Family.EQUITIES,
            kind=InstrumentKind.EQUITY,
            name="Equity 0001",
            currency="USD",
            sector="sector_01",
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=1.0,
            expiry_rule=None,
        )
    ]
    prices = [
        Price(seed="T", date=dates[0], instrument_id="EQ-0001", price=100.0, spread_bp=None),
        Price(seed="T", date=dates[2], instrument_id="EQ-0001", price=101.0, spread_bp=None),
    ]
    expected_message = re.escape("instrument 'EQ-0001' has no price on 2026-01-06")
    with pytest.raises(EngineError, match=expected_message):
        MarketView.build(
            seed="T",
            dates=dates,
            instruments=instruments,
            prices=prices,
            curves=[],
            consensus=[],
            calendar=[],
            regimes=[],
        )


def test_peer_move_is_the_mean_trailing_move_of_the_peer_group(fixture_view: MarketView):
    ids = ["EQ-0001", "EQ-0002", "EQ-0003"]
    t, h = 40, 10
    expected = sum(fixture_view.trailing_move(_outright(i), t, h) for i in ids) / len(ids)
    assert fixture_view.peer_move(ids, _outright, t, h) == pytest.approx(expected)


def test_peer_move_is_zero_for_an_empty_peer_group(fixture_view: MarketView):
    assert fixture_view.peer_move([], _outright, 10, 5) == 0.0
