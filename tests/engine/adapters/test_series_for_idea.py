"""Tests for `series_for_idea`: rebuilding an idea's level series from its public legs."""

from datetime import date

import numpy as np
import pytest

from pm_traitbench.engine.adapters.base import Adapter, leg_side, series_for_idea
from pm_traitbench.engine.adapters.commodities import CommoditiesAdapter
from pm_traitbench.engine.adapters.equities import EquitiesAdapter
from pm_traitbench.engine.adapters.rates_credit import RatesCreditAdapter
from pm_traitbench.engine.series import LegRef
from pm_traitbench.enums import Expression, Side, StreetView
from pm_traitbench.tables.schema import Idea, Leg

_HORIZON = 10
_ENTRY_DATE = date(2026, 1, 5)


def _idea(*, expression: Expression, instrument_id: str, side: Side, legs: tuple[Leg, ...]) -> Idea:
    """A minimal, otherwise-arbitrary idea row carrying the fields `series_for_idea` reads."""
    return Idea(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        instrument_id=instrument_id,
        expression=expression,
        side=side,
        legs=legs,
        entry_date=_ENTRY_DATE,
        exit_date=None,
        entry_level=100.0,
        target_level=110.0,
        stop_level=90.0,
        thesis="thesis",
        outcome=None,
        own_signal=0.5,
        forecast=101.0,
        interval_lo=99.0,
        interval_hi=103.0,
        street_view_at_entry=StreetView.NEUTRAL,
        conflict=False,
        followed_street=None,
        conviction=3,
        size_rank=3,
    )


def _idea_legs(
    series_legs: tuple[LegRef, ...],
    side_sign: int,
    bullish_sign: int,
    adapter: Adapter,
    *,
    keeps_tenor: bool,
) -> tuple[Leg, ...]:
    """The idea's own legs, built exactly as `attempt_entry` builds `idea_legs` from `legs`."""
    position_legs = tuple(
        Leg(
            instrument_id=leg.instrument_id,
            tenor=leg.tenor,
            side=leg_side(side_sign, bullish_sign, leg.coeff, adapter.leg_bullish(leg)),
            weight=abs(leg.coeff),
        )
        for leg in series_legs
    )
    if keeps_tenor:
        return position_legs
    return tuple(leg.model_copy(update={"tenor": None}) for leg in position_legs)


def _assert_round_trip(
    *,
    adapter: Adapter,
    form: Expression,
    instrument_id: str,
    series_legs: tuple[LegRef, ...],
    side: Side,
    keeps_tenor: bool,
) -> None:
    side_sign = 1 if side == Side.BUY else -1
    if form == Expression.OUTRIGHT:
        bullish_sign = adapter.outright_series(instrument_id).bullish_sign
        expected = adapter.outright_series(instrument_id)
    else:
        bullish_sign = adapter.series(form, series_legs).bullish_sign
        expected = adapter.series(form, series_legs)

    idea_legs = _idea_legs(series_legs, side_sign, bullish_sign, adapter, keeps_tenor=keeps_tenor)
    idea = _idea(expression=form, instrument_id=instrument_id, side=side, legs=idea_legs)

    result = series_for_idea(idea, adapter)

    assert result.legs == expected.legs
    assert result.bullish_sign == expected.bullish_sign
    assert result.unit == expected.unit


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_equities_outright(fixture_view, side) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    legs = adapter.build_legs(
        Expression.OUTRIGHT, "EQ-0001", fixture_view, 0, (), frozenset(), np.random.default_rng(0)
    )
    assert legs is not None
    _assert_round_trip(
        adapter=adapter,
        form=Expression.OUTRIGHT,
        instrument_id="EQ-0001",
        series_legs=legs,
        side=side,
        keeps_tenor=False,
    )


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_equities_pair(fixture_view, side) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    universe = ("EQ-0001", "EQ-0002", "EQ-0003")
    legs = adapter.build_legs(
        Expression.PAIR,
        "EQ-0001",
        fixture_view,
        40,
        universe,
        frozenset(),
        np.random.default_rng(0),
    )
    assert legs is not None
    _assert_round_trip(
        adapter=adapter,
        form=Expression.PAIR,
        instrument_id="EQ-0001",
        series_legs=legs,
        side=side,
        keeps_tenor=False,
    )


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_rates_credit_sovereign_outright(fixture_view, side) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs = adapter.build_legs(
        Expression.OUTRIGHT, "RT-USD", fixture_view, 0, (), frozenset(), np.random.default_rng(0)
    )
    assert legs is not None
    _assert_round_trip(
        adapter=adapter,
        form=Expression.OUTRIGHT,
        instrument_id="RT-USD",
        series_legs=legs,
        side=side,
        keeps_tenor=False,
    )


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_rates_credit_curve(fixture_view, side) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs = adapter.build_legs(
        Expression.CURVE, "RT-USD", fixture_view, 0, (), frozenset(), np.random.default_rng(0)
    )
    assert legs is not None
    _assert_round_trip(
        adapter=adapter,
        form=Expression.CURVE,
        instrument_id="RT-USD",
        series_legs=legs,
        side=side,
        keeps_tenor=True,
    )


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_rates_credit_issuer_outright(fixture_view, side) -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    legs = adapter.build_legs(
        Expression.OUTRIGHT,
        "CR-IG-001",
        fixture_view,
        0,
        (),
        frozenset(),
        np.random.default_rng(0),
    )
    assert legs is not None
    _assert_round_trip(
        adapter=adapter,
        form=Expression.OUTRIGHT,
        instrument_id="CR-IG-001",
        series_legs=legs,
        side=side,
        keeps_tenor=False,
    )


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_commodities_outright(fixture_view, side) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    legs = adapter.build_legs(
        Expression.OUTRIGHT, "CM-CRD", fixture_view, 0, (), frozenset(), np.random.default_rng(0)
    )
    assert legs is not None
    _assert_round_trip(
        adapter=adapter,
        form=Expression.OUTRIGHT,
        instrument_id="CM-CRD",
        series_legs=legs,
        side=side,
        keeps_tenor=False,
    )


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_commodities_calendar_spread(fixture_view, side) -> None:
    adapter = CommoditiesAdapter("curve_and_spread", _HORIZON)
    legs = adapter.build_legs(
        Expression.CALENDAR_SPREAD,
        "CM-CRD",
        fixture_view,
        0,
        (),
        frozenset(),
        np.random.default_rng(0),
    )
    assert legs is not None
    _assert_round_trip(
        adapter=adapter,
        form=Expression.CALENDAR_SPREAD,
        instrument_id="CM-CRD",
        series_legs=legs,
        side=side,
        keeps_tenor=True,
    )
