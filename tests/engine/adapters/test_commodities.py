"""Tests for the commodities adapter."""

import math
from datetime import date

import numpy as np
import pytest

from pm_traitbench.engine.adapters.base import leg_side, pnl_unit, relative_move
from pm_traitbench.engine.adapters.commodities import CommoditiesAdapter
from pm_traitbench.engine.constants import CALENDAR_BACK_TENORS, TRAILING_HIGH_DAYS
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.enums import (
    Action,
    CommodityGroup,
    ExpiryRule,
    Expression,
    Family,
    InstrumentKind,
    Op,
    RuleScope,
    RuleSource,
    Side,
    Tenor,
)
from pm_traitbench.errors import EngineError
from pm_traitbench.market.constants import CONTRACT_MULTIPLIER
from pm_traitbench.market.levels import log_grid_step, nearest_level
from pm_traitbench.tables.schema import CurvePoint, Instrument, Leg, Rule

_HORIZON = 10


def _exclusion_rule(group: str) -> Rule:
    return Rule(
        pm_id="pm_001",
        rule_id="r_99",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="exclusion",
        field="commodity_group",
        op=Op.NE,
        level=group,
        unit=None,
        window=1,
        action=Action.EXCLUDE,
        text=f"nothing in the {group} group",
    )


def _make_position(
    *,
    series: Series,
    legs: tuple[Leg, ...],
    side: Side = Side.BUY,
    instrument_id: str = "CM-CRD",
    entry_t: int = 0,
    entry_level: float = 430.0,
    target_level: float = 460.0,
) -> Position:
    return Position(
        trade_idea_id="ti_001",
        expression=Expression.OUTRIGHT,
        instrument_id=instrument_id,
        legs=legs,
        series=series,
        side=side,
        entry_t=entry_t,
        entry_level=entry_level,
        target_level=target_level,
        stop_level=400.0,
        sd_h_at_entry=1.0,
        forecast=450.0,
        size_pct_book=2.0,
        original_size_pct_book=2.0,
        size_at_entry=2.0,
        conviction=1,
        size_rank=1,
        triggers_fired=1,
        consumed_rule_ids=frozenset(),
        run_counters=(),
        size_changed_t=0,
    )


def _single_day_view(price_m1: float, code: str = "CRD") -> MarketView:
    """A one-day view holding a single commodity's M1 curve level, for tests that need
    an exact price rather than the shared fixture's random walk."""
    day = date(2026, 1, 5)
    instrument_id = f"CM-{code}"
    instrument = Instrument(
        instrument_id=instrument_id,
        family=Family.COMMODITIES,
        kind=InstrumentKind.COMMODITY,
        name=code,
        currency="USD",
        sector=None,
        rating_band=None,
        commodity_group=CommodityGroup.ENERGY,
        duration_years=None,
        beta=None,
        expiry_rule=ExpiryRule.MONTHLY_THIRD_FRIDAY,
    )
    curve = CurvePoint(seed="X", date=day, curve_id=instrument_id, tenor=Tenor.M1, level=price_m1)
    return MarketView.build(
        seed="X",
        dates=(day,),
        instruments=[instrument],
        prices=[],
        curves=[curve],
        consensus=[],
        calendar=[],
        regimes=[],
    )


def test_construction_rejects_unknown_sub_style() -> None:
    with pytest.raises(EngineError):
        CommoditiesAdapter("trend", _HORIZON)


def test_universe_drops_excluded_group(fixture_instruments) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    instruments = {i.instrument_id: i for i in fixture_instruments}
    assert adapter.universe(instruments, []) == ("CM-CRD", "CM-GLD")
    assert adapter.universe(instruments, [_exclusion_rule("energy")]) == ("CM-GLD",)


def test_forms_by_sub_style() -> None:
    directional = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    curve_and_spread = CommoditiesAdapter("curve_and_spread", _HORIZON)
    assert directional.forms("commodity_futures_directional") == (Expression.OUTRIGHT,)
    assert curve_and_spread.forms("curve_and_spread") == (
        Expression.OUTRIGHT,
        Expression.CALENDAR_SPREAD,
    )


def test_build_legs_outright_is_m1() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    legs = adapter.build_legs(
        Expression.OUTRIGHT, "CM-CRD", None, 0, (), frozenset(), np.random.default_rng(0)
    )
    assert legs == (LegRef("CM-CRD", Tenor.M1, 1.0),)


def test_build_legs_calendar_spread_picks_back_tenor_uniformly_with_rng() -> None:
    adapter = CommoditiesAdapter("curve_and_spread", _HORIZON)
    rng = np.random.default_rng(0)
    expected_idx = int(np.random.default_rng(0).integers(len(CALENDAR_BACK_TENORS)))
    back_tenor = CALENDAR_BACK_TENORS[expected_idx]

    legs = adapter.build_legs(Expression.CALENDAR_SPREAD, "CM-CRD", None, 0, (), frozenset(), rng)
    assert legs == (LegRef("CM-CRD", Tenor.M1, 1.0), LegRef("CM-CRD", back_tenor, -1.0))


def test_series_outright_is_bullish_sign_positive() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    legs = (LegRef("CM-CRD", Tenor.M1, 1.0),)
    series = adapter.series(Expression.OUTRIGHT, legs)
    assert series.bullish_sign == 1
    assert series.unit == "pct"


def test_series_calendar_spread_level_is_100_times_ln_m1_minus_ln_back_tenor(
    fixture_view,
) -> None:
    adapter = CommoditiesAdapter("curve_and_spread", _HORIZON)
    back_tenor = Tenor.M5
    legs = (LegRef("CM-CRD", Tenor.M1, 1.0), LegRef("CM-CRD", back_tenor, -1.0))
    series = adapter.series(Expression.CALENDAR_SPREAD, legs)
    assert series.bullish_sign == 1

    t = 20
    m1 = fixture_view.raw_level("CM-CRD", Tenor.M1, t)
    m5 = fixture_view.raw_level("CM-CRD", back_tenor, t)
    assert fixture_view.level(series, t) == pytest.approx(m1 - m5)
    assert fixture_view.level(series, t) == pytest.approx(
        100.0 * (math.log(math.exp(m1 / 100.0)) - math.log(math.exp(m5 / 100.0)))
    )


def test_leg_side_calendar_spread_buy_gives_m1_buy_back_tenor_sell() -> None:
    adapter = CommoditiesAdapter("curve_and_spread", _HORIZON)
    m1_leg = LegRef("CM-CRD", Tenor.M1, 1.0)
    back_leg = LegRef("CM-CRD", Tenor.M5, -1.0)
    assert leg_side(1, 1, m1_leg.coeff, adapter.leg_bullish(m1_leg)) == Side.BUY
    assert leg_side(1, 1, back_leg.coeff, adapter.leg_bullish(back_leg)) == Side.SELL


def test_leg_side_outright_buy_gives_buy() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    leg = LegRef("CM-CRD", Tenor.M1, 1.0)
    series = adapter.series(Expression.OUTRIGHT, (leg,))
    assert leg_side(1, series.bullish_sign, leg.coeff, adapter.leg_bullish(leg)) == Side.BUY


def test_pnl_unit_outright_buy_is_positive_when_price_rises() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    buy = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=430.0)
    assert pnl_unit(buy, 440.0) == pytest.approx(10.0)
    assert pnl_unit(buy, 420.0) == pytest.approx(-10.0)


def test_position_fields_pnl_from_entry_equals_pnl_unit(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=430.0)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields = adapter.position_fields(pos, fixture_view, 0, state, -7.5)
    assert fields["pnl_from_entry"] == pytest.approx(-7.5)


def test_position_fields_days_to_expiry_is_10000_when_no_expiry() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    view = _single_day_view(72.0)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=0.0, entry_t=0)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields = adapter.position_fields(pos, view, 0, state, 0.0)
    assert fields["days_to_expiry"] == 10_000


def test_position_fields_days_to_expiry_reaches_zero_on_expiry_day(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    expiry_t = next(
        t for t in range(fixture_view.n_days) if fixture_view.is_expiry_day("CM-CRD", t)
    )
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields = adapter.position_fields(pos, fixture_view, expiry_t, state, 0.0)
    assert fields["days_to_expiry"] == 0


def test_position_fields_price_is_m1_price(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields = adapter.position_fields(pos, fixture_view, 0, state, 0.0)
    expected_price = math.exp(fixture_view.raw_level("CM-CRD", Tenor.M1, 0) / 100.0)
    assert fields["price"] == pytest.approx(expected_price)


def test_position_fields_commodity_group(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-GLD")
    leg = Leg(instrument_id="CM-GLD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(
        series=series,
        legs=(leg,),
        side=Side.BUY,
        instrument_id="CM-GLD",
        entry_level=entry_level,
    )
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields = adapter.position_fields(pos, fixture_view, 0, state, 0.0)
    assert fields["commodity_group"] == "precious"


def test_position_fields_has_every_field(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)
    fields = adapter.position_fields(pos, fixture_view, 0, state, 0.0)
    assert set(fields) == CommoditiesAdapter.FIELDS


def test_size_and_risk_crude() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    view = _single_day_view(72.0, code="CRD")
    legs = (LegRef("CM-CRD", Tenor.M1, 1.0),)
    size, risk = adapter.size_and_risk(5.0, legs, view, 0, 1e8)
    assert size == pytest.approx(69.0)
    assert risk == pytest.approx(5e6)


def test_size_and_risk_floors_at_one_contract() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    view = _single_day_view(72.0, code="CRD")
    legs = (LegRef("CM-CRD", Tenor.M1, 1.0),)
    size, _ = adapter.size_and_risk(0.0001, legs, view, 0, 1e8)
    assert size == pytest.approx(1.0)


def test_leg_price_is_the_tenor_curve_level(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    leg = LegRef("CM-CRD", Tenor.M3, 1.0)
    price = adapter.leg_price(leg, fixture_view, 0)
    raw = fixture_view.raw_level("CM-CRD", Tenor.M3, 0)
    assert price == pytest.approx(math.exp(raw / 100.0))


def test_instrument_type_reads_view(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    assert adapter.instrument_type("CM-CRD", fixture_view) == InstrumentKind.COMMODITY


def test_round_step_matches_equities_semantics(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    level = fixture_view.level(series, 0)
    rounded = adapter.round_step(series, level)
    price = math.exp(level / 100.0)
    expected = 100.0 * math.log(nearest_level(price, log_grid_step(price)))
    assert rounded == pytest.approx(expected)


def test_anchors_in_range_regime_include_rounded_m1_price(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    t_range = 10
    assert fixture_view.regime(t_range).value == "range"

    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)
    anchors = adapter.anchors(pos, fixture_view, t_range)
    assert len(anchors) == 3
    level_now = fixture_view.level(series, t_range)
    price = math.exp(level_now / 100.0)
    expected_round = 100.0 * math.log(nearest_level(price, log_grid_step(price)))
    assert anchors[1] == pytest.approx(expected_round)
    if pos.adverse_dir < 0:
        assert anchors[2] == pytest.approx(
            fixture_view.trailing_high(series, t_range, TRAILING_HIGH_DAYS)
        )
    else:
        assert anchors[2] == pytest.approx(
            fixture_view.trailing_low(series, t_range, TRAILING_HIGH_DAYS)
        )


def test_anchors_outside_range_has_no_round_level(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    t_risk_off = 45
    assert fixture_view.regime(t_risk_off).value == "risk_off"

    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)
    anchors = adapter.anchors(pos, fixture_view, t_risk_off)
    assert len(anchors) == 2


def test_peer_ids_same_commodity_group_includes_self(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    assert adapter.peer_ids("CM-CRD", fixture_view.instruments) == ("CM-CRD",)
    assert adapter.peer_label("CM-CRD", fixture_view.instruments) == "energy"


def test_relative_move_compares_against_same_group(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)
    t = 40

    own_move = fixture_view.trailing_move(series, t, _HORIZON)
    peer_move = fixture_view.trailing_move(series, t, _HORIZON)
    expected = series.bullish_sign * pos.side_sign * (own_move - peer_move)
    assert relative_move(fixture_view, adapter, pos, t, _HORIZON) == pytest.approx(expected)


def test_stop_distance_reads_pnl_from_entry_field() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    rule = Rule(
        pm_id="pm_001",
        rule_id="r_02",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="stop_loss",
        field="pnl_from_entry",
        op=Op.LE,
        level=-10.0,
        unit="pct",
        window=1,
        action=Action.EXIT,
        text="stop at a 10pct adverse move",
    )
    assert adapter.stop_distance(rule, series) == pytest.approx(10.0)


def test_stop_distance_raises_for_a_field_it_does_not_serve() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    rule = Rule(
        pm_id="pm_001",
        rule_id="r_02",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="stop_loss",
        field="adverse_yield_move_bp",
        op=Op.GE,
        level=20.0,
        unit="bp",
        window=1,
        action=Action.EXIT,
        text="stop rule",
    )
    with pytest.raises(EngineError):
        adapter.stop_distance(rule, series)


def test_leg_bullish_is_always_positive() -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    assert adapter.leg_bullish(LegRef("CM-CRD", Tenor.M1, 1.0)) == 1
    assert adapter.leg_bullish(LegRef("CM-CRD", Tenor.M5, -1.0)) == 1


def test_roll_legs_outright_gives_m1_to_m2_levels(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)

    pairs = adapter.roll_legs(pos, fixture_view, 5)
    assert len(pairs) == 1
    (current, level_current), (rolled, level_rolled) = pairs[0]
    assert current == LegRef("CM-CRD", Tenor.M1, 1.0)
    assert rolled == LegRef("CM-CRD", Tenor.M2, 1.0)
    assert level_current == pytest.approx(fixture_view.raw_level("CM-CRD", Tenor.M1, 5))
    assert level_rolled == pytest.approx(fixture_view.raw_level("CM-CRD", Tenor.M2, 5))


def test_roll_legs_calendar_spread_gives_two_pairs(fixture_view) -> None:
    adapter = CommoditiesAdapter("curve_and_spread", _HORIZON)
    back_tenor = Tenor.M5
    series_legs = (LegRef("CM-CRD", Tenor.M1, 1.0), LegRef("CM-CRD", back_tenor, -1.0))
    series = adapter.series(Expression.CALENDAR_SPREAD, series_legs)
    legs = (
        Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0),
        Leg(instrument_id="CM-CRD", tenor=back_tenor, side=Side.SELL, weight=1.0),
    )
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=legs, side=Side.BUY, entry_level=entry_level)

    pairs = adapter.roll_legs(pos, fixture_view, 5)
    assert len(pairs) == 2
    (m1_now, m1_level), (m2_now, m2_level) = pairs[0]
    (mk_now, mk_level), (mk1_now, mk1_level) = pairs[1]
    assert m1_now == LegRef("CM-CRD", Tenor.M1, 1.0)
    assert m2_now == LegRef("CM-CRD", Tenor.M2, 1.0)
    assert mk_now == LegRef("CM-CRD", back_tenor, -1.0)
    assert mk1_now == LegRef("CM-CRD", Tenor.M6, -1.0)
    assert m1_level == pytest.approx(fixture_view.raw_level("CM-CRD", Tenor.M1, 5))
    assert m2_level == pytest.approx(fixture_view.raw_level("CM-CRD", Tenor.M2, 5))
    assert mk_level == pytest.approx(fixture_view.raw_level("CM-CRD", back_tenor, 5))
    assert mk1_level == pytest.approx(fixture_view.raw_level("CM-CRD", Tenor.M6, 5))


def test_roll_shift_outright_equals_100_times_ln_m2_minus_ln_m1(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = adapter.outright_series("CM-CRD")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)

    t = 5
    m1 = fixture_view.raw_level("CM-CRD", Tenor.M1, t)
    m2 = fixture_view.raw_level("CM-CRD", Tenor.M2, t)
    assert adapter.roll_shift(pos, fixture_view, t) == pytest.approx(m2 - m1)


def test_roll_shift_calendar_spread_matches_two_leg_formula(fixture_view) -> None:
    adapter = CommoditiesAdapter("curve_and_spread", _HORIZON)
    back_tenor = Tenor.M5
    series_legs = (LegRef("CM-CRD", Tenor.M1, 1.0), LegRef("CM-CRD", back_tenor, -1.0))
    series = adapter.series(Expression.CALENDAR_SPREAD, series_legs)
    legs = (
        Leg(instrument_id="CM-CRD", tenor=Tenor.M1, side=Side.BUY, weight=1.0),
        Leg(instrument_id="CM-CRD", tenor=back_tenor, side=Side.SELL, weight=1.0),
    )
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=legs, side=Side.BUY, entry_level=entry_level)

    t = 5
    m1 = fixture_view.raw_level("CM-CRD", Tenor.M1, t)
    m2 = fixture_view.raw_level("CM-CRD", Tenor.M2, t)
    mk = fixture_view.raw_level("CM-CRD", back_tenor, t)
    mk1 = fixture_view.raw_level("CM-CRD", Tenor.M6, t)
    expected = (m2 - mk1) - (m1 - mk)
    assert adapter.roll_shift(pos, fixture_view, t) == pytest.approx(expected)


def test_roll_shift_raises_past_m12(fixture_view) -> None:
    adapter = CommoditiesAdapter("commodity_futures_directional", _HORIZON)
    series = Series(legs=(LegRef("CM-CRD", Tenor.M12, 1.0),), bullish_sign=1, unit="pct")
    leg = Leg(instrument_id="CM-CRD", tenor=Tenor.M12, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)
    with pytest.raises(EngineError):
        adapter.roll_shift(pos, fixture_view, 5)


def test_contract_multiplier_lookup_uses_id_suffix_after_cm() -> None:
    assert "CRD" in CONTRACT_MULTIPLIER
