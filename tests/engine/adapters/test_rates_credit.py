"""Tests for the rates and credit adapter."""

import numpy as np
import pytest

from pm_traitbench.engine.adapters.base import leg_side, pnl_unit, relative_move
from pm_traitbench.engine.adapters.rates_credit import RatesCreditAdapter
from pm_traitbench.engine.constants import CURVE_PAIRS, DV01_PER_MILLION, TRAILING_HIGH_DAYS
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.enums import (
    Action,
    Expression,
    InstrumentKind,
    Op,
    RuleScope,
    RuleSource,
    Side,
    Tenor,
)
from pm_traitbench.errors import EngineError
from pm_traitbench.market.levels import SPREAD_STEP_BP, YIELD_STEP_PCT, nearest_level
from pm_traitbench.tables.schema import Leg, Rule

_HORIZON = 10


def _exclusion_rule(band: str) -> Rule:
    return Rule(
        pm_id="pm_001",
        rule_id="r_99",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="exclusion",
        field="rating_band",
        op=Op.NE,
        level=band,
        unit=None,
        window=1,
        action=Action.EXCLUDE,
        text=f"nothing rated {band}",
    )


def _make_position(
    *,
    series: Series,
    legs: tuple[Leg, ...],
    side: Side = Side.BUY,
    instrument_id: str = "RT-USD",
    entry_t: int = 0,
    entry_level: float = 400.0,
    target_level: float = 350.0,
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
        stop_level=430.0,
        sd_h_at_entry=1.0,
        forecast=380.0,
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


def test_construction_rejects_unknown_sub_style() -> None:
    with pytest.raises(EngineError):
        RatesCreditAdapter("junk_bonds", _HORIZON)


def test_universe_sovereign_is_the_one_curve(fixture_instruments) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    instruments = {i.instrument_id: i for i in fixture_instruments}
    assert adapter.universe(instruments, []) == ("RT-USD",)


def test_universe_credit_drops_excluded_band(fixture_instruments) -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    instruments = {i.instrument_id: i for i in fixture_instruments}
    assert adapter.universe(instruments, []) == ("CR-IG-001", "CR-IG-002")
    assert adapter.universe(instruments, [_exclusion_rule("BBB")]) == ("CR-IG-001",)


def test_forms_by_sub_style() -> None:
    sovereign = RatesCreditAdapter("sovereign_rates", _HORIZON)
    credit = RatesCreditAdapter("long_short_credit", _HORIZON)
    assert sovereign.forms("sovereign_rates") == (Expression.OUTRIGHT, Expression.CURVE)
    assert credit.forms("long_short_credit") == (Expression.OUTRIGHT,)


def test_build_legs_outright_rates_is_10y() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs = adapter.build_legs(
        Expression.OUTRIGHT, "RT-USD", None, 0, (), frozenset(), np.random.default_rng(0)
    )
    assert legs == (LegRef("RT-USD", Tenor.Y10, 1.0),)


def test_build_legs_outright_credit_has_no_tenor() -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    legs = adapter.build_legs(
        Expression.OUTRIGHT, "CR-IG-001", None, 0, (), frozenset(), np.random.default_rng(0)
    )
    assert legs == (LegRef("CR-IG-001", None, 1.0),)


def test_build_legs_curve_picks_pair_uniformly_with_rng() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    rng = np.random.default_rng(0)
    expected_idx = int(np.random.default_rng(0).integers(len(CURVE_PAIRS)))
    short_tenor, long_tenor = CURVE_PAIRS[expected_idx]

    legs = adapter.build_legs(Expression.CURVE, "RT-USD", None, 0, (), frozenset(), rng)
    assert legs == (
        LegRef("RT-USD", long_tenor, 1.0),
        LegRef("RT-USD", short_tenor, -1.0),
    )


def test_build_legs_curve_rejects_credit() -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    with pytest.raises(EngineError):
        adapter.build_legs(
            Expression.CURVE, "CR-IG-001", None, 0, (), frozenset(), np.random.default_rng(0)
        )


def test_series_outright_is_bearish_sign_negative() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs = (LegRef("RT-USD", Tenor.Y10, 1.0),)
    series = adapter.series(Expression.OUTRIGHT, legs)
    assert series.bullish_sign == -1
    assert series.unit == "bp"


def test_series_curve_level_is_100_times_10y_minus_2y(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs = (LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0))
    series = adapter.series(Expression.CURVE, legs)
    assert series.bullish_sign == 1

    t = 20
    y10 = fixture_view.raw_level("RT-USD", Tenor.Y10, t)
    y2 = fixture_view.raw_level("RT-USD", Tenor.Y2, t)
    assert fixture_view.level(series, t) == pytest.approx(y10 - y2)
    assert fixture_view.level(series, t) == pytest.approx(100.0 * (y10 / 100.0 - y2 / 100.0))


def test_leg_side_steepener_buy_gives_10y_sell_2y_buy() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    leg_10y = LegRef("RT-USD", Tenor.Y10, 1.0)
    leg_2y = LegRef("RT-USD", Tenor.Y2, -1.0)
    assert leg_side(1, 1, leg_10y.coeff, adapter.leg_bullish(leg_10y)) == Side.SELL
    assert leg_side(1, 1, leg_2y.coeff, adapter.leg_bullish(leg_2y)) == Side.BUY


def test_leg_side_outright_rates_buy_gives_buy() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    leg = LegRef("RT-USD", Tenor.Y10, 1.0)
    series = adapter.series(Expression.OUTRIGHT, (leg,))
    assert leg_side(1, series.bullish_sign, leg.coeff, adapter.leg_bullish(leg)) == Side.BUY


def test_leg_side_outright_credit_buy_gives_buy() -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    leg = LegRef("CR-IG-001", None, 1.0)
    series = adapter.series(Expression.OUTRIGHT, (leg,))
    assert leg_side(1, series.bullish_sign, leg.coeff, adapter.leg_bullish(leg)) == Side.BUY


def test_pnl_unit_outright_rates_buy_is_positive_when_yields_fall() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
    leg = Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.BUY, weight=1.0)
    buy = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=400.0)
    assert pnl_unit(buy, 390.0) == pytest.approx(10.0)
    assert pnl_unit(buy, 410.0) == pytest.approx(-10.0)


def test_position_fields_adverse_yield_move_is_positive_part_of_negative_pnl(
    fixture_view,
) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
    leg = Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.BUY, weight=1.0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=400.0)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields_adverse = adapter.position_fields(pos, fixture_view, 0, state, -10.0)
    fields_favourable = adapter.position_fields(pos, fixture_view, 0, state, 10.0)

    assert fields_adverse["adverse_yield_move_bp"] == pytest.approx(10.0)
    assert fields_adverse["adverse_spread_move_bp"] == pytest.approx(0.0)
    assert fields_favourable["adverse_yield_move_bp"] == pytest.approx(0.0)


def test_position_fields_pnl_from_entry_10bp_adverse_on_10y_is_minus_0_8(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
    leg = Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.BUY, weight=1.0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=400.0)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields = adapter.position_fields(pos, fixture_view, 0, state, -10.0)
    assert fields["pnl_from_entry"] == pytest.approx(-0.8)


def test_position_fields_credit_rating_band_and_spread_bp(fixture_view) -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    series = adapter.outright_series("CR-IG-001")
    leg = Leg(instrument_id="CR-IG-001", tenor=None, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(
        series=series,
        legs=(leg,),
        side=Side.BUY,
        instrument_id="CR-IG-001",
        entry_level=entry_level,
    )
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields = adapter.position_fields(pos, fixture_view, 0, state, -5.0)
    assert fields["rating_band"] == "AA"
    assert fields["spread_bp"] == pytest.approx(fixture_view.level(series, 0))
    assert fields["adverse_spread_move_bp"] == pytest.approx(5.0)
    assert fields["adverse_yield_move_bp"] == pytest.approx(0.0)


def test_position_fields_curve_rating_band_is_empty(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs_ref = (LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0))
    series = adapter.series(Expression.CURVE, legs_ref)
    legs = (
        Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.SELL, weight=1.0),
        Leg(instrument_id="RT-USD", tenor=Tenor.Y2, side=Side.BUY, weight=1.0),
    )
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=legs, side=Side.BUY, entry_level=entry_level)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields = adapter.position_fields(pos, fixture_view, 0, state, 0.0)
    assert fields["rating_band"] == ""


def test_position_fields_has_every_field(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
    leg = Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.BUY, weight=1.0)
    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=400.0)
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)
    fields = adapter.position_fields(pos, fixture_view, 0, state, -10.0)
    assert set(fields) == RatesCreditAdapter.FIELDS


def test_size_and_risk_10y(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs = (LegRef("RT-USD", Tenor.Y10, 1.0),)
    size, risk = adapter.size_and_risk(5.0, legs, fixture_view, 0, 1e8)
    assert size == pytest.approx(4000.0)
    assert risk == pytest.approx(5e6)


def test_size_and_risk_curve_uses_long_tenor_dv01(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs = (LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0))
    size, risk = adapter.size_and_risk(5.0, legs, fixture_view, 0, 1e8)
    assert size == pytest.approx(5e6 / 1e6 * DV01_PER_MILLION[Tenor.Y10])
    assert risk == pytest.approx(5e6)


def test_leg_risk_amount_follows_dv01_ratio_for_curve_short_leg(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    long_leg = LegRef("RT-USD", Tenor.Y10, 1.0)
    short_leg = LegRef("RT-USD", Tenor.Y2, -1.0)
    size, risk = adapter.size_and_risk(5.0, (long_leg, short_leg), fixture_view, 0, 1e8)

    assert adapter.leg_risk_amount(long_leg, size, fixture_view, risk) == pytest.approx(risk)
    expected_short_risk = size / DV01_PER_MILLION[Tenor.Y2] * 1e6
    assert adapter.leg_risk_amount(short_leg, size, fixture_view, risk) == pytest.approx(
        expected_short_risk
    )
    assert expected_short_risk != pytest.approx(risk)


def test_leg_risk_amount_credit_uses_duration(fixture_view) -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    leg = LegRef("CR-IG-001", None, 1.0)
    size, risk = adapter.size_and_risk(5.0, (leg,), fixture_view, 0, 1e8)
    expected = size / (6.0 * 100) * 1e6
    assert adapter.leg_risk_amount(leg, size, fixture_view, risk) == pytest.approx(expected)


def test_leg_price_tenor_is_percent_and_credit_is_spread_bp(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    leg = LegRef("RT-USD", Tenor.Y10, 1.0)
    price = adapter.leg_price(leg, fixture_view, 0)
    assert price == pytest.approx(fixture_view.raw_level("RT-USD", Tenor.Y10, 0) / 100.0)

    credit_adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    credit_leg = LegRef("CR-IG-001", None, 1.0)
    spread = credit_adapter.leg_price(credit_leg, fixture_view, 0)
    assert spread == pytest.approx(fixture_view.raw_level("CR-IG-001", None, 0))


def test_instrument_type_reads_view(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    assert adapter.instrument_type("RT-USD", fixture_view) == InstrumentKind.SOVEREIGN_CURVE


def test_round_step_tenor_rounds_to_quarter_point(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
    level = fixture_view.level(series, 0)
    rounded = adapter.round_step(series, level)
    assert rounded == pytest.approx(nearest_level(level / 100.0, YIELD_STEP_PCT) * 100.0)


def test_round_step_credit_rounds_to_10bp(fixture_view) -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    series = adapter.outright_series("CR-IG-001")
    level = fixture_view.level(series, 0)
    rounded = adapter.round_step(series, level)
    assert rounded == pytest.approx(nearest_level(level, SPREAD_STEP_BP))


def test_round_step_curve_rounds_to_25bp(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs = (LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0))
    series = adapter.series(Expression.CURVE, legs)
    level = fixture_view.level(series, 0)
    rounded = adapter.round_step(series, level)
    assert rounded == pytest.approx(nearest_level(level, 25.0))


def test_anchors_in_range_regime_include_rounded_slope_for_curve(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs_ref = (LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0))
    series = adapter.series(Expression.CURVE, legs_ref)
    legs = (
        Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.BUY, weight=1.0),
        Leg(instrument_id="RT-USD", tenor=Tenor.Y2, side=Side.SELL, weight=1.0),
    )
    entry_level = fixture_view.level(series, 0)
    t_range = 10
    assert fixture_view.regime(t_range).value == "range"

    pos = _make_position(series=series, legs=legs, side=Side.BUY, entry_level=entry_level)
    anchors = adapter.anchors(pos, fixture_view, t_range)
    assert len(anchors) == 3
    level_now = fixture_view.level(series, t_range)
    assert anchors[1] == pytest.approx(nearest_level(level_now, 25.0))
    if pos.adverse_dir < 0:
        assert anchors[2] == pytest.approx(
            fixture_view.trailing_high(series, t_range, TRAILING_HIGH_DAYS)
        )
    else:
        assert anchors[2] == pytest.approx(
            fixture_view.trailing_low(series, t_range, TRAILING_HIGH_DAYS)
        )


def test_anchors_in_range_regime_include_rounded_spread_for_credit(fixture_view) -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    series = adapter.outright_series("CR-IG-001")
    leg = Leg(instrument_id="CR-IG-001", tenor=None, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    t_range = 10
    assert fixture_view.regime(t_range).value == "range"

    pos = _make_position(
        series=series,
        legs=(leg,),
        side=Side.BUY,
        instrument_id="CR-IG-001",
        entry_level=entry_level,
    )
    anchors = adapter.anchors(pos, fixture_view, t_range)
    assert len(anchors) == 3
    level_now = fixture_view.level(series, t_range)
    assert anchors[1] == pytest.approx(nearest_level(level_now, SPREAD_STEP_BP))


def test_anchors_outside_range_has_no_round_level(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
    leg = Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.BUY, weight=1.0)
    entry_level = fixture_view.level(series, 0)
    t_risk_off = 45
    assert fixture_view.regime(t_risk_off).value == "risk_off"

    pos = _make_position(series=series, legs=(leg,), side=Side.BUY, entry_level=entry_level)
    anchors = adapter.anchors(pos, fixture_view, t_risk_off)
    assert len(anchors) == 2


def test_peer_ids_curve_is_itself(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    assert adapter.peer_ids("RT-USD", fixture_view.instruments) == ("RT-USD",)
    assert adapter.peer_label("RT-USD", fixture_view.instruments) == "the 10Y"


def test_peer_ids_credit_is_same_rating_band(fixture_instruments) -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    instruments = {i.instrument_id: i for i in fixture_instruments}
    assert adapter.peer_ids("CR-IG-001", instruments) == ("CR-IG-001",)
    assert adapter.peer_label("CR-IG-001", instruments) == "the AA band"


def test_relative_move_curve_compares_against_10y(fixture_view) -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    legs_ref = (LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0))
    series = adapter.series(Expression.CURVE, legs_ref)
    legs = (
        Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.BUY, weight=1.0),
        Leg(instrument_id="RT-USD", tenor=Tenor.Y2, side=Side.SELL, weight=1.0),
    )
    entry_level = fixture_view.level(series, 0)
    pos = _make_position(series=series, legs=legs, side=Side.BUY, entry_level=entry_level)
    t = 40

    ten_year_series = adapter.outright_series("RT-USD")
    own_move = fixture_view.trailing_move(series, t, _HORIZON)
    peer_move = fixture_view.trailing_move(ten_year_series, t, _HORIZON)
    expected = series.bullish_sign * pos.side_sign * (own_move - peer_move)

    assert relative_move(fixture_view, adapter, pos, t, _HORIZON) == pytest.approx(expected)


def test_stop_distance_sovereign_reads_adverse_yield_field() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
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
        text="stop at a 20bp adverse yield move",
    )
    assert adapter.stop_distance(rule, series) == pytest.approx(20.0)


def test_stop_distance_credit_reads_adverse_spread_field() -> None:
    adapter = RatesCreditAdapter("long_short_credit", _HORIZON)
    series = adapter.outright_series("CR-IG-001")
    rule = Rule(
        pm_id="pm_001",
        rule_id="r_02",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="stop_loss",
        field="adverse_spread_move_bp",
        op=Op.GE,
        level=30.0,
        unit="bp",
        window=1,
        action=Action.EXIT,
        text="stop at a 30bp adverse spread move",
    )
    assert adapter.stop_distance(rule, series) == pytest.approx(30.0)


def test_stop_distance_raises_for_a_field_it_does_not_serve() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
    rule = Rule(
        pm_id="pm_001",
        rule_id="r_02",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="stop_loss",
        field="pnl_from_entry",
        op=Op.LE,
        level=-12.0,
        unit="pct",
        window=1,
        action=Action.EXIT,
        text="stop rule",
    )
    with pytest.raises(EngineError):
        adapter.stop_distance(rule, series)


def test_leg_bullish_is_always_negative() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    assert adapter.leg_bullish(LegRef("RT-USD", Tenor.Y10, 1.0)) == -1
    assert adapter.leg_bullish(LegRef("RT-USD", Tenor.Y2, -1.0)) == -1
