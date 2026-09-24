"""Tests for the equities adapter and the shared adapter helpers it exercises."""

import math

import numpy as np
import pytest

from pm_traitbench.engine.adapters.base import leg_side, pnl_unit, relative_move
from pm_traitbench.engine.adapters.equities import EquitiesAdapter
from pm_traitbench.engine.constants import TRAILING_HIGH_DAYS
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
)
from pm_traitbench.errors import EngineError
from pm_traitbench.market.levels import log_grid_step
from pm_traitbench.tables.schema import Leg, Rule

_HORIZON = 10


def _exclusion_rule(sector: str) -> Rule:
    return Rule(
        pm_id="pm_001",
        rule_id="r_99",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="exclusion",
        field="sector",
        op=Op.NE,
        level=sector,
        unit=None,
        window=1,
        action=Action.EXCLUDE,
        text=f"nothing in the {sector} sector",
    )


def _make_position(
    *,
    series: Series,
    side: Side = Side.BUY,
    instrument_id: str = "EQ-0001",
    entry_t: int = 0,
    entry_level: float = 100.0,
    target_level: float = 110.0,
) -> Position:
    leg = Leg(instrument_id=instrument_id, tenor=None, side=side, weight=1.0)
    return Position(
        trade_idea_id="ti_001",
        expression=Expression.OUTRIGHT,
        instrument_id=instrument_id,
        legs=(leg,),
        series=series,
        side=side,
        entry_t=entry_t,
        entry_level=entry_level,
        target_level=target_level,
        stop_level=90.0,
        sd_h_at_entry=1.0,
        forecast=105.0,
        size_pct_book=2.0,
        original_size_pct_book=2.0,
        conviction=1,
        size_rank=1,
        triggers_fired=1,
        consumed_rule_ids=frozenset(),
        run_counters=(),
        size_changed_t=0,
    )


def test_universe_drops_excluded_sector(fixture_instruments) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    instruments = {i.instrument_id: i for i in fixture_instruments}
    result = adapter.universe(instruments, [_exclusion_rule("sector_02")])
    assert result == ("EQ-0001", "EQ-0002", "EQ-0003")


def test_universe_keeps_every_equity_without_an_exclusion(fixture_instruments) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    instruments = {i.instrument_id: i for i in fixture_instruments}
    result = adapter.universe(instruments, [])
    assert result == ("EQ-0001", "EQ-0002", "EQ-0003", "EQ-0004")


def test_build_legs_outright_is_one_leg_weight_one() -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    legs = adapter.build_legs(
        Expression.OUTRIGHT, "EQ-0001", None, 0, (), frozenset(), np.random.default_rng(0)
    )
    assert legs == (LegRef("EQ-0001", None, 1.0),)


def test_build_legs_pair_picks_same_sector_partner_not_candidate(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    universe = ("EQ-0001", "EQ-0002", "EQ-0003")
    t = 40

    legs = adapter.build_legs(
        Expression.PAIR, "EQ-0001", fixture_view, t, universe, frozenset(), np.random.default_rng(0)
    )
    assert legs is not None
    assert legs[0] == LegRef("EQ-0001", None, 1.0)

    same_sector = [iid for iid in universe if iid != "EQ-0001"]
    expected_partner = min(
        same_sector,
        key=lambda iid: (
            fixture_view.trailing_move(adapter.outright_series(iid), t, _HORIZON),
            iid,
        ),
    )
    assert legs[1] == LegRef(expected_partner, None, -1.0)


def test_build_legs_pair_excludes_held_instruments(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    universe = ("EQ-0001", "EQ-0002", "EQ-0003")
    t = 40
    same_sector = [iid for iid in universe if iid != "EQ-0001"]
    expected_partner = min(
        same_sector,
        key=lambda iid: (
            fixture_view.trailing_move(adapter.outright_series(iid), t, _HORIZON),
            iid,
        ),
    )

    legs = adapter.build_legs(
        Expression.PAIR,
        "EQ-0001",
        fixture_view,
        t,
        universe,
        frozenset({expected_partner}),
        np.random.default_rng(0),
    )
    assert legs is not None
    assert legs[1].instrument_id != expected_partner
    assert legs[1].instrument_id in same_sector


def test_build_legs_pair_returns_none_without_a_partner(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    legs = adapter.build_legs(
        Expression.PAIR,
        "EQ-0001",
        fixture_view,
        40,
        ("EQ-0001",),
        frozenset(),
        np.random.default_rng(0),
    )
    assert legs is None


def test_pnl_unit_buy_and_sell() -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    series = adapter.outright_series("EQ-0001")
    buy = _make_position(series=series, side=Side.BUY, entry_level=100.0)
    sell = _make_position(series=series, side=Side.SELL, entry_level=100.0)
    assert pnl_unit(buy, 105.0) == pytest.approx(5.0)
    assert pnl_unit(sell, 105.0) == pytest.approx(-5.0)


def test_leg_side_pair_buy_gives_buy_then_sell() -> None:
    assert leg_side(1, 1, 1.0, 1) == Side.BUY
    assert leg_side(1, 1, -1.0, 1) == Side.SELL


def test_position_fields_has_every_field_and_target_hit_flips(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    series = adapter.outright_series("EQ-0001")
    levels = [fixture_view.level(series, t) for t in range(fixture_view.n_days)]
    target_level = sum(levels) / len(levels)
    below_t = next(t for t, lv in enumerate(levels) if lv < target_level)
    above_t = next(t for t, lv in enumerate(levels) if lv >= target_level)

    pos = _make_position(
        series=series, side=Side.BUY, entry_level=levels[0], target_level=target_level
    )
    state = PmState(pm_id="pm_001", positions=(pos,), next_idea=2, next_rule=1)

    fields_below = adapter.position_fields(
        pos, fixture_view, below_t, state, pnl_unit(pos, levels[below_t])
    )
    fields_above = adapter.position_fields(
        pos, fixture_view, above_t, state, pnl_unit(pos, levels[above_t])
    )

    assert set(fields_below) == EquitiesAdapter.FIELDS
    assert fields_below["target_hit"] == 0
    assert fields_above["target_hit"] == 1
    assert fields_below["sector"] == "sector_01"
    assert fields_below["n_positions"] == 1
    assert fields_below["triggers_fired"] == 1


def test_size_and_risk(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    legs = (LegRef("EQ-0001", None, 1.0),)
    size, risk = adapter.size_and_risk(5.0, legs, fixture_view, 0, 1e8)
    assert size == pytest.approx(5.0)
    assert risk == pytest.approx(5e6)


def test_anchors_round_level_and_trailing_extreme_by_side(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    series = adapter.outright_series("EQ-0001")
    entry_level = fixture_view.level(series, 0)
    buy = _make_position(series=series, side=Side.BUY, entry_level=entry_level)
    sell = _make_position(series=series, side=Side.SELL, entry_level=entry_level)
    t_range, t_risk_off = 10, 45

    buy_range = adapter.anchors(buy, fixture_view, t_range)
    buy_risk_off = adapter.anchors(buy, fixture_view, t_risk_off)
    assert len(buy_range) == 3
    assert len(buy_risk_off) == 2

    level_now = fixture_view.level(series, t_range)
    expected_round_level = adapter.round_step(series, level_now)
    assert buy_range[1] == pytest.approx(expected_round_level)

    # buy: target sits above entry (adverse_dir < 0), so the anchor is the trailing high.
    assert buy_range[2] == pytest.approx(
        fixture_view.trailing_high(series, t_range, TRAILING_HIGH_DAYS)
    )
    # sell: target sits below entry (adverse_dir > 0), so the anchor is the trailing low.
    sell_range = adapter.anchors(sell, fixture_view, t_range)
    assert sell_range[2] == pytest.approx(
        fixture_view.trailing_low(series, t_range, TRAILING_HIGH_DAYS)
    )


def test_relative_move_matches_view_computation_and_is_nonzero(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    series = adapter.outright_series("EQ-0001")
    entry_level = fixture_view.level(series, 0)
    buy = _make_position(series=series, side=Side.BUY, entry_level=entry_level)
    sell = _make_position(series=series, side=Side.SELL, entry_level=entry_level)
    t = 40

    peers = adapter.peer_ids("EQ-0001", fixture_view.instruments)
    peer_move = fixture_view.peer_move(peers, adapter.outright_series, t, _HORIZON)
    own_move = fixture_view.trailing_move(series, t, _HORIZON)
    expected = series.bullish_sign * buy.side_sign * (own_move - peer_move)

    move_buy = relative_move(fixture_view, adapter, buy, t, _HORIZON)
    move_sell = relative_move(fixture_view, adapter, sell, t, _HORIZON)

    assert move_buy == pytest.approx(expected)
    assert move_buy != pytest.approx(0.0)
    assert move_sell == pytest.approx(-move_buy)


def test_stop_distance_reads_pnl_from_entry_rule() -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    series = adapter.outright_series("EQ-0001")
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
        text="stop at -12% from entry",
    )
    assert adapter.stop_distance(rule, series) == pytest.approx(12.0)


def test_stop_distance_raises_for_a_field_it_does_not_serve() -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    series = adapter.outright_series("EQ-0001")
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


def test_forms_returns_outright_and_pair_for_any_sub_style() -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    assert adapter.forms("value") == (Expression.OUTRIGHT, Expression.PAIR)
    assert EquitiesAdapter("growth", _HORIZON).forms("growth") == (
        Expression.OUTRIGHT,
        Expression.PAIR,
    )


def test_peer_ids_and_label(fixture_instruments) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    instruments = {i.instrument_id: i for i in fixture_instruments}
    assert adapter.peer_ids("EQ-0001", instruments) == ("EQ-0002", "EQ-0003")
    assert adapter.peer_label("EQ-0001") == "sector"


def test_leg_bullish_is_always_positive() -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    assert adapter.leg_bullish(LegRef("EQ-0001", None, 1.0)) == 1
    assert adapter.leg_bullish(LegRef("EQ-0001", None, -1.0)) == 1


def test_instrument_type_reads_view(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    assert adapter.instrument_type("EQ-0001", fixture_view) == InstrumentKind.EQUITY


def test_leg_price_recovers_raw_price(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    price = adapter.leg_price(LegRef("EQ-0001", None, 1.0), fixture_view, 0)
    expected = math.exp(fixture_view.raw_level("EQ-0001", None, 0) / 100.0)
    assert price == pytest.approx(expected)


def test_round_step_returns_a_round_level_near_the_price(fixture_view) -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    series = adapter.outright_series("EQ-0001")
    level = fixture_view.level(series, 0)
    rounded = adapter.round_step(series, level)

    price = math.exp(level / 100.0)
    step = log_grid_step(price)
    rounded_price = math.exp(rounded / 100.0)
    assert abs(rounded_price - price) <= step / 2 + 1e-6
