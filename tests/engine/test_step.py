"""Tests for the daily step: `step`, `handle_triggers` and `handle_discretionary`."""

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.engine.adapters.commodities import CommoditiesAdapter
from pm_traitbench.engine.adapters.equities import EquitiesAdapter
from pm_traitbench.engine.biases import disposition as disposition_module
from pm_traitbench.engine.biases import loss_aversion as loss_aversion_module
from pm_traitbench.engine.biases.exit_deficiency import LATE_ROLL_FLAG
from pm_traitbench.engine.biases.loss_aversion import LossSideChoice
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.params import ParamSchedule
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.engine.step import PmContext, step
from pm_traitbench.enums import (
    Action,
    AssetClass,
    CommodityGroup,
    EventType,
    ExpiryRule,
    Expression,
    Family,
    InstrumentKind,
    Op,
    PositionAction,
    Regime,
    RuleResponse,
    RuleScope,
    RuleSource,
    Side,
    Tenor,
)
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import (
    CalendarEvent,
    CurvePoint,
    Instrument,
    Leg,
    Price,
    RegimeSpan,
    Rule,
    Trait,
)

_SEED = "STEP"
_SUB_STYLE = "equity_long_short"
_COMMODITY_SUB_STYLE = "commodity_futures_directional"


def _dates(n: int) -> tuple[date, ...]:
    start = date(2026, 1, 5)
    return tuple(start + timedelta(days=i) for i in range(n))


def _view(
    levels: Sequence[float], *, instrument_id: str = "EQ-A", sector: str = "sector_01"
) -> MarketView:
    """A single-equity market whose level on day `t` is exactly `levels[t]`."""
    dates = _dates(len(levels))
    instrument = Instrument(
        instrument_id=instrument_id,
        family=Family.EQUITIES,
        kind=InstrumentKind.EQUITY,
        name="Test equity",
        currency="USD",
        sector=sector,
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=1.0,
        expiry_rule=None,
    )
    prices = [
        Price(
            seed=_SEED,
            date=d,
            instrument_id=instrument_id,
            price=math.exp(lv / 100.0),
            spread_bp=None,
        )
        for d, lv in zip(dates, levels, strict=True)
    ]
    regimes = [RegimeSpan(seed=_SEED, regime=Regime.RANGE, date_start=dates[0], date_end=dates[-1])]
    return MarketView.build(
        seed=_SEED,
        dates=dates,
        instruments=[instrument],
        prices=prices,
        curves=[],
        consensus=[],
        calendar=[],
        regimes=regimes,
    )


def _commodity_view(
    m1_levels: Sequence[float],
    m2_levels: Sequence[float],
    expiry_t: int,
    *,
    instrument_id: str = "CM-CRD",
) -> MarketView:
    """A single-commodity market with M1/M2 curve levels controlled directly, and one expiry."""
    dates = _dates(len(m1_levels))
    instrument = Instrument(
        instrument_id=instrument_id,
        family=Family.COMMODITIES,
        kind=InstrumentKind.COMMODITY,
        name="Test commodity",
        currency="USD",
        sector=None,
        rating_band=None,
        commodity_group=CommodityGroup.ENERGY,
        duration_years=None,
        beta=None,
        expiry_rule=ExpiryRule.MONTHLY_THIRD_FRIDAY,
    )
    curves = []
    for t, d in enumerate(dates):
        curves.append(
            CurvePoint(
                seed=_SEED,
                date=d,
                curve_id=instrument_id,
                tenor=Tenor.M1,
                level=math.exp(m1_levels[t] / 100.0),
            )
        )
        curves.append(
            CurvePoint(
                seed=_SEED,
                date=d,
                curve_id=instrument_id,
                tenor=Tenor.M2,
                level=math.exp(m2_levels[t] / 100.0),
            )
        )
    calendar = [
        CalendarEvent(
            seed=_SEED,
            date=dates[expiry_t],
            instrument_id=instrument_id,
            event=EventType.CONTRACT_EXPIRY,
            surprise=None,
            affected="commodities",
        )
    ]
    regimes = [RegimeSpan(seed=_SEED, regime=Regime.RANGE, date_start=dates[0], date_end=dates[-1])]
    return MarketView.build(
        seed=_SEED,
        dates=dates,
        instruments=[instrument],
        prices=[],
        curves=curves,
        consensus=[],
        calendar=calendar,
        regimes=regimes,
    )


def _position(
    trade_idea_id: str,
    instrument_id: str,
    *,
    side: Side = Side.BUY,
    entry_t: int = 0,
    entry_level: float = 0.0,
    target_level: float = 100.0,
    stop_level: float = -100.0,
    sd_h_at_entry: float = 5.0,
    forecast: float = 8.0,
    size_pct_book: float = 5.0,
    original_size_pct_book: float | None = None,
    conviction: int = 3,
    size_rank: int = 3,
    triggers_fired: int = 0,
    consumed_rule_ids: frozenset[str] = frozenset(),
    run_counters: tuple[tuple[str, int], ...] = (),
    size_changed_t: int | None = None,
    rolled_offset: float = 0.0,
    roll_breached: bool = False,
    tenor: Tenor | None = None,
) -> Position:
    leg = Leg(instrument_id=instrument_id, tenor=tenor, side=side, weight=1.0)
    series = Series(legs=(LegRef(instrument_id, tenor, 1.0),), bullish_sign=1, unit="pct")
    return Position(
        trade_idea_id=trade_idea_id,
        expression=Expression.OUTRIGHT,
        instrument_id=instrument_id,
        legs=(leg,),
        series=series,
        side=side,
        entry_t=entry_t,
        entry_level=entry_level,
        target_level=target_level,
        stop_level=stop_level,
        sd_h_at_entry=sd_h_at_entry,
        forecast=forecast,
        size_pct_book=size_pct_book,
        original_size_pct_book=(
            original_size_pct_book if original_size_pct_book is not None else size_pct_book
        ),
        conviction=conviction,
        size_rank=size_rank,
        triggers_fired=triggers_fired,
        consumed_rule_ids=consumed_rule_ids,
        run_counters=run_counters,
        size_changed_t=size_changed_t if size_changed_t is not None else entry_t,
        rolled_offset=rolled_offset,
        roll_breached=roll_breached,
    )


def _stop_rule(pm_id: str, tid: str, rid: str, level: float, *, adverse_dir: int = -1) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id=rid,
        source=RuleSource.SELF,
        scope=RuleScope.IDEA,
        trade_idea_id=tid,
        param="stop",
        field="level",
        op=Op.GE if adverse_dir > 0 else Op.LE,
        level=level,
        unit="pct",
        window=1,
        action=Action.EXIT,
        text="stop",
    )


def _target_rule(pm_id: str, tid: str, rid: str, level: float, *, adverse_dir: int = -1) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id=rid,
        source=RuleSource.SELF,
        scope=RuleScope.IDEA,
        trade_idea_id=tid,
        param="target",
        field="level",
        op=Op.LE if adverse_dir > 0 else Op.GE,
        level=level,
        unit="pct",
        window=1,
        action=Action.TARGET,
        text="target",
    )


def _signpost_rule(pm_id: str, tid: str, rid: str, level: float, *, adverse_dir: int = -1) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id=rid,
        source=RuleSource.SELF,
        scope=RuleScope.IDEA,
        trade_idea_id=tid,
        param="signpost",
        field="level",
        op=Op.GE if adverse_dir > 0 else Op.LE,
        level=level,
        unit="pct",
        window=1,
        action=Action.SIGNPOST,
        text="signpost",
    )


def _rng_for(config: Config, pm_id: str) -> Callable[..., np.random.Generator]:
    def rng_for(purpose: str, *keys) -> np.random.Generator:
        return stream(config.seed.root, "engine", pm_id, purpose, *keys)

    return rng_for


def _traits_with(
    traits: Sequence[Trait], overrides: Mapping[str, tuple[float, bool]]
) -> tuple[Trait, ...]:
    updated = []
    for trait in traits:
        if trait.param in overrides:
            value, active = overrides[trait.param]
            trait = trait.model_copy(update={"value": value, "active": active})
        updated.append(trait)
    return tuple(updated)


def _ctx(persona, traits, pm_rules, adapter, universe, catalogue, config) -> PmContext:
    schedule = ParamSchedule.build(list(traits), (), config)
    return PmContext(
        persona=persona,
        traits=tuple(traits),
        pm_rules=tuple(pm_rules),
        adapter=adapter,
        universe=tuple(universe),
        catalogue=catalogue,
        config=config,
        schedule=schedule,
        rng_for=_rng_for(config, persona.pm_id),
    )


@pytest.fixture
def eq_parts(neutral_pm, catalogue):
    """A neutral equities PM's persona, traits, pm_rules, adapter, catalogue and config."""
    persona, traits, pm_rules = neutral_pm(AssetClass.EQUITIES, _SUB_STYLE)
    adapter = EquitiesAdapter(sub_style=_SUB_STYLE, horizon_days=20)
    config = Config()
    return persona, traits, pm_rules, adapter, catalogue, config


@pytest.fixture
def cm_parts(neutral_pm, catalogue):
    """A neutral commodities PM's persona, traits, pm_rules, adapter, catalogue and config."""
    persona, traits, pm_rules = neutral_pm(AssetClass.COMMODITIES, _COMMODITY_SUB_STYLE)
    adapter = CommoditiesAdapter(sub_style=_COMMODITY_SUB_STYLE, horizon_days=20)
    config = Config()
    return persona, traits, pm_rules, adapter, catalogue, config


def _idea_rules_for(pos: Position, pm_id: str, *, signpost_level: float | None = None) -> dict:
    rules = [
        _stop_rule(pm_id, pos.trade_idea_id, "r_90", pos.stop_level),
        _target_rule(pm_id, pos.trade_idea_id, "r_91", pos.target_level),
    ]
    if signpost_level is not None:
        rules.append(_signpost_rule(pm_id, pos.trade_idea_id, "r_92", signpost_level))
    return {pos.trade_idea_id: tuple(rules)}


# --- Determinism -------------------------------------------------------------


def test_step_is_deterministic_and_does_not_mutate_state(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0] * 10, instrument_id="EQ-A")
    pos = _position("ti_001", "EQ-A", entry_t=0)
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    state_before = state
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state1, out1 = step(state, 1, view, ctx, idea_rules)
    new_state2, out2 = step(state, 1, view, ctx, idea_rules)

    assert new_state1 == new_state2
    assert out1 == out2
    assert state.positions == (pos,)
    assert state == state_before


# --- Stop / signpost / trim / hold triggers -----------------------------------


def test_stop_crossed_closes_position_with_one_exit_row(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0], instrument_id="EQ-A")
    pos = _position("ti_001", "EQ-A", entry_t=0, target_level=100.0, stop_level=-10.0)
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert len(out.rule_events) == 1
    assert out.rule_events[0].rule_id == "r_90"
    assert out.rule_events[0].response == RuleResponse.ACTED
    assert len(out.ledger_rows) == 1
    assert out.ledger_rows[0].side == Side.SELL
    assert out.ledger_rows[0].rule_id is None
    assert out.ledger_rows[0].bias_flag is None
    assert len(out.closed) == 1
    assert out.closed[0][0] == "ti_001"
    assert "the stop" in out.closed[0][2]
    assert out.opportunities["exits"] == 1
    assert out.opportunities["triggers_fired"] == 1


def test_stop_and_signpost_same_day_two_events_one_exit(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0], instrument_id="EQ-A")
    pos = _position("ti_001", "EQ-A", entry_t=0, target_level=100.0, stop_level=-10.0)
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id, signpost_level=-8.0)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert len(out.rule_events) == 2
    assert {ev.rule_id for ev in out.rule_events} == {"r_90", "r_92"}
    assert all(ev.response == RuleResponse.ACTED for ev in out.rule_events)
    assert len(out.ledger_rows) == 1
    assert len(out.closed) == 1


def test_full_exit_deficiency_breach_acks_stop_and_keeps_position(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0], instrument_id="EQ-A")
    # At the mandate cap already, so a same-day discretionary add (from the loss) has no room to
    # act and produces no row, keeping the "does not re-fire" assertion below unambiguous.
    pos = _position(
        "ti_001", "EQ-A", entry_t=0, target_level=100.0, stop_level=-10.0, size_pct_book=10.0
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (1.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert len(out.rule_events) == 1
    assert out.rule_events[0].response == RuleResponse.ACKED_NO_ACTION
    assert out.ledger_rows == ()
    assert out.closed == ()

    new_state2, out2 = step(new_state, 2, view, ctx, idea_rules)
    assert new_state2.n_positions == 1
    assert out2.rule_events == ()
    assert out2.ledger_rows == ()


def test_full_exit_deficiency_breach_at_loss_with_lambda_active_adds(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0, -15.0, -15.0, -15.0], instrument_id="EQ-A")
    pos = _position(
        "ti_001", "EQ-A", entry_t=0, target_level=100.0, stop_level=-10.0, size_pct_book=5.0
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(
        traits, {"exit_deficiency": (1.0, False), "loss_aversion_lambda": (1.1, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert out.rule_events[0].response == RuleResponse.ADDED
    assert len(out.ledger_rows) == 1
    row = out.ledger_rows[0]
    assert row.rule_id == "r_90"
    assert row.bias_flag == "exit_deficiency:added"
    assert row.side == Side.BUY
    new_pos = new_state.position("ti_001")
    assert new_pos.size_pct_book > pos.size_pct_book
    assert new_pos.size_pct_book <= 10.0


def test_trim_at_target_halves_size_with_one_sell_row(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, 15.0, 15.0, 15.0, 15.0], instrument_id="EQ-A")
    pos = _position(
        "ti_001", "EQ-A", entry_t=0, target_level=10.0, stop_level=-1000.0, size_pct_book=4.0
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert len(out.rule_events) == 2
    assert {ev.response for ev in out.rule_events} == {RuleResponse.ACTED}
    trim_rule = next(r for r in pm_rules if r.param == "trim_at_target")
    assert {ev.rule_id for ev in out.rule_events} == {"r_91", trim_rule.rule_id}
    assert len(out.ledger_rows) == 1
    assert out.ledger_rows[0].side == Side.SELL
    new_pos = new_state.position("ti_001")
    assert new_pos.size_pct_book == pytest.approx(2.0)
    assert out.closed == ()


def test_hold_rule_overridden_by_stop(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    levels = [0.0] * 6 + [-15.0]
    view = _view(levels, instrument_id="EQ-A")
    pos = _position("ti_001", "EQ-A", entry_t=0, target_level=100.0, stop_level=-10.0)
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 6, view, ctx, idea_rules)

    hold_rule = next(r for r in pm_rules if r.param == "min_holding_period")
    events_by_id = {ev.rule_id: ev for ev in out.rule_events}
    assert events_by_id["r_90"].response == RuleResponse.ACTED
    assert events_by_id[hold_rule.rule_id].response == RuleResponse.OVERRIDDEN
    assert new_state.n_positions == 0


# --- Commodity roll ------------------------------------------------------------


def test_commodity_roll_retags_legs_and_resets_at_expiry(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    # A constant-maturity curve: M1 and M2 hold their levels every day, including at expiry.
    m1 = [100.0] * 14
    m2 = [102.0] * 14
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=1,
        entry_level=90.0,
        target_level=1000.0,
        stop_level=-1000.0,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(
        traits, {"exit_deficiency": (0.0, False), "disposition_ratio": (1e-8, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)

    roll_days: list[int] = []
    for t in range(4, 13):
        state, out = step(state, t, view, ctx, idea_rules)
        assert state.n_positions == 1
        assert out.position_days[0].pnl_unit == pytest.approx(10.0)
        pos_now = state.position("ti_001")
        if out.ledger_rows:
            roll_days.append(t)
            assert len(out.ledger_rows) == 2
            sides = {row.tenor: row.side for row in out.ledger_rows}
            assert sides[Tenor.M1] == Side.SELL
            assert sides[Tenor.M2] == Side.BUY
        if t < 5:
            assert pos_now.series.legs[0].tenor == Tenor.M1
            assert pos_now.legs[0].tenor == Tenor.M1
            assert pos_now.rolled_until_t is None
        elif t <= 10:
            assert pos_now.series.legs[0].tenor == Tenor.M2
            assert pos_now.legs[0].tenor == Tenor.M2
            assert pos_now.rolled_until_t == 10
        else:
            assert pos_now.series.legs[0].tenor == Tenor.M1
            assert pos_now.legs[0].tenor == Tenor.M1
            assert pos_now.rolled_until_t is None

    assert roll_days == [5]
    final_pos = state.position("ti_001")
    assert final_pos.entry_level == pytest.approx(90.0)
    assert final_pos.rolled_offset == pytest.approx(0.0)
    assert final_pos.roll_breached is False


def test_roll_does_not_spuriously_trigger_target_via_raw_m2_level(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    # Contango: M2's raw level (102) sits above a target of 101, though the tracked (M1-frame)
    # level never moves off 100 - comparing the raw mid-roll level against target would fire.
    m1 = [100.0] * 14
    m2 = [102.0] * 14
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=1,
        entry_level=90.0,
        target_level=101.0,
        stop_level=-1000.0,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    target_rule_id = idea_rules["ti_001"][1].rule_id
    trim_rule = next(r for r in pm_rules if r.param == "trim_at_target")
    traits = _traits_with(
        traits, {"exit_deficiency": (0.0, False), "disposition_ratio": (1e-8, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)

    for t in range(4, 13):
        state, out = step(state, t, view, ctx, idea_rules)
        assert state.n_positions == 1
        assert out.position_days[0].pnl_unit == pytest.approx(10.0)
        assert out.closed == ()
        fired_ids = {ev.rule_id for ev in out.rule_events}
        assert target_rule_id not in fired_ids
        assert trim_rule.rule_id not in fired_ids
        if t == 5:
            assert len(out.ledger_rows) == 2
        else:
            assert out.ledger_rows == ()

    final_pos = state.position("ti_001")
    assert final_pos.size_pct_book == pytest.approx(pos.size_pct_book)


def test_roll_does_not_spuriously_stop_out_via_raw_m2_level_in_backwardation(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    # Backwardation: M2's raw level (98) sits below a stop of 99, though the tracked (M1-frame)
    # level never moves off 100 - comparing the raw mid-roll level against stop would fire.
    m1 = [100.0] * 14
    m2 = [98.0] * 14
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=1,
        entry_level=90.0,
        target_level=1000.0,
        stop_level=99.0,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    stop_rule_id = idea_rules["ti_001"][0].rule_id
    traits = _traits_with(
        traits, {"exit_deficiency": (0.0, False), "disposition_ratio": (1e-8, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)

    for t in range(4, 13):
        state, out = step(state, t, view, ctx, idea_rules)
        assert state.n_positions == 1
        assert out.position_days[0].pnl_unit == pytest.approx(10.0)
        assert out.closed == ()
        fired_ids = {ev.rule_id for ev in out.rule_events}
        assert stop_rule_id not in fired_ids
        if t == 5:
            assert len(out.ledger_rows) == 2
        else:
            assert out.ledger_rows == ()


def test_breached_roll_force_rolls_at_expiry_with_late_roll_flag(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    m1 = [100.0] * 13
    m2 = [102.0] * 13
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=0,
        entry_level=90.0,
        target_level=1000.0,
        stop_level=-1000.0,
        tenor=Tenor.M1,
        roll_breached=True,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(
        traits, {"exit_deficiency": (1.0, False), "disposition_ratio": (1e-8, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)
    roll_rule = next(r for r in pm_rules if r.param == "roll_before_expiry")

    for t in range(9, 12):
        state, out = step(state, t, view, ctx, idea_rules)
        assert state.n_positions == 1
        assert out.position_days[0].pnl_unit == pytest.approx(10.0)
        pos_now = state.position("ti_001")
        if t == 10:
            assert len(out.ledger_rows) == 2
            assert all(row.bias_flag == LATE_ROLL_FLAG for row in out.ledger_rows)
            assert pos_now.series.legs[0].tenor == Tenor.M2
            assert pos_now.rolled_until_t == 10
            # The roll rule's own breach fires an event; the force roll adds no event row.
            roll_events = [ev for ev in out.rule_events if ev.rule_id == roll_rule.rule_id]
            assert len(roll_events) == 1
            assert roll_events[0].response == RuleResponse.ACKED_NO_ACTION
        else:
            assert out.ledger_rows == ()
            assert pos_now.series.legs[0].tenor == Tenor.M1
            assert pos_now.rolled_until_t is None

    final_pos = state.position("ti_001")
    assert final_pos.entry_level == pytest.approx(90.0)
    assert final_pos.rolled_offset == pytest.approx(0.0)
    assert final_pos.roll_breached is False


def test_force_roll_without_a_roll_rule_carries_no_flag(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    pm_rules_no_roll = tuple(r for r in pm_rules if r.param != "roll_before_expiry")
    m1 = [100.0] * 13
    m2 = [102.0] * 13
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=0,
        entry_level=90.0,
        target_level=1000.0,
        stop_level=-1000.0,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(
        traits, {"exit_deficiency": (0.0, False), "disposition_ratio": (1e-8, True)}
    )
    ctx = _ctx(persona, traits, pm_rules_no_roll, adapter, ["CM-CRD"], catalogue, config)

    for t in range(9, 12):
        state, out = step(state, t, view, ctx, idea_rules)
        assert state.n_positions == 1
        assert out.position_days[0].pnl_unit == pytest.approx(10.0)
        pos_now = state.position("ti_001")
        if t == 10:
            assert len(out.ledger_rows) == 2
            assert all(row.bias_flag is None for row in out.ledger_rows)
            assert pos_now.series.legs[0].tenor == Tenor.M2
            assert pos_now.rolled_until_t == 10
        else:
            assert out.ledger_rows == ()
            assert pos_now.series.legs[0].tenor == Tenor.M1
            assert pos_now.rolled_until_t is None

    final_pos = state.position("ti_001")
    assert final_pos.entry_level == pytest.approx(90.0)
    assert final_pos.rolled_offset == pytest.approx(0.0)


# --- Discretionary block -------------------------------------------------------


def test_anchoring_exit_when_rho_one_and_level_at_anchor(
    eq_parts, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([20.0] * 8, instrument_id="EQ-A")
    pos = _position(
        "ti_001", "EQ-A", entry_t=0, entry_level=0.0, target_level=100.0, stop_level=-100.0
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False), "anchoring_rho": (1.0, True)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)
    monkeypatch.setattr(type(adapter), "anchors", lambda self, p, v, t: (15.0,))

    new_state, out = step(state, 6, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert len(out.closed) == 1
    position_day = out.position_days[0]
    assert position_day.bias_flag == "anchoring:exit_at_anchor"
    assert position_day.action == PositionAction.EXIT
    assert position_day.anchor_level == 15.0


def test_disposition_hazard_sells_a_gain_early(eq_parts, monkeypatch: pytest.MonkeyPatch) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([20.0] * 8, instrument_id="EQ-A")
    pos = _position(
        "ti_001", "EQ-A", entry_t=0, entry_level=0.0, target_level=100.0, stop_level=-100.0
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(
        traits, {"exit_deficiency": (0.0, False), "disposition_ratio": (1e6, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)
    monkeypatch.setattr(disposition_module, "draw_sell", lambda h, rng: True)

    new_state, out = step(state, 6, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    position_day = out.position_days[0]
    assert position_day.bias_flag == "disposition:realise_gain_early"
    assert position_day.action == PositionAction.EXIT


def test_min_holding_period_blocks_a_discretionary_cut(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, 0.0, -30.0, -30.0, -30.0], instrument_id="EQ-A")
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=0,
        entry_level=0.0,
        target_level=100.0,
        stop_level=-100.0,
        forecast=-1000.0,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    min_holding_rule = next(r for r in pm_rules if r.param == "min_holding_period")
    assert float(min_holding_rule.level) == 5.0
    traits = _traits_with(
        traits, {"exit_deficiency": (0.0, False), "loss_aversion_lambda": (0.0, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 2, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert out.ledger_rows == ()
    assert out.closed == ()
    position_day = out.position_days[0]
    assert position_day.action == PositionAction.NONE


def test_discretionary_add_through_a_no_add_breach(
    eq_parts, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0, -15.0, -15.0, -15.0], instrument_id="EQ-A")
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=0,
        entry_level=0.0,
        target_level=100.0,
        stop_level=-100.0,
        size_pct_book=5.0,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    no_add_rule = next(r for r in pm_rules if r.param == "no_add_before_trigger")
    traits = _traits_with(traits, {"exit_deficiency": (1.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)
    monkeypatch.setattr(
        loss_aversion_module,
        "choose",
        lambda *a, **kw: LossSideChoice(PositionAction.ADD, "loss_aversion:add"),
    )

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert len(out.ledger_rows) == 1
    row = out.ledger_rows[0]
    assert row.rule_id == no_add_rule.rule_id
    assert row.bias_flag == "loss_aversion:add_before_trigger"
    new_pos = new_state.position("ti_001")
    assert new_pos.size_pct_book == pytest.approx(7.5)


def test_discretionary_add_without_a_breach(eq_parts, monkeypatch: pytest.MonkeyPatch) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0, -15.0, -15.0, -15.0], instrument_id="EQ-A")
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=0,
        entry_level=0.0,
        target_level=100.0,
        stop_level=-100.0,
        size_pct_book=5.0,
        # Already past a trigger, so the no-add rule's guard never applies regardless of e.
        triggers_fired=1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)
    monkeypatch.setattr(
        loss_aversion_module,
        "choose",
        lambda *a, **kw: LossSideChoice(PositionAction.ADD, "loss_aversion:add"),
    )

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert len(out.ledger_rows) == 1
    row = out.ledger_rows[0]
    assert row.rule_id is None
    assert row.bias_flag == "loss_aversion:add"
    new_pos = new_state.position("ti_001")
    assert new_pos.size_pct_book == pytest.approx(7.5)


# --- Position-day rows and close-out -------------------------------------------


def test_position_day_rows_one_per_open_position(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0], instrument_id="EQ-A")
    pos = _position("ti_001", "EQ-A", entry_t=0, target_level=100.0, stop_level=-10.0)
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert len(out.position_days) == 1
    row = out.position_days[0]
    assert row.trade_idea_id == "ti_001"
    assert row.trigger_pending is True
    assert row.action == PositionAction.EXIT


def test_last_day_close_out_exits_every_open_position(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0] * 8, instrument_id="EQ-A")
    last = view.n_days - 1
    pos = _position(
        "ti_001", "EQ-A", entry_t=last - 1, entry_level=0.0, target_level=100.0, stop_level=-100.0
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, last, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert len(out.closed) == 1
    assert out.closed[0][0] == "ti_001"
    assert "the year end" in out.closed[0][2]
    row = out.position_days[0]
    assert row.action == PositionAction.EXIT
    assert row.bias_flag is None
    assert out.opportunities["exits"] == 1
    assert out.opportunities["sell_day_position_days"] == 1
