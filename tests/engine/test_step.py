"""Tests for the daily step: `step`, `handle_triggers` and `handle_discretionary`."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.engine.adapters.commodities import CommoditiesAdapter
from pm_traitbench.engine.adapters.equities import EquitiesAdapter
from pm_traitbench.engine.adapters.rates_credit import RatesCreditAdapter
from pm_traitbench.engine.biases import anchoring
from pm_traitbench.engine.biases import disposition as disposition_module
from pm_traitbench.engine.biases import loss_aversion as loss_aversion_module
from pm_traitbench.engine.biases.exit_deficiency import LATE_ROLL_FLAG
from pm_traitbench.engine.biases.loss_aversion import LossSideChoice
from pm_traitbench.engine.discretionary import handle_discretionary
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.params import ParamSchedule
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.engine.step import PmContext, step
from pm_traitbench.engine.triggers import exit_rows, trim_half
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
    PnlState,
    PositionAction,
    RatingBand,
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
    regime: Regime = Regime.RANGE,
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
    regimes = [RegimeSpan(seed=_SEED, regime=regime, date_start=dates[0], date_end=dates[-1])]
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


def _credit_view(spreads: Sequence[float], *, instrument_id: str) -> MarketView:
    """A single credit issuer whose spread (bp) is controlled directly, in a range regime."""
    dates = _dates(len(spreads))
    instrument = Instrument(
        instrument_id=instrument_id,
        family=Family.CREDIT,
        kind=InstrumentKind.CREDIT_ISSUER,
        name="Test issuer",
        currency="USD",
        sector="sector_01",
        rating_band=RatingBand.A,
        commodity_group=None,
        duration_years=5.0,
        beta=None,
        expiry_rule=None,
    )
    prices = [
        Price(
            seed=_SEED,
            date=d,
            instrument_id=instrument_id,
            price=100.0,
            spread_bp=spreads[t],
        )
        for t, d in enumerate(dates)
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
    size_at_entry: float | None = None,
    conviction: int = 3,
    size_rank: int = 3,
    triggers_fired: int = 0,
    consumed_rule_ids: frozenset[str] = frozenset(),
    run_counters: tuple[tuple[str, int], ...] = (),
    size_changed_t: int | None = None,
    rolled_offset: float = 0.0,
    roll_breached: bool = False,
    tenor: Tenor | None = None,
    anchor_level: float | None = None,
    anchored: bool = False,
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
        size_at_entry=(
            size_at_entry if size_at_entry is not None else original_size_pct_book or size_pct_book
        ),
        conviction=conviction,
        size_rank=size_rank,
        triggers_fired=triggers_fired,
        consumed_rule_ids=consumed_rule_ids,
        run_counters=run_counters,
        size_changed_t=size_changed_t if size_changed_t is not None else entry_t,
        rolled_offset=rolled_offset,
        roll_breached=roll_breached,
        anchor_level=anchor_level,
        anchored=anchored,
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
    view = _view([0.0, -15.0, -15.0], instrument_id="EQ-A")
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


def test_commodity_exit_sells_the_contracts_bought_at_entry_after_a_price_move(
    cm_parts,
) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    view = _commodity_view([460.0, 455.0, 440.0], [461.0, 456.0, 441.0], expiry_t=2)
    legs = (LegRef("CM-CRD", Tenor.M1, 1.0),)
    size_at_entry, _ = adapter.size_and_risk(5.0, legs, view, 0, persona.mandate.book_size)
    repriced, _ = adapter.size_and_risk(5.0, legs, view, 1, persona.mandate.book_size)
    assert repriced != size_at_entry
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=0,
        entry_level=460.0,
        target_level=1000.0,
        stop_level=457.0,
        size_pct_book=5.0,
        size_at_entry=size_at_entry,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert [(row.side, row.size) for row in out.ledger_rows] == [(Side.SELL, size_at_entry)]


def test_commodity_trim_then_exit_sizes_sum_to_the_entry_size(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    view = _commodity_view([460.0, 455.0, 440.0], [461.0, 456.0, 441.0], expiry_t=2)
    legs = (LegRef("CM-CRD", Tenor.M1, 1.0),)
    size_at_entry, _ = adapter.size_and_risk(5.0, legs, view, 0, persona.mandate.book_size)
    pos = _position(
        "ti_001", "CM-CRD", size_pct_book=5.0, size_at_entry=size_at_entry, tenor=Tenor.M1
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)

    trimmed, trim_rows = trim_half(pos, ctx, view, 1)
    close_rows = exit_rows(trimmed, ctx, view, 2, None, None)

    assert sum(row.size for row in (*trim_rows, *close_rows)) == pytest.approx(size_at_entry)


def test_stop_and_signpost_same_day_two_events_one_exit(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0, -15.0], instrument_id="EQ-A")
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
    levels = [0.0] * 6 + [-15.0, -15.0]
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


def test_breached_roll_fires_once_then_force_rolls_at_expiry_with_late_roll_flag(
    cm_parts,
) -> None:
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
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(
        traits, {"exit_deficiency": (1.0, False), "disposition_ratio": (1e-8, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)
    roll_rule = next(r for r in pm_rules if r.param == "roll_before_expiry")

    roll_events = []
    for t in range(4, 12):
        state, out = step(state, t, view, ctx, idea_rules)
        assert state.n_positions == 1
        assert out.position_days[0].pnl_unit == pytest.approx(10.0)
        roll_events.extend((t, ev) for ev in out.rule_events if ev.rule_id == roll_rule.rule_id)
        pos_now = state.position("ti_001")
        if t == 10:
            assert len(out.ledger_rows) == 2
            assert all(row.bias_flag == LATE_ROLL_FLAG for row in out.ledger_rows)
            assert pos_now.series.legs[0].tenor == Tenor.M2
            assert pos_now.rolled_until_t == 10
        else:
            assert out.ledger_rows == ()
        if 5 <= t < 10:
            # Breached on its first firing day and silent until the force roll.
            assert pos_now.roll_breached is True

    assert [t for t, _ in roll_events] == [5]
    assert roll_events[0][1].response == RuleResponse.ACKED_NO_ACTION
    final_pos = state.position("ti_001")
    assert final_pos.series.legs[0].tenor == Tenor.M1
    assert final_pos.rolled_until_t is None
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


def test_force_roll_skips_discretionary_even_when_loss_aversion_would_add(
    cm_parts, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    pm_rules_no_roll = tuple(r for r in pm_rules if r.param != "roll_before_expiry")
    m1 = [100.0] * 13
    m2 = [102.0] * 13
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=0,
        entry_level=110.0,
        target_level=1000.0,
        stop_level=-1000.0,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules_no_roll, adapter, ["CM-CRD"], catalogue, config)
    monkeypatch.setattr(
        loss_aversion_module,
        "choose",
        lambda *a, **kw: LossSideChoice(PositionAction.ADD, "loss_aversion:add"),
    )

    state, out = step(state, 10, view, ctx, idea_rules)

    assert state.n_positions == 1
    assert out.position_days[0].pnl_unit == pytest.approx(-10.0)
    assert len(out.ledger_rows) == 2
    keys = {(row.instrument_id, row.tenor, row.side) for row in out.ledger_rows}
    assert keys == {("CM-CRD", Tenor.M1, Side.SELL), ("CM-CRD", Tenor.M2, Side.BUY)}


def test_expiry_day_stop_after_force_roll_produces_unique_ledger_keys(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    # M1 drops through the stop on the expiry day itself; the force roll retags to M2 before the
    # stop is evaluated, so it fires against the tracked (M1-equivalent) level, not a raw M2
    # price, and its exit row sells the already-rolled M2 leg.
    m1 = [100.0] * 10 + [80.0] * 3
    m2 = [level * 1.02 for level in m1]
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=0,
        entry_level=100.0,
        target_level=1000.0,
        stop_level=90.0,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)

    new_state, out = step(state, 10, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert len(out.ledger_rows) == 3
    keys = {(row.tenor, row.side) for row in out.ledger_rows}
    assert len(keys) == 3
    assert keys == {(Tenor.M1, Side.SELL), (Tenor.M2, Side.BUY), (Tenor.M2, Side.SELL)}
    assert len(out.closed) == 1


def test_expiry_day_trim_after_force_roll_produces_unique_ledger_keys(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    # Same setup, but M1 rises through the target instead: the trim's sell row must also land on
    # the already-rolled M2 leg, distinct from the roll's own M1-sell/M2-buy pair.
    m1 = [100.0] * 10 + [120.0] * 3
    m2 = [level * 1.02 for level in m1]
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=0,
        entry_level=100.0,
        target_level=110.0,
        stop_level=-1000.0,
        size_pct_book=4.0,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)

    new_state, out = step(state, 10, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert len(out.ledger_rows) == 3
    keys = {(row.tenor, row.side) for row in out.ledger_rows}
    assert len(keys) == 3
    assert keys == {(Tenor.M1, Side.SELL), (Tenor.M2, Side.BUY), (Tenor.M2, Side.SELL)}
    new_pos = new_state.position("ti_001")
    assert new_pos.size_pct_book == pytest.approx(2.0)
    assert out.closed == ()


def test_expiry_day_added_response_writes_no_add_row(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    # Same stop-breach setup, but the breach response is `added` (exit deficiency plus loss
    # aversion, both active). On a force-roll day the add is skipped entirely - the same
    # convention as an add skipped at the mandate cap - since the force roll's own "open the new
    # leg" row already claims that (instrument, tenor, side) key today; only the roll rows are
    # written, the `added` event and bias flag still record what happened, and size is unchanged.
    m1 = [100.0] * 10 + [80.0] * 3
    m2 = [level * 1.02 for level in m1]
    view = _commodity_view(m1, m2, expiry_t=10, instrument_id="CM-CRD")
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=0,
        entry_level=100.0,
        target_level=1000.0,
        stop_level=90.0,
        size_pct_book=4.0,
        tenor=Tenor.M1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    stop_rule_id = idea_rules["ti_001"][0].rule_id
    traits = _traits_with(
        traits, {"exit_deficiency": (1.0, False), "loss_aversion_lambda": (1.1, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)

    new_state, out = step(state, 10, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert len(out.ledger_rows) == 2
    keys = {(row.tenor, row.side) for row in out.ledger_rows}
    assert keys == {(Tenor.M1, Side.SELL), (Tenor.M2, Side.BUY)}
    assert all(row.rule_id is None for row in out.ledger_rows)
    stop_events = [ev for ev in out.rule_events if ev.rule_id == stop_rule_id]
    assert len(stop_events) == 1
    assert stop_events[0].response == RuleResponse.ADDED
    new_pos = new_state.position("ti_001")
    assert new_pos.size_pct_book == pytest.approx(pos.size_pct_book)


# --- Discretionary block -------------------------------------------------------


def test_anchored_idea_exits_when_level_reaches_the_anchor(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([20.0] * 8, instrument_id="EQ-A")
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=0,
        entry_level=0.0,
        target_level=100.0,
        stop_level=-100.0,
        anchor_level=15.0,
        anchored=True,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False), "anchoring_rho": (1.0, True)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 6, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert len(out.closed) == 1
    position_day = out.position_days[0]
    assert position_day.bias_flag == "anchoring:exit_at_anchor"
    assert position_day.action == PositionAction.EXIT
    assert position_day.anchor_level == 15.0


def test_unanchored_idea_never_exits_at_a_level_and_targets_effective_exit(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    m1 = [100.0] * 8
    m2 = [98.0] * 8
    view = _commodity_view(m1, m2, expiry_t=7, instrument_id="CM-CRD", regime=Regime.RISK_OFF)
    t = 5
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=t,
        entry_level=150.0,
        target_level=200.0,
        stop_level=-1000.0,
        tenor=Tenor.M2,
        rolled_offset=-2.0,
        anchor_level=None,
        anchored=False,
    )
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)
    params = ctx.schedule.for_day(view.dates[t], view.regime(t))

    anchored = anchoring.evaluate(pos, 100.0)
    assert anchored.anchor_level is None
    assert anchored.effective_exit_level == pytest.approx(200.0)

    result = handle_discretionary(pos, view, t, ctx, params, 100.0, PnlState.LOSS, -1.0)

    assert result.position is not None
    assert result.bias_flag is None or "anchoring:exit_at_anchor" not in result.bias_flag


def test_anchored_idea_reaches_the_anchor_in_the_tracked_frame_mid_roll(cm_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = cm_parts
    # The anchor (100) was fixed at entry, in the same frame `level_now` is passed in here
    # (already re-based by the caller), so a mid-roll position still compares correctly.
    m1 = [100.0] * 8
    m2 = [98.0] * 8
    view = _commodity_view(m1, m2, expiry_t=7, instrument_id="CM-CRD", regime=Regime.RISK_OFF)
    t = 5
    pos = _position(
        "ti_001",
        "CM-CRD",
        entry_t=0,
        entry_level=90.0,
        target_level=105.0,
        stop_level=-1000.0,
        tenor=Tenor.M2,
        rolled_offset=-2.0,
        anchor_level=100.0,
        anchored=True,
    )
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False), "anchoring_rho": (1.0, True)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CM-CRD"], catalogue, config)
    params = ctx.schedule.for_day(view.dates[t], view.regime(t))

    anchored = anchoring.evaluate(pos, 100.0)
    assert anchored.anchor_level == pytest.approx(100.0)
    assert anchored.effective_exit_level == pytest.approx(100.0)
    assert anchored.reached

    result = handle_discretionary(pos, view, t, ctx, params, 100.0, PnlState.GAIN, 10.0 / 15.0)

    assert result.position is None
    assert result.bias_flag == "anchoring:exit_at_anchor"


def test_trigger_day_row_still_carries_the_ideas_anchor(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0, -15.0], instrument_id="EQ-A")
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=0,
        target_level=100.0,
        stop_level=-10.0,
        anchor_level=40.0,
        anchored=True,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    # A stop-loss trigger closed this position, not the discretionary block: the row still
    # carries the idea's anchor, since it was fixed at entry rather than computed that day.
    assert out.opportunities["triggers_fired"] == 1
    position_day = out.position_days[0]
    assert position_day.action == PositionAction.EXIT
    assert position_day.anchor_level == pytest.approx(40.0)
    assert position_day.effective_exit_level == pytest.approx(40.0)


def test_last_day_row_still_carries_the_ideas_anchor(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0] * 8, instrument_id="EQ-A")
    last = view.n_days - 1
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=last - 1,
        entry_level=0.0,
        target_level=100.0,
        stop_level=-100.0,
        anchor_level=30.0,
        anchored=True,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, last, view, ctx, idea_rules)

    row = out.position_days[0]
    assert row.action == PositionAction.EXIT
    assert row.anchor_level == pytest.approx(30.0)
    assert row.effective_exit_level == pytest.approx(30.0)


def test_min_holding_period_blocks_an_anchored_exit(
    eq_parts, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([20.0] * 8, instrument_id="EQ-A")
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=0,
        entry_level=0.0,
        target_level=100.0,
        stop_level=-100.0,
        anchor_level=15.0,
        anchored=True,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    min_holding_rule = next(r for r in pm_rules if r.param == "min_holding_period")
    assert float(min_holding_rule.level) == 5.0
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False), "anchoring_rho": (1.0, True)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)
    # Isolates the min-holding assertion from the unrelated disposition sell hazard, which
    # would otherwise also be free to fire and close the position for a different reason.
    monkeypatch.setattr(disposition_module, "draw_sell", lambda h, rng: False)

    new_state, out = step(state, 2, view, ctx, idea_rules)

    assert new_state.n_positions == 1
    assert out.closed == ()
    position_day = out.position_days[0]
    assert position_day.action != PositionAction.EXIT
    assert position_day.anchor_level == pytest.approx(15.0)


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
    # lambda near zero pushes the cut hazard (0.05 / lambda) past 1, forcing a cut.
    traits = _traits_with(
        traits, {"exit_deficiency": (0.0, False), "loss_aversion_lambda": (0.01, True)}
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


def test_add_blocked_at_the_cap_is_a_hold_and_disposition_still_runs(
    eq_parts, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0] + [-15.0] * 7, instrument_id="EQ-A")
    cap = float(next(r for r in pm_rules if r.param == "max_risk_pct").level)
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=0,
        entry_level=0.0,
        target_level=100.0,
        stop_level=-100.0,
        size_pct_book=cap,
        # Already past a trigger, so the no-add rule's guard never applies.
        triggers_fired=1,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    traits = _traits_with(
        traits, {"exit_deficiency": (0.0, False), "loss_aversion_lambda": (2.0, True)}
    )
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)
    monkeypatch.setattr(
        loss_aversion_module,
        "choose",
        lambda *a, **kw: LossSideChoice(PositionAction.ADD, "loss_aversion:add"),
    )
    monkeypatch.setattr(disposition_module, "draw_sell", lambda *a, **kw: True)

    new_state, out = step(state, 6, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert len(out.closed) == 1
    position_day = out.position_days[0]
    assert position_day.action == PositionAction.EXIT
    assert position_day.bias_flag == "loss_aversion:hold"


# --- Position-day rows and close-out -------------------------------------------


def test_position_day_rows_one_per_open_position(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, -15.0, -15.0], instrument_id="EQ-A")
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


def test_position_day_triggers_fired_is_cumulative(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    view = _view([0.0, 0.0, -15.0, -15.0], instrument_id="EQ-A")
    pos = _position("ti_001", "EQ-A", stop_level=-10.0, size_pct_book=10.0, triggers_fired=2)
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    # e=1 acks the stop, so the position survives to a second day.
    traits = _traits_with(traits, {"exit_deficiency": (1.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    state, quiet = step(state, 1, view, ctx, idea_rules)
    assert quiet.position_days[0].triggers_fired == 2
    state, fired = step(state, 2, view, ctx, idea_rules)
    assert fired.rule_events
    assert fired.position_days[0].triggers_fired == 3
    state, closing = step(state, 3, view, ctx, idea_rules)
    assert closing.position_days[0].action == PositionAction.EXIT
    assert closing.position_days[0].triggers_fired == 3


def test_credit_buy_stops_out_on_a_spread_widening_with_negative_pnl(neutral_pm, catalogue) -> None:
    persona, traits, pm_rules = neutral_pm(AssetClass.RATES_CREDIT, "long_short_credit")
    adapter = RatesCreditAdapter(sub_style="long_short_credit", horizon_days=20)
    config = Config()
    view = _credit_view([100.0, 115.0, 115.0], instrument_id="CR-A")
    # A credit buy is long the bond: a falling spread is good, so the stop sits above entry.
    series = Series(legs=(LegRef("CR-A", None, 1.0),), bullish_sign=-1, unit="bp")
    leg = Leg(instrument_id="CR-A", tenor=None, side=Side.BUY, weight=1.0)
    pos = replace(
        _position("ti_001", "CR-A", entry_level=100.0, target_level=60.0, stop_level=110.0),
        series=series,
        legs=(leg,),
    )
    assert pos.adverse_dir == 1
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = {
        "ti_001": (
            _stop_rule(persona.pm_id, "ti_001", "r_90", 110.0, adverse_dir=1),
            _target_rule(persona.pm_id, "ti_001", "r_91", 60.0, adverse_dir=1),
        )
    }
    traits = _traits_with(traits, {"exit_deficiency": (0.0, False)})
    ctx = _ctx(persona, traits, pm_rules, adapter, ["CR-A"], catalogue, config)

    new_state, out = step(state, 1, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert [ev.rule_id for ev in out.rule_events] == ["r_90"]
    assert out.position_days[0].pnl_unit == pytest.approx(-15.0)
    assert out.position_days[0].pnl_state == PnlState.LOSS
    assert [row.side for row in out.ledger_rows] == [Side.SELL]
    assert "the stop" in out.closed[0][2]


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
    assert "the horizon end" in out.closed[0][2]
    row = out.position_days[0]
    assert row.action == PositionAction.EXIT
    assert row.bias_flag is None
    assert out.opportunities["exits"] == 1
    assert out.opportunities["sell_day_position_days"] == 1


def test_last_day_close_out_runs_before_any_trigger_is_evaluated(eq_parts) -> None:
    persona, traits, pm_rules, adapter, catalogue, config = eq_parts
    # Both the idea's own target and the PM's trim_at_target rule would fire if evaluated on
    # this last day - close-out must run first and skip evaluation entirely, so neither the
    # trim's sell row nor a rule event ever appears alongside the close-out's own sell row.
    view = _view([0.0] * 7 + [20.0], instrument_id="EQ-A")
    last = view.n_days - 1
    pos = _position(
        "ti_001",
        "EQ-A",
        entry_t=0,
        entry_level=0.0,
        target_level=10.0,
        stop_level=-100.0,
        size_pct_book=5.0,
    )
    state = PmState(pm_id=persona.pm_id, positions=(pos,), next_idea=2, next_rule=100)
    idea_rules = _idea_rules_for(pos, persona.pm_id)
    ctx = _ctx(persona, traits, pm_rules, adapter, ["EQ-A"], catalogue, config)

    new_state, out = step(state, last, view, ctx, idea_rules)

    assert new_state.n_positions == 0
    assert out.rule_events == ()
    assert len(out.ledger_rows) == 1
    assert out.ledger_rows[0].side == Side.SELL
    assert len(out.closed) == 1
    assert "the horizon end" in out.closed[0][2]
    row = out.position_days[0]
    assert row.action == PositionAction.EXIT
    assert row.trigger_pending is False
    assert row.triggers_fired == 0
