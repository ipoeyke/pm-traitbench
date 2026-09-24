"""Rule triggers: evaluating a position's rules for one day and executing the
winner's response, including a commodity roll and a forced late roll.

`handle_triggers` is pure: it never mutates its `position` argument, only
returns a new one. Row-builder helpers here are reused by `discretionary.py`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING

from pm_traitbench.engine.adapters.base import leg_side, tracked_level
from pm_traitbench.engine.adapters.base import pnl_unit as compute_pnl_unit
from pm_traitbench.engine.adapters.commodities import CommoditiesAdapter
from pm_traitbench.engine.biases import join_flags
from pm_traitbench.engine.biases.exit_deficiency import LATE_ROLL_FLAG, respond
from pm_traitbench.engine.constants import ADD_FRACTION
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.precedence import resolve
from pm_traitbench.engine.rules_eval import evaluate_day
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.engine.templates import render_outcome
from pm_traitbench.enums import Action, PnlState, PositionAction, RuleResponse, RuleScope, Side
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import LedgerRow, Rule, RuleEvent

if TYPE_CHECKING:
    from pm_traitbench.engine.step import PmContext


@dataclass(frozen=True)
class TriggerOutcome:
    """What happened to one position's rule triggers on one day."""

    position: Position | None
    events: tuple[RuleEvent, ...]
    ledger_rows: tuple[LedgerRow, ...]
    action: PositionAction
    bias_flag: str | None
    trigger_pending: bool
    fired_non_hold: int
    closed: tuple[str, date, str] | None
    sold: bool
    rolled_today: bool


def find_pm_rule(rules: Sequence[Rule], param: str) -> Rule | None:
    """The PM-scope rule for `param`, or None if the PM holds no such rule."""
    for rule in rules:
        if rule.scope == RuleScope.PM and rule.param == param:
            return rule
    return None


def mandate_cap(pm_rules: Sequence[Rule]) -> float:
    """The PM's mandate risk cap level, from its `max_risk_pct` rule."""
    rule = find_pm_rule(pm_rules, "max_risk_pct")
    if rule is None:
        raise EngineError("no mandate risk cap rule found for this PM")
    return float(rule.level)


def _leg_rows(
    position: Position,
    size_pct_book: float,
    side_sign: int,
    ctx: "PmContext",
    view: MarketView,
    t: int,
    bias_flag: str | None,
    rule_id: str | None,
) -> tuple[LedgerRow, ...]:
    """One ledger row per series leg at `size_pct_book`, direction from `side_sign`."""
    adapter = ctx.adapter
    book_size = ctx.persona.mandate.book_size
    legs = position.series.legs
    bullish_sign = position.series.bullish_sign
    size, risk_amount = adapter.size_and_risk(size_pct_book, legs, view, t, book_size)
    rows = []
    for leg in legs:
        side = leg_side(side_sign, bullish_sign, leg.coeff, adapter.leg_bullish(leg))
        rows.append(
            LedgerRow(
                pm_id=ctx.persona.pm_id,
                date=view.dates[t],
                trade_idea_id=position.trade_idea_id,
                instrument_id=leg.instrument_id,
                tenor=leg.tenor,
                instrument_type=adapter.instrument_type(leg.instrument_id, view),
                side=side,
                size=size,
                risk_amount=adapter.leg_risk_amount(leg, size, view, risk_amount),
                price_or_yield=adapter.leg_price(leg, view, t),
                stated_conviction=position.conviction,
                bias_flag=bias_flag,
                rule_id=rule_id,
            )
        )
    return tuple(rows)


def exit_rows(
    position: Position,
    ctx: "PmContext",
    view: MarketView,
    t: int,
    bias_flag: str | None,
    rule_id: str | None,
    size_pct_book: float | None = None,
) -> tuple[LedgerRow, ...]:
    """Sell (or buy, for a short) rows closing `size_pct_book` (default: the full size)."""
    spb = position.size_pct_book if size_pct_book is None else size_pct_book
    return _leg_rows(position, spb, -position.side_sign, ctx, view, t, bias_flag, rule_id)


def add_rows(
    position: Position,
    delta_size_pct_book: float,
    ctx: "PmContext",
    view: MarketView,
    t: int,
    bias_flag: str | None,
    rule_id: str | None,
) -> tuple[LedgerRow, ...]:
    """Rows opening `delta_size_pct_book` more, in the position's original direction."""
    return _leg_rows(
        position, delta_size_pct_book, position.side_sign, ctx, view, t, bias_flag, rule_id
    )


def apply_add(
    position: Position,
    cap_level: float,
    ctx: "PmContext",
    view: MarketView,
    t: int,
    bias_flag: str | None,
    rule_id: str | None,
) -> tuple[Position, tuple[LedgerRow, ...]]:
    """Buy `ADD_FRACTION` of the original size, capped; no row when already at cap."""
    if position.size_pct_book >= cap_level:
        return position, ()
    delta = min(cap_level, position.size_pct_book + ADD_FRACTION * position.original_size_pct_book)
    delta -= position.size_pct_book
    if delta <= 0:
        return position, ()
    rows = add_rows(position, delta, ctx, view, t, bias_flag, rule_id)
    new_size = position.size_pct_book + delta
    if new_size > cap_level:
        raise EngineError("an add must not push size_pct_book above the mandate cap")
    return replace(position, size_pct_book=new_size), rows


def trim_half(
    position: Position, ctx: "PmContext", view: MarketView, t: int
) -> tuple[Position, tuple[LedgerRow, ...]]:
    """Sell half the position; the other half stays open."""
    half = position.size_pct_book / 2
    rows = exit_rows(position, ctx, view, t, None, None, size_pct_book=half)
    if half <= 0:
        raise EngineError("a trim must not leave size_pct_book at or below 0")
    return replace(position, size_pct_book=half, size_changed_t=t), rows


def roll_rows(
    position: Position,
    adapter: CommoditiesAdapter,
    view: MarketView,
    t: int,
    ctx: "PmContext",
    bias_flag: str | None,
) -> tuple[LedgerRow, ...]:
    """A sell row for the old leg and a buy row for the new leg, per rolled pair."""
    book_size = ctx.persona.mandate.book_size
    size, risk_amount = adapter.size_and_risk(
        position.size_pct_book, position.series.legs, view, t, book_size
    )
    rows = []
    bullish_sign = position.series.bullish_sign
    for (old_leg, _old_level), (new_leg, _new_level) in adapter.roll_legs(position, view, t):
        entry_side = leg_side(
            position.side_sign, bullish_sign, old_leg.coeff, adapter.leg_bullish(old_leg)
        )
        close_side = Side.SELL if entry_side == Side.BUY else Side.BUY
        for leg, side in ((old_leg, close_side), (new_leg, entry_side)):
            rows.append(
                LedgerRow(
                    pm_id=ctx.persona.pm_id,
                    date=view.dates[t],
                    trade_idea_id=position.trade_idea_id,
                    instrument_id=leg.instrument_id,
                    tenor=leg.tenor,
                    instrument_type=adapter.instrument_type(leg.instrument_id, view),
                    side=side,
                    size=size,
                    risk_amount=adapter.leg_risk_amount(leg, size, view, risk_amount),
                    price_or_yield=adapter.leg_price(leg, view, t),
                    stated_conviction=position.conviction,
                    bias_flag=bias_flag,
                    rule_id=None,
                )
            )
    return tuple(rows)


def apply_roll(
    position: Position, adapter: CommoditiesAdapter, view: MarketView, t: int
) -> Position:
    """Shift `rolled_offset` by the roll's level gap, retag legs one month out, and record the
    expiry day the retag must be undone on.

    `entry_level`/`stop_level`/`target_level` never move: `rolled_offset` alone carries the
    tenor's raw-level gap, so `tracked_level` stays comparable with those fixed thresholds.

    Ledger rows must be built from `position` before calling this, since `roll_rows` and
    `roll_shift` read the current (pre-roll) tenor.
    """
    shift = adapter.roll_shift(position, view, t)
    days_to_expiry = view.days_to_expiry(position.instrument_id, t)
    if days_to_expiry is None:
        raise EngineError(f"no expiry on record for '{position.instrument_id}' to roll against")
    retagged = adapter.retag_legs(position, 1)
    return replace(
        retagged,
        rolled_offset=position.rolled_offset + shift,
        rolled_until_t=t + days_to_expiry,
    )


def reset_roll_tag(
    position: Position, adapter: CommoditiesAdapter, view: MarketView, t: int
) -> Position:
    """Undo a roll's leg retag the day after the expiry day it was rolled toward; a no-op any
    other day, including the expiry day itself.

    The curves are constant-maturity (`M_k` is a fixed function of spot every day), so nothing
    here mirrors a trade: only `rolled_offset` absorbs today's M2-to-M1 raw-level gap, leaving
    `entry_level` untouched so `tracked_level` does not move. A roll is P&L-neutral by
    construction.
    """
    if position.rolled_until_t is None or t <= position.rolled_until_t:
        return position
    level_before = view.level(position.series, t)
    retagged = adapter.retag_legs(position, -1)
    level_after = view.level(retagged.series, t)
    shift = level_after - level_before
    return replace(
        retagged,
        rolled_offset=position.rolled_offset + shift,
        rolled_until_t=None,
        roll_breached=False,
    )


def render_close(
    ctx: "PmContext", position: Position, view: MarketView, t: int, closer: str
) -> tuple[str, date, str]:
    """Render an exit's outcome text; win/loss follows the sign of realised P&L."""
    level_now = tracked_level(position, view, t)
    realized = compute_pnl_unit(position, level_now)
    kind = "win" if realized > 0 else "loss"
    text = render_outcome(
        ctx.catalogue,
        kind=kind,
        pnl=realized,
        unit=position.series.unit,
        closer=closer,
        rng=ctx.rng_for("templates", t, position.trade_idea_id),
    )
    return (position.trade_idea_id, view.dates[t], text)


_Rank3Result = tuple[
    Position | None, tuple[LedgerRow, ...], tuple[str, date, str] | None, PositionAction, bool
]


def _execute_rank3(
    position: Position, rules: Sequence[Rule], ctx: "PmContext", view: MarketView, t: int
) -> _Rank3Result:
    """Trim half when a trim rule is among `rules`, else exit whole on a target rule."""
    trim_rule = next((r for r in rules if r.action == Action.TRIM_HALF), None)
    if trim_rule is not None:
        new_position, rows = trim_half(position, ctx, view, t)
        return new_position, rows, None, PositionAction.TRIM, True
    target_rule = next((r for r in rules if r.action == Action.TARGET), None)
    if target_rule is not None:
        rows = exit_rows(position, ctx, view, t, None, None)
        closed = render_close(ctx, position, view, t, target_rule.param)
        return None, rows, closed, PositionAction.EXIT, True
    return position, (), None, PositionAction.NONE, False


def _execute_fired(
    position: Position,
    view: MarketView,
    t: int,
    ctx: "PmContext",
    params: EffectiveParams,
    fired: Sequence[Rule],
    fired_non_hold: Sequence[Rule],
    pnl_state: PnlState,
) -> tuple[TriggerOutcome, bool]:
    """Draw the exit-deficiency response and execute the winner (and roll's trim/target)."""
    resolution = resolve(fired)
    winner = resolution.winner
    if winner is None:
        raise EngineError("resolve returned no winner for a non-empty fired list")
    date_t = view.dates[t]

    consumed_new = {
        r.rule_id for r in fired if r.scope == RuleScope.IDEA or r.param == "trim_at_target"
    }
    position = replace(position, consumed_rule_ids=position.consumed_rule_ids | consumed_new)
    position = replace(position, triggers_fired=position.triggers_fired + len(fired_non_hold))

    at_loss = pnl_state == PnlState.LOSS
    resp = respond(params, at_loss, ctx.rng_for("response", t, position.trade_idea_id))
    events = [
        RuleEvent(
            pm_id=ctx.persona.pm_id,
            rule_id=r.rule_id,
            trade_idea_id=position.trade_idea_id,
            date_fired=date_t,
            response=resp.response,
            response_date=date_t,
        )
        for r in (winner, *resolution.also_acted)
    ]
    events.extend(
        RuleEvent(
            pm_id=ctx.persona.pm_id,
            rule_id=r.rule_id,
            trade_idea_id=position.trade_idea_id,
            date_fired=date_t,
            response=RuleResponse.OVERRIDDEN,
            response_date=date_t,
        )
        for r in resolution.overridden
    )

    result: Position | None = position
    ledger_rows: list[LedgerRow] = []
    action = PositionAction.NONE
    bias_flag: str | None = None
    closed: tuple[str, date, str] | None = None
    sold = False
    rolled_today = False

    if resp.response == RuleResponse.ACKED_NO_ACTION:
        if winner.action == Action.ROLL:
            result = replace(result, roll_breached=True)
    elif resp.response == RuleResponse.ADDED:
        bias_flag = "exit_deficiency:added"
        cap = mandate_cap(ctx.pm_rules)
        result, add_ledger = apply_add(result, cap, ctx, view, t, bias_flag, winner.rule_id)
        if add_ledger:
            ledger_rows.extend(add_ledger)
            action = PositionAction.ADD
        if winner.action == Action.ROLL:
            result = replace(result, roll_breached=True)
    elif winner.action in (Action.EXIT, Action.SIGNPOST):
        ledger_rows.extend(exit_rows(result, ctx, view, t, None, None))
        closed = render_close(ctx, result, view, t, winner.param)
        result = None
        action = PositionAction.EXIT
        sold = True
    elif winner.action == Action.ROLL:
        rolled_today = True
        result = replace(result, roll_breached=False)
        ledger_rows.extend(roll_rows(result, ctx.adapter, view, t, ctx, None))
        result = apply_roll(result, ctx.adapter, view, t)
        action = PositionAction.ROLL
        result, rank3_rows, rank3_closed, rank3_action, rank3_sold = _execute_rank3(
            result, resolution.also_acted, ctx, view, t
        )
        ledger_rows.extend(rank3_rows)
        if rank3_action != PositionAction.NONE:
            action = rank3_action
            sold = rank3_sold
        closed = rank3_closed
    elif winner.action in (Action.TRIM_HALF, Action.TARGET):
        result, rank3_rows, closed, action, sold = _execute_rank3(
            result, (winner, *resolution.also_acted), ctx, view, t
        )
        ledger_rows.extend(rank3_rows)
    else:
        raise EngineError(f"unexpected winner action '{winner.action.value}' for a trigger")

    outcome = TriggerOutcome(
        position=result,
        events=tuple(events),
        ledger_rows=tuple(ledger_rows),
        action=action,
        bias_flag=bias_flag,
        trigger_pending=True,
        fired_non_hold=len(fired_non_hold),
        closed=closed,
        sold=sold,
        rolled_today=rolled_today,
    )
    return outcome, rolled_today


def _apply_force_roll(
    outcome: TriggerOutcome, ctx: "PmContext", view: MarketView, t: int
) -> TriggerOutcome:
    """Force-roll a position still on its old leg at expiry; no rule fired, no event row."""
    position = outcome.position
    if position is None:
        raise EngineError("cannot force-roll a position that closed earlier today")
    flag = LATE_ROLL_FLAG if position.roll_breached else None
    rows = roll_rows(position, ctx.adapter, view, t, ctx, flag)
    new_position = apply_roll(position, ctx.adapter, view, t)
    action = outcome.action if outcome.action != PositionAction.NONE else PositionAction.ROLL
    bias_flag = join_flags([outcome.bias_flag, flag])
    return replace(
        outcome,
        position=new_position,
        ledger_rows=outcome.ledger_rows + rows,
        action=action,
        bias_flag=bias_flag,
        rolled_today=True,
    )


def handle_triggers(
    position: Position,
    state: PmState,
    view: MarketView,
    t: int,
    ctx: "PmContext",
    idea_rules_for_id: Sequence[Rule],
    params: EffectiveParams,
    pnl: float,
    pnl_state: PnlState,
) -> TriggerOutcome:
    """Evaluate today's rules for `position` and execute the winning response."""
    fields = ctx.adapter.position_fields(position, view, t, state, pnl)
    rules = (*ctx.pm_rules, *idea_rules_for_id)
    if position.rolled_until_t is not None:
        # Already mid-roll: the roll rule stays silent until the retag unwinds at expiry.
        rules = tuple(r for r in rules if r.param != "roll_before_expiry")
    position, fired = evaluate_day(position, rules, fields)
    fired_non_hold = tuple(r for r in fired if r.action != Action.HOLD)

    if not fired or (len(fired) == 1 and fired[0].action == Action.HOLD):
        outcome = TriggerOutcome(
            position=position,
            events=(),
            ledger_rows=(),
            action=PositionAction.NONE,
            bias_flag=None,
            trigger_pending=False,
            fired_non_hold=0,
            closed=None,
            sold=False,
            rolled_today=False,
        )
        rolled_today = False
    else:
        outcome, rolled_today = _execute_fired(
            position, view, t, ctx, params, fired, fired_non_hold, pnl_state
        )

    if (
        outcome.position is not None
        and not rolled_today
        and outcome.position.rolled_until_t is None
        and isinstance(ctx.adapter, CommoditiesAdapter)
        and view.is_expiry_day(outcome.position.instrument_id, t)
    ):
        outcome = _apply_force_roll(outcome, ctx, view, t)

    return outcome
