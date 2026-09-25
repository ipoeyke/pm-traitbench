"""The discretionary block: anchoring, loss-side choice and disposition sell
hazard, evaluated in that order once no non-hold trigger fired today.

Pure, like `triggers.py`: `handle_discretionary` never mutates `position`.
"""

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from pm_traitbench.engine.biases import anchoring, disposition, join_flags, loss_aversion
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.state import Position
from pm_traitbench.engine.triggers import (
    apply_add,
    exit_rows,
    find_pm_rule,
    mandate_cap,
    render_close,
)
from pm_traitbench.enums import PnlState, PositionAction
from pm_traitbench.tables.schema import LedgerRow

if TYPE_CHECKING:
    from pm_traitbench.engine.step import PmContext


@dataclass(frozen=True)
class DiscretionaryOutcome:
    """What happened to one open position's discretionary block on one day."""

    position: Position | None
    ledger_rows: tuple[LedgerRow, ...]
    action: PositionAction
    bias_flag: str | None
    anchor_level: float | None
    effective_exit_level: float | None
    closed: tuple[str, date, str] | None
    sold: bool


def handle_discretionary(
    position: Position,
    view: MarketView,
    t: int,
    ctx: "PmContext",
    params: EffectiveParams,
    level_now: float,
    pnl_z: float,
    pnl_state: PnlState,
    progress: float,
) -> DiscretionaryOutcome:
    """Anchoring, then a losing position's cut/hold/add, then the sell hazard."""
    min_holding_rule = find_pm_rule(ctx.pm_rules, "min_holding_period")
    min_holding = float(min_holding_rule.level) if min_holding_rule is not None else 0.0
    can_exit = (t - position.entry_t) >= min_holding

    anchors = ctx.adapter.anchors(position, view, t)
    anchored = anchoring.evaluate(position, level_now, anchors, params)
    anchor_level = anchored.anchor_level
    effective_exit_level = anchored.effective_exit_level

    if anchored.reached and can_exit:
        rows = exit_rows(position, ctx, view, t, None, None)
        closed = render_close(ctx, position, view, t, "discretionary")
        return DiscretionaryOutcome(
            position=None,
            ledger_rows=rows,
            action=PositionAction.EXIT,
            bias_flag=anchoring.flag(params),
            anchor_level=anchor_level,
            effective_exit_level=effective_exit_level,
            closed=closed,
            sold=True,
        )

    result: Position | None = position
    ledger_rows: list[LedgerRow] = []
    flags: list[str | None] = []
    action = PositionAction.NONE
    closed = None
    sold = False
    b_acted = False

    if pnl_state == PnlState.LOSS:
        no_add_rule = find_pm_rule(ctx.pm_rules, "no_add_before_trigger")
        breach = False
        if no_add_rule is not None and position.triggers_fired == 0:
            e = params.value("exit_deficiency")
            u = ctx.rng_for("no_add_breach", t, position.trade_idea_id).uniform()
            breach = u < e
            add_allowed = breach
        else:
            add_allowed = True

        choice = loss_aversion.choose(
            params,
            ctx.rng_for("loss_side", t, position.trade_idea_id),
            add_allowed,
        )

        if choice.action == PositionAction.CUT:
            b_acted = True
            if can_exit:
                ledger_rows.extend(exit_rows(result, ctx, view, t, None, None))
                closed = render_close(ctx, result, view, t, "discretionary")
                result = None
                action = PositionAction.CUT
                sold = True
        elif choice.action == PositionAction.ADD:
            b_acted = True
            if no_add_rule is not None and position.triggers_fired == 0 and breach:
                bias_flag_add: str | None = "loss_aversion:add_before_trigger"
                rule_id_add: str | None = no_add_rule.rule_id
            else:
                bias_flag_add = choice.flag
                rule_id_add = None
            cap = mandate_cap(ctx.pm_rules)
            result, rows = apply_add(result, cap, ctx, view, t, bias_flag_add, rule_id_add)
            if rows:
                ledger_rows.extend(rows)
                flags.append(bias_flag_add)
                action = PositionAction.ADD
        else:
            action = PositionAction.HOLD
            flags.append(choice.flag)

    if result is not None and not b_acted:
        h = disposition.sell_hazard(progress, pnl_state, params, ctx.config)
        drew_sell = disposition.draw_sell(h, ctx.rng_for("hazard", t, position.trade_idea_id))
        executed = drew_sell and can_exit
        if executed:
            ledger_rows.extend(exit_rows(result, ctx, view, t, None, None))
            closed = render_close(ctx, result, view, t, "discretionary")
            flags.append(disposition.flag(pnl_state, progress, True, params))
            result = None
            action = PositionAction.EXIT
            sold = True
        else:
            flags.append(disposition.flag(pnl_state, progress, False, params))

    return DiscretionaryOutcome(
        position=result,
        ledger_rows=tuple(ledger_rows),
        action=action,
        bias_flag=join_flags(flags),
        anchor_level=anchor_level,
        effective_exit_level=effective_exit_level,
        closed=closed,
        sold=sold,
    )
