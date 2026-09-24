"""The daily step: marks each open position, resolves rule triggers and the
discretionary block, enters new ideas, and closes out the book at the
horizon end.

`step` is pure: it never mutates `state`, only returns a new one alongside
the day's `DayOutput`.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.engine.adapters.base import Adapter, tracked_level
from pm_traitbench.engine.adapters.base import pnl_unit as compute_pnl_unit
from pm_traitbench.engine.adapters.commodities import CommoditiesAdapter
from pm_traitbench.engine.biases import join_flags
from pm_traitbench.engine.discretionary import handle_discretionary
from pm_traitbench.engine.ideas import entries_for_day
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.params import ParamSchedule
from pm_traitbench.engine.state import PmState
from pm_traitbench.engine.templates import render_outcome
from pm_traitbench.engine.triggers import exit_rows, handle_triggers, reset_roll_tag
from pm_traitbench.enums import PnlState, PositionAction
from pm_traitbench.tables.schema import (
    Idea,
    LedgerRow,
    Persona,
    PositionDay,
    Rule,
    RuleEvent,
    Trait,
)

_OPPORTUNITY_KEYS = (
    "loss_side_untriggered_days",
    "triggers_fired",
    "conflict_entries",
    "exits",
    "sell_day_position_days",
    "ideas",
    "entries_after_run",
)


@dataclass(frozen=True)
class PmContext:
    """What one PM's daily step does not need to recompute per day."""

    persona: Persona
    traits: tuple[Trait, ...]
    pm_rules: tuple[Rule, ...]
    adapter: Adapter
    universe: tuple[str, ...]
    catalogue: Catalogue
    config: Config
    schedule: ParamSchedule
    rng_for: Callable[..., np.random.Generator]


@dataclass(frozen=True)
class DayOutput:
    """Everything one PM's daily step produced."""

    ledger_rows: tuple[LedgerRow, ...]
    rule_events: tuple[RuleEvent, ...]
    position_days: tuple[PositionDay, ...]
    new_ideas: tuple[Idea, ...]
    idea_rules: tuple[Rule, ...]
    closed: tuple[tuple[str, date, str], ...]
    opportunities: "Counter[str]"


def _pnl_state(pnl_z: float) -> PnlState:
    if abs(pnl_z) < 1e-9:
        return PnlState.FLAT
    return PnlState.GAIN if pnl_z > 0 else PnlState.LOSS


def step(
    state: PmState,
    t: int,
    view: MarketView,
    ctx: PmContext,
    idea_rules: Mapping[str, tuple[Rule, ...]],
) -> tuple[PmState, DayOutput]:
    """Advance one PM's book by one day: mark, trigger, discretionary, entries, close-out."""
    params = ctx.schedule.for_day(view.dates[t], view.regime(t))

    positions_at_start = state.positions
    n_at_start = len(positions_at_start)
    running_state = state

    events: list[RuleEvent] = []
    ledger_rows: list[LedgerRow] = []
    closed: list[tuple[str, date, str]] = []
    position_days: dict[str, PositionDay] = {}
    opportunities: Counter[str] = Counter({key: 0 for key in _OPPORTUNITY_KEYS})
    any_sold = False

    for pos in positions_at_start:
        level_now = tracked_level(pos, view, t)
        pnl = compute_pnl_unit(pos, level_now)
        pnl_z = pnl / pos.sd_h_at_entry
        pnl_state = _pnl_state(pnl_z)

        trig = handle_triggers(
            pos,
            running_state,
            view,
            t,
            ctx,
            idea_rules.get(pos.trade_idea_id, ()),
            params,
            pnl,
            pnl_state,
        )
        events.extend(trig.events)
        ledger_rows.extend(trig.ledger_rows)
        opportunities["triggers_fired"] += trig.fired_non_hold

        action = trig.action
        bias_flag = trig.bias_flag
        anchor_level: float | None = None
        effective_exit_level: float | None = None
        sold_today = trig.sold
        final_position = trig.position

        if trig.closed is not None:
            closed.append(trig.closed)
            opportunities["exits"] += 1

        if final_position is not None and trig.fired_non_hold == 0:
            # Re-mark from `final_position`, not the stale `level_now`/`pnl` above: a same-day
            # force roll can retag its legs before discretionary runs.
            cur_level_now = tracked_level(final_position, view, t)
            cur_pnl = compute_pnl_unit(final_position, cur_level_now)
            cur_pnl_z = cur_pnl / final_position.sd_h_at_entry
            cur_pnl_state = _pnl_state(cur_pnl_z)
            cur_progress = cur_pnl / abs(final_position.target_level - final_position.entry_level)
            disc = handle_discretionary(
                final_position,
                view,
                t,
                ctx,
                params,
                cur_level_now,
                cur_pnl_z,
                cur_pnl_state,
                cur_progress,
            )
            ledger_rows.extend(disc.ledger_rows)
            anchor_level = disc.anchor_level
            effective_exit_level = disc.effective_exit_level
            if disc.action != PositionAction.NONE:
                action = disc.action
            bias_flag = join_flags([bias_flag, disc.bias_flag])
            sold_today = sold_today or disc.sold
            if disc.closed is not None:
                closed.append(disc.closed)
                opportunities["exits"] += 1
            final_position = disc.position

        if final_position is not None and isinstance(ctx.adapter, CommoditiesAdapter):
            # A no-op unless today is the day after the expiry day this position rolled toward.
            final_position = reset_roll_tag(final_position, ctx.adapter, view, t)

        if final_position is None:
            running_state = running_state.remove_position(pos.trade_idea_id)
        else:
            running_state = running_state.replace_position(final_position)

        if pnl_state == PnlState.LOSS and trig.fired_non_hold == 0:
            opportunities["loss_side_untriggered_days"] += 1
        if sold_today:
            any_sold = True

        position_days[pos.trade_idea_id] = PositionDay(
            pm_id=ctx.persona.pm_id,
            date=view.dates[t],
            trade_idea_id=pos.trade_idea_id,
            pnl_unit=pnl,
            pnl_z=pnl_z,
            pnl_state=pnl_state,
            sessions_held=t - pos.entry_t,
            triggers_fired=trig.fired_non_hold,
            trigger_pending=trig.trigger_pending,
            action=action,
            bias_flag=bias_flag,
            anchor_level=anchor_level,
            effective_exit_level=effective_exit_level,
        )

    running_state, new_ideas = entries_for_day(
        running_state,
        t,
        view,
        ctx.adapter,
        params,
        ctx.persona,
        ctx.pm_rules,
        ctx.traits,
        ctx.universe,
        ctx.config,
        ctx.catalogue,
        ctx.rng_for,
    )
    for new_idea in new_ideas:
        ledger_rows.extend(new_idea.ledger_rows)
    idea_rows = tuple(new_idea.idea for new_idea in new_ideas)
    idea_rule_rows = tuple(rule for new_idea in new_ideas for rule in new_idea.rules)
    opportunities["ideas"] += len(new_ideas)
    opportunities["conflict_entries"] += sum(1 for new_idea in new_ideas if new_idea.conflict)
    opportunities["entries_after_run"] += sum(
        1 for new_idea in new_ideas if new_idea.entered_after_run
    )

    if t == view.n_days - 1:
        for pos in running_state.positions:
            level_now = tracked_level(pos, view, t)
            realized = compute_pnl_unit(pos, level_now)
            text = render_outcome(
                ctx.catalogue,
                kind="open",
                pnl=realized,
                unit=pos.series.unit,
                closer="horizon_end",
                rng=ctx.rng_for("templates", t, pos.trade_idea_id),
            )
            ledger_rows.extend(exit_rows(pos, ctx, view, t, None, None))
            closed.append((pos.trade_idea_id, view.dates[t], text))
            opportunities["exits"] += 1
            any_sold = True
            running_state = running_state.remove_position(pos.trade_idea_id)
            existing = position_days.get(pos.trade_idea_id)
            if existing is not None:
                position_days[pos.trade_idea_id] = existing.model_copy(
                    update={"action": PositionAction.EXIT, "bias_flag": None}
                )

    if any_sold:
        opportunities["sell_day_position_days"] += n_at_start

    return running_state, DayOutput(
        ledger_rows=tuple(ledger_rows),
        rule_events=tuple(events),
        position_days=tuple(position_days[pid] for pid in sorted(position_days)),
        new_ideas=idea_rows,
        idea_rules=idea_rule_rows,
        closed=tuple(closed),
        opportunities=opportunities,
    )
