"""The behaviour engine's per-PM daily loop.

`run_pm` drives `step` across one PM's full horizon and assembles the rows it
produced into a `PmResult`. It is pure: it never touches the data store, only
takes rows and a market view as input and returns rows.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.engine.adapters import adapter_for
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.params import ParamSchedule
from pm_traitbench.engine.rules_eval import required_fields
from pm_traitbench.engine.state import PmState
from pm_traitbench.engine.step import PmContext, step
from pm_traitbench.errors import EngineError
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import (
    DriftEvent,
    Idea,
    LedgerRow,
    Persona,
    PositionDay,
    Rule,
    RuleEvent,
    Trait,
)


@dataclass(frozen=True)
class PmResult:
    """One PM's full-horizon run: every row it produced, plus opportunity counts."""

    ideas: list[Idea]
    idea_rules: list[Rule]
    ledger: list[LedgerRow]
    rule_events: list[RuleEvent]
    position_days: list[PositionDay]
    opportunities: dict[str, int]


def _rule_number(rid: str) -> int:
    return int(rid.split("_", 1)[1])


def run_pm(
    persona: Persona,
    traits: Sequence[Trait],
    pm_rules: Sequence[Rule],
    drift_events: Sequence[DriftEvent],
    view: MarketView,
    config: Config,
    catalogue: Catalogue,
) -> PmResult:
    """Run one PM's full horizon: build its adapter and context, then step
    day by day, collecting rows and finalising each idea when it closes.
    """
    adapter = adapter_for(
        persona.mandate.asset_class, persona.mandate.sub_style, config.engine.horizon_days
    )

    # Checked against the PM-scope rules alone, before any day runs: an idea-scope
    # rule referencing an unknown field would only surface once that idea exists.
    missing = required_fields(pm_rules) - adapter.FIELDS
    if missing:
        offending = min((r for r in pm_rules if r.field in missing), key=lambda r: r.rule_id)
        raise EngineError(
            f"rule '{offending.rule_id}' references field '{offending.field}' unknown to "
            f"the '{persona.mandate.asset_class.value}' adapter"
        )

    universe = adapter.universe(view.instruments, pm_rules)
    schedule = ParamSchedule.build(traits, drift_events, config)

    def rng_for(purpose: str, *keys: str | int):
        return stream(config.seed.root, "engine", persona.pm_id, purpose, *keys)

    ctx = PmContext(
        persona=persona,
        traits=tuple(traits),
        pm_rules=tuple(pm_rules),
        adapter=adapter,
        universe=universe,
        catalogue=catalogue,
        config=config,
        schedule=schedule,
        rng_for=rng_for,
    )

    max_rule_n = max((_rule_number(r.rule_id) for r in pm_rules), default=0)
    state = PmState(pm_id=persona.pm_id, positions=(), next_idea=1, next_rule=max_rule_n + 1)

    idea_rules: dict[str, tuple[Rule, ...]] = {}
    open_ideas: dict[str, Idea] = {}
    closed_ideas: dict[str, Idea] = {}
    ledger: list[LedgerRow] = []
    rule_events: list[RuleEvent] = []
    position_days: list[PositionDay] = []
    idea_rule_rows: list[Rule] = []
    opportunities: Counter[str] = Counter()

    for t in range(view.n_days):
        state, day_out = step(state, t, view, ctx, idea_rules)

        ledger.extend(day_out.ledger_rows)
        rule_events.extend(day_out.rule_events)
        position_days.extend(day_out.position_days)
        idea_rule_rows.extend(day_out.idea_rules)
        opportunities.update(day_out.opportunities)

        for idea in day_out.new_ideas:
            open_ideas[idea.trade_idea_id] = idea

        by_idea: dict[str, list[Rule]] = {}
        for rule in day_out.idea_rules:
            by_idea.setdefault(rule.trade_idea_id, []).append(rule)  # type: ignore[arg-type]
        for trade_idea_id, rules in by_idea.items():
            idea_rules[trade_idea_id] = tuple(rules)

        for trade_idea_id, exit_date, outcome in day_out.closed:
            if trade_idea_id not in open_ideas:
                raise EngineError(f"trade idea '{trade_idea_id}' closed twice")
            idea = open_ideas.pop(trade_idea_id)
            closed_ideas[trade_idea_id] = idea.model_copy(
                update={"exit_date": exit_date, "outcome": outcome}
            )

    if open_ideas:
        raise EngineError(f"idea(s) {sorted(open_ideas)} never closed by the end of the horizon")

    return PmResult(
        ideas=list(closed_ideas.values()),
        idea_rules=idea_rule_rows,
        ledger=ledger,
        rule_events=rule_events,
        position_days=position_days,
        opportunities=dict(opportunities),
    )
