"""Engine stage: run the behaviour engine for every non-multi-asset PM and
write its ideas, ledger, rule events and position days.
"""

from typing import Any

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.engine.loop import run_pm
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.enums import AssetClass, RuleScope
from pm_traitbench.market.axis import build_axis
from pm_traitbench.stages import Append, Stage
from pm_traitbench.tables.schema import (
    DriftEvent,
    Idea,
    LedgerRow,
    PositionDay,
    Rule,
    RuleEvent,
    Trait,
)
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    ENGINE_TABLES,
    IDEAS,
    LEDGER,
    MARKET_CALENDAR,
    MARKET_CONSENSUS,
    MARKET_CURVES,
    MARKET_INSTRUMENTS,
    MARKET_PRICES,
    MARKET_REGIMES,
    MARKET_TABLES,
    PERSONAS,
    POSITION_DAYS,
    RULE_EVENTS,
    RULES,
    TRAITS,
)
from pm_traitbench.tables.store import DataStore


def _group_by_pm(rows: list) -> dict[str, list]:
    by_pm: dict[str, list] = {}
    for row in rows:
        by_pm.setdefault(row.pm_id, []).append(row)
    return by_pm


def run(config: Config, store: DataStore) -> dict[str, Any]:
    """Run every PM's daily loop and write the four engine tables plus the
    rules table: its PM-scope rows followed by this run's idea-scope rules.

    One `MarketView` is built per distinct market seed among the personas, on
    the config's published horizon; a `multi_asset` persona has no adapter
    yet, so it is skipped rather than run.
    """
    personas = store.read(PERSONAS)
    traits: list[Trait] = store.read(TRAITS)
    # Idea-scope rows belong to an earlier engine run and are replaced, never reused.
    rules: list[Rule] = [rule for rule in store.read(RULES) if rule.scope == RuleScope.PM]
    drift_events: list[DriftEvent] = store.read(DRIFT_EVENTS)
    instruments = store.read(MARKET_INSTRUMENTS)
    prices = store.read(MARKET_PRICES)
    curves = store.read(MARKET_CURVES)
    consensus = store.read(MARKET_CONSENSUS)
    calendar = store.read(MARKET_CALENDAR)
    regimes = store.read(MARKET_REGIMES)

    axis = build_axis(config.timeline(), config.market.burn_in_days)
    dates = axis.dates[axis.horizon]

    catalogue = load_catalogue()

    views: dict[str, MarketView] = {}
    for seed in sorted({persona.market_seed for persona in personas}):
        views[seed] = MarketView.build(
            seed=seed,
            dates=dates,
            instruments=instruments,
            prices=prices,
            curves=curves,
            consensus=consensus,
            calendar=calendar,
            regimes=regimes,
        )

    traits_by_pm = _group_by_pm(traits)
    rules_by_pm = _group_by_pm(rules)
    drift_by_pm = _group_by_pm(drift_events)

    ideas: list[Idea] = []
    idea_rules: list[Rule] = []
    ledger: list[LedgerRow] = []
    rule_events: list[RuleEvent] = []
    position_days: list[PositionDay] = []
    ideas_per_pm: dict[str, int] = {}
    opportunities: dict[str, dict[str, int]] = {}
    skipped: list[str] = []

    for persona in sorted(personas, key=lambda p: p.pm_id):
        if persona.mandate.asset_class == AssetClass.MULTI_ASSET:
            skipped.append(persona.pm_id)
            continue

        result = run_pm(
            persona,
            traits_by_pm.get(persona.pm_id, []),
            rules_by_pm.get(persona.pm_id, []),
            drift_by_pm.get(persona.pm_id, []),
            views[persona.market_seed],
            config,
            catalogue,
        )
        ideas.extend(result.ideas)
        idea_rules.extend(result.idea_rules)
        ledger.extend(result.ledger)
        rule_events.extend(result.rule_events)
        position_days.extend(result.position_days)
        ideas_per_pm[persona.pm_id] = len(result.ideas)
        opportunities[persona.pm_id] = result.opportunities

    store.write(IDEAS, ideas)
    store.write(LEDGER, ledger)
    store.write(RULE_EVENTS, rule_events)
    store.write(POSITION_DAYS, position_days)
    store.write(RULES, [*rules, *idea_rules])

    return {"ideas_per_pm": ideas_per_pm, "opportunities": opportunities, "skipped": skipped}


ENGINE_STAGE = Stage(
    number=3,
    name="engine",
    help="run the behaviour engine: ideas, ledger, rule events per PM",
    run=run,
    reads=(PERSONAS, TRAITS, RULES, DRIFT_EVENTS, *MARKET_TABLES),
    writes=ENGINE_TABLES,
    appends=(Append(RULES, owned=lambda record: record["scope"] == RuleScope.IDEA),),
)
