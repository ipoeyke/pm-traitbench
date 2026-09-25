"""Per-PM inputs bundle: one PM's rows, indexed and paired with its market view.

`build_inputs` partitions the behaviour engine's flat tables by PM and pairs each
PM with the market view its ideas trade against, so every gate 1 estimator reads
one `PmInputs` instead of re-filtering the full tables.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.engine.adapters import adapter_for
from pm_traitbench.engine.adapters.base import series_for_idea
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import Series
from pm_traitbench.enums import AssetClass, Kind, PositionAction, RuleResponse
from pm_traitbench.errors import Gate1Error
from pm_traitbench.tables.schema import (
    DriftEvent,
    Idea,
    LedgerRow,
    Persona,
    PositionDay,
    RuleEvent,
    Trait,
)

_SELL_ACTIONS = frozenset({PositionAction.CUT, PositionAction.TRIM, PositionAction.EXIT})


@dataclass(frozen=True)
class PmInputs:
    """One PM's partitioned rows, market view and day index for gate 1 estimators to read."""

    pm_id: str
    seed: str
    asset_class: AssetClass
    is_real_seed: bool
    traits: Mapping[str, Trait]
    drift_dates: Mapping[str, tuple[date, ...]]
    ideas: tuple[Idea, ...]
    series: Mapping[str, Series]
    entry_risk: Mapping[str, float]
    entry_conviction: Mapping[str, int]
    position_days: tuple[PositionDay, ...]
    rule_events: tuple[RuleEvent, ...]
    sell_dates: frozenset[date]
    acted: frozenset[tuple[str, date]]
    view: MarketView
    day_index: Mapping[date, int]
    horizon_days: int
    engine_counts: Mapping[str, int]

    @property
    def last_date(self) -> date:
        return self.view.dates[-1]


def _group_by_pm(rows: Sequence) -> dict[str, list]:
    by_pm: dict[str, list] = {}
    for row in rows:
        by_pm.setdefault(row.pm_id, []).append(row)
    return by_pm


def _drift_dates(
    pm_traits: Sequence[Trait], pm_drift: Sequence[DriftEvent], bias_traits: Mapping[str, Trait]
) -> dict[str, tuple[date, ...]]:
    trait_id_to_param = {trait.trait_id: trait.param for trait in pm_traits}
    by_param: dict[str, list[date]] = {param: [] for param in BIAS_PARAMS}
    for event in pm_drift:
        param = trait_id_to_param.get(event.trait_id)
        if param in bias_traits:
            by_param[param].append(event.date)
    return {param: tuple(sorted(dates)) for param, dates in by_param.items()}


def _entry_stats(
    pm_ideas: Sequence[Idea], pm_ledger: Sequence[LedgerRow]
) -> tuple[dict[str, float], dict[str, int]]:
    entry_risk: dict[str, float] = {}
    entry_conviction: dict[str, int] = {}
    for idea in pm_ideas:
        entry_rows = [
            row
            for row in pm_ledger
            if row.trade_idea_id == idea.trade_idea_id and row.date == idea.entry_date
        ]
        entry_risk[idea.trade_idea_id] = sum(row.risk_amount for row in entry_rows)
        if entry_rows:
            entry_conviction[idea.trade_idea_id] = entry_rows[0].stated_conviction
    return entry_risk, entry_conviction


def _build_pm_inputs(
    config: Config,
    persona: Persona,
    pm_traits: Sequence[Trait],
    pm_drift: Sequence[DriftEvent],
    pm_ideas: Sequence[Idea],
    pm_ledger: Sequence[LedgerRow],
    pm_rule_events: Sequence[RuleEvent],
    pm_position_days: Sequence[PositionDay],
    views: Mapping[str, MarketView],
    engine_counts: Mapping[str, Mapping[str, int]],
) -> PmInputs:
    pm_id = persona.pm_id
    bias_traits = {trait.param: trait for trait in pm_traits if trait.kind == Kind.BIAS}
    missing = set(BIAS_PARAMS) - set(bias_traits)
    if missing:
        raise Gate1Error(f"PM '{pm_id}' is missing bias trait(s): {sorted(missing)}")
    if pm_id not in engine_counts:
        raise Gate1Error(f"PM '{pm_id}' has no entry in engine_counts")
    if persona.market_seed not in views:
        raise Gate1Error(f"PM '{pm_id}' has no market view for seed '{persona.market_seed}'")

    view = views[persona.market_seed]
    day_index = {day: t for t, day in enumerate(view.dates)}
    adapter = adapter_for(
        persona.mandate.asset_class, persona.mandate.sub_style, config.engine.horizon_days
    )
    series = {idea.trade_idea_id: series_for_idea(idea, adapter) for idea in pm_ideas}
    entry_risk, entry_conviction = _entry_stats(pm_ideas, pm_ledger)
    sell_dates = frozenset(row.date for row in pm_position_days if row.action in _SELL_ACTIONS)
    acted = frozenset(
        (event.trade_idea_id, event.response_date)
        for event in pm_rule_events
        if event.response == RuleResponse.ACTED
    )

    return PmInputs(
        pm_id=pm_id,
        seed=persona.market_seed,
        asset_class=persona.mandate.asset_class,
        is_real_seed=persona.market_seed in config.market.real.seeds,
        traits=bias_traits,
        drift_dates=_drift_dates(pm_traits, pm_drift, bias_traits),
        ideas=tuple(pm_ideas),
        series=series,
        entry_risk=entry_risk,
        entry_conviction=entry_conviction,
        position_days=tuple(pm_position_days),
        rule_events=tuple(pm_rule_events),
        sell_dates=sell_dates,
        acted=acted,
        view=view,
        day_index=day_index,
        horizon_days=config.engine.horizon_days,
        engine_counts=dict(engine_counts[pm_id]),
    )


def kept_personas(personas: Sequence[Persona], skipped: Collection[str]) -> list[Persona]:
    """Direct-asset personas gate 1 covers, sorted by `pm_id`.

    Drops multi-asset personas (no adapter routes their legs) and every
    `pm_id` in `skipped`; shared with the stage module so it builds market
    views for exactly the seeds `build_inputs` will use.
    """
    return [
        persona
        for persona in sorted(personas, key=lambda p: p.pm_id)
        if persona.mandate.asset_class != AssetClass.MULTI_ASSET and persona.pm_id not in skipped
    ]


def build_inputs(
    config: Config,
    personas: Sequence[Persona],
    traits: Sequence[Trait],
    drift_events: Sequence[DriftEvent],
    ideas: Sequence[Idea],
    ledger: Sequence[LedgerRow],
    rule_events: Sequence[RuleEvent],
    position_days: Sequence[PositionDay],
    views: Mapping[str, MarketView],
    engine_counts: Mapping[str, Mapping[str, int]],
    skipped: Collection[str],
) -> list[PmInputs]:
    """One `PmInputs` per direct-asset PM, sorted by `pm_id`.

    Skips multi-asset personas (gate 1 covers single-mandate PMs only) and every
    `pm_id` in `skipped`. Raises `Gate1Error` naming the PM when it lacks any of
    the eight `BIAS_PARAMS` traits, or when it has no entry in `engine_counts`.
    """
    traits_by_pm = _group_by_pm(traits)
    drift_by_pm = _group_by_pm(drift_events)
    ideas_by_pm = _group_by_pm(ideas)
    ledger_by_pm = _group_by_pm(ledger)
    rule_events_by_pm = _group_by_pm(rule_events)
    position_days_by_pm = _group_by_pm(position_days)

    result = []
    for persona in kept_personas(personas, skipped):
        result.append(
            _build_pm_inputs(
                config,
                persona,
                traits_by_pm.get(persona.pm_id, []),
                drift_by_pm.get(persona.pm_id, []),
                ideas_by_pm.get(persona.pm_id, []),
                ledger_by_pm.get(persona.pm_id, []),
                rule_events_by_pm.get(persona.pm_id, []),
                position_days_by_pm.get(persona.pm_id, []),
                views,
                engine_counts,
            )
        )
    return result
