"""Per-PM inputs bundle: one PM's rows from the earlier stages, paired with the trading calendar.

`build_inputs` partitions the flat stage 1 and stage 3 tables by PM, so the rest of
this package reads one `PlanInputs` per PM instead of re-filtering the full tables.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from pm_traitbench.config import Config
from pm_traitbench.enums import DriftEventType
from pm_traitbench.errors import PlanError
from pm_traitbench.market.axis import build_axis
from pm_traitbench.tables.schema import (
    DriftEvent,
    Idea,
    LedgerRow,
    Persona,
    PositionDay,
    RuleEvent,
    Trait,
)


@dataclass(frozen=True)
class PlanInputs:
    """One PM's partitioned rows and the trading days the plan may place a session on."""

    persona: Persona
    traits: tuple[Trait, ...]
    drift_events: tuple[DriftEvent, ...]
    ideas: Mapping[str, Idea]
    ledger: tuple[LedgerRow, ...]
    rule_events: tuple[RuleEvent, ...]
    position_days: tuple[PositionDay, ...]
    trading_days: tuple[date, ...]

    def trait(self, trait_id: str) -> Trait:
        """The PM's trait with this id, or `PlanError` when it has none."""
        for candidate in self.traits:
            if candidate.trait_id == trait_id:
                return candidate
        raise PlanError(f"PM '{self.persona.pm_id}' has no trait '{trait_id}'")

    def dormant_windows(self, trait_id: str) -> tuple[tuple[date, date], ...]:
        """Half-open `[dormant_date, revive_date)` windows, one per `dormant` event.

        Each dormant date pairs with the first later `revive` of the same trait; a
        dormant date with no later revive stays dormant through `date.max`.
        """
        dormant_dates = sorted(
            event.date
            for event in self.drift_events
            if event.trait_id == trait_id and event.event == DriftEventType.DORMANT
        )
        revive_dates = sorted(
            event.date
            for event in self.drift_events
            if event.trait_id == trait_id and event.event == DriftEventType.REVIVE
        )
        windows = []
        for dormant_date in dormant_dates:
            later_revives = [r for r in revive_dates if r > dormant_date]
            end = min(later_revives) if later_revives else date.max
            windows.append((dormant_date, end))
        return tuple(windows)

    def is_dormant(self, trait_id: str, day: date) -> bool:
        """Whether `day` falls in one of the trait's dormant windows."""
        return any(start <= day < end for start, end in self.dormant_windows(trait_id))

    def value_at(self, trait_id: str, day: date) -> float | str:
        """The trait's value on `day`: its base value with every `update` up to `day` applied."""
        trait = self.trait(trait_id)
        value = trait.value
        updates = sorted(
            (
                event
                for event in self.drift_events
                if event.trait_id == trait_id
                and event.event == DriftEventType.UPDATE
                and event.date <= day
            ),
            key=lambda event: event.date,
        )
        for event in updates:
            value = event.to_value
        return value


def trading_days(config: Config) -> tuple[date, ...]:
    """The engine's published horizon dates: every date the engine could have written a row on."""
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    return tuple(axis.dates[axis.horizon])


def _group_by_pm(rows: Sequence) -> dict[str, list]:
    by_pm: dict[str, list] = {}
    for row in rows:
        by_pm.setdefault(row.pm_id, []).append(row)
    return by_pm


def _build_pm_inputs(
    persona: Persona,
    pm_traits: Sequence[Trait],
    pm_drift: Sequence[DriftEvent],
    pm_ideas: Sequence[Idea],
    pm_ledger: Sequence[LedgerRow],
    pm_rule_events: Sequence[RuleEvent],
    pm_position_days: Sequence[PositionDay],
    days: tuple[date, ...],
) -> PlanInputs:
    pm_id = persona.pm_id
    traits = tuple(sorted(pm_traits, key=lambda t: t.trait_id))
    trait_ids = {t.trait_id for t in traits}
    drift_events = tuple(sorted(pm_drift, key=lambda e: (e.date, e.trait_id, e.event)))
    for event in drift_events:
        if event.trait_id not in trait_ids:
            raise PlanError(f"PM '{pm_id}': drift event names unknown trait '{event.trait_id}'")

    ideas = {idea.trade_idea_id: idea for idea in sorted(pm_ideas, key=lambda i: i.trade_idea_id)}
    ledger = tuple(sorted(pm_ledger, key=lambda r: (r.date, r.trade_idea_id)))
    rule_events = tuple(pm_rule_events)
    position_days = tuple(sorted(pm_position_days, key=lambda r: (r.date, r.trade_idea_id)))
    for row in (*ledger, *rule_events, *position_days):
        if row.trade_idea_id not in ideas:
            raise PlanError(f"PM '{pm_id}': row names unknown trade idea '{row.trade_idea_id}'")

    return PlanInputs(
        persona=persona,
        traits=traits,
        drift_events=drift_events,
        ideas=ideas,
        ledger=ledger,
        rule_events=rule_events,
        position_days=position_days,
        trading_days=days,
    )


def build_inputs(
    personas: Sequence[Persona],
    traits: Sequence[Trait],
    drift_events: Sequence[DriftEvent],
    ideas: Sequence[Idea],
    ledger: Sequence[LedgerRow],
    rule_events: Sequence[RuleEvent],
    position_days: Sequence[PositionDay],
    days: tuple[date, ...],
    skipped: Collection[str],
) -> list[PlanInputs]:
    """One `PlanInputs` per persona not in `skipped`, sorted by `pm_id`.

    Raises `PlanError` naming the PM and the id when a drift event's `trait_id` is
    not one of that PM's traits, or a ledger, rule-event or position-day row names
    a `trade_idea_id` missing from that PM's ideas.
    """
    traits_by_pm = _group_by_pm(traits)
    drift_by_pm = _group_by_pm(drift_events)
    ideas_by_pm = _group_by_pm(ideas)
    ledger_by_pm = _group_by_pm(ledger)
    rule_events_by_pm = _group_by_pm(rule_events)
    position_days_by_pm = _group_by_pm(position_days)

    result = []
    for persona in sorted(personas, key=lambda p: p.pm_id):
        if persona.pm_id in skipped:
            continue
        pm_id = persona.pm_id
        result.append(
            _build_pm_inputs(
                persona,
                traits_by_pm.get(pm_id, []),
                drift_by_pm.get(pm_id, []),
                ideas_by_pm.get(pm_id, []),
                ledger_by_pm.get(pm_id, []),
                rule_events_by_pm.get(pm_id, []),
                position_days_by_pm.get(pm_id, []),
                days,
            )
        )
    return result
