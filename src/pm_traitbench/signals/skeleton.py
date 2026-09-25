"""Skeleton rendering: turns each PM's planned sessions into hidden `Skeleton` rows.

A skeleton's stances are short instruction lines a later LLM narrator turns into
the PM's own words; the advisor-violation line and the forbidden trait and
preference-param lists ride along on the same row so a narrator's paraphrase
never surfaces a trait the plan never signalled, or breaches a preference the
PM has not been given.
"""

from datetime import date

import numpy as np

from pm_traitbench.catalogues.loader import STANCE_SLOTS, render_stance
from pm_traitbench.catalogues.models import Catalogue, PreferenceGroup
from pm_traitbench.enums import AssetClass, Kind, Ownership, SessionKind, SignalMode, StanceEntry
from pm_traitbench.signals.assemble import Assembly, PlacedSignal
from pm_traitbench.signals.inputs import PlanInputs
from pm_traitbench.tables.schema import Skeleton, Stance, Trait

_WHO_BY_OWNERSHIP: dict[Ownership, str] = {
    Ownership.COLLEAGUE: "a colleague",
    Ownership.CLIENT: "a client",
}


def forbidden_sets(
    inputs: PlanInputs, catalogue: Catalogue
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(inactive bias trait_ids sorted, catalogue preference params for the PM's asset class
    that the PM does not hold, sorted).
    """
    inactive_trait_ids = tuple(
        sorted(
            trait.trait_id
            for trait in inputs.traits
            if trait.kind == Kind.BIAS and not trait.active
        )
    )
    held_params = {trait.param for trait in inputs.traits if trait.kind == Kind.PREFERENCE}
    asset_class = inputs.persona.mandate.asset_class
    unheld_params = tuple(
        sorted(
            entry.param
            for entry in catalogue.preferences_for(asset_class)
            if entry.param not in held_params
        )
    )
    return inactive_trait_ids, unheld_params


def format_level(x: float) -> str:
    """Render a price or level to at most 4 significant figures."""
    return f"{x:.4g}"


def _draw_line(lines: tuple[str, ...], rng: np.random.Generator) -> str:
    return str(lines[rng.integers(len(lines))])


def _claim_stance(
    ps: PlacedSignal,
    trait: Trait,
    catalogue: Catalogue,
    asset_class: AssetClass,
    rng: np.random.Generator,
) -> Stance:
    lines = catalogue.stances.lines(trait.param, StanceEntry.CLAIM, asset_class)
    line = _draw_line(lines, rng)
    return Stance(
        signal_id=ps.signal_id,
        trait_id=trait.trait_id,
        mode=SignalMode.CONTRADICTION,
        entry=StanceEntry.CLAIM,
        stance=render_stance(line, {}),
    )


def _drift_event_values(inputs: PlanInputs, ps: PlacedSignal) -> tuple[str | float, str | float]:
    """(from_value, to_value) of the drift event a `DRIFT_UPDATE` note announces."""
    drift_date = ps.planned.drift_date
    for event in inputs.drift_events:
        if event.trait_id == ps.planned.trait_id and event.date == drift_date:
            assert event.from_value is not None and event.to_value is not None
            return event.from_value, event.to_value
    raise AssertionError(
        f"drift note for trait '{ps.planned.trait_id}' names no matching drift event "
        f"on {drift_date}"
    )


def _signal_stance(
    inputs: PlanInputs,
    catalogue: Catalogue,
    trait: Trait,
    ps: PlacedSignal,
    session_date: date,
    asset_class: AssetClass,
    pref_groups: dict[str, PreferenceGroup],
    rng: np.random.Generator,
) -> tuple[Stance, str | None]:
    """One non-claim stance, plus the preference value backing an advisor violation, if any."""
    entry = ps.planned.entry
    is_bias = trait.kind == Kind.BIAS
    key: str | PreferenceGroup = trait.param if is_bias else pref_groups[trait.param]
    allowed = STANCE_SLOTS[(trait.kind, entry)]
    values: dict[str, str] = {}
    violation_value: str | None = None

    if entry == StanceEntry.THIRD_PARTY:
        values["who"] = _WHO_BY_OWNERSHIP[ps.planned.ownership]
        if not is_bias:
            values["value"] = str(ps.planned.third_party_value)
    elif entry == StanceEntry.STATED or entry == StanceEntry.RETRACT:
        if not is_bias:
            values["value"] = str(inputs.value_at(trait.trait_id, session_date))
    elif entry == StanceEntry.REVEALED_REACTION:
        values["value"] = str(inputs.value_at(trait.trait_id, session_date))
        violation_value = values["value"]
    elif entry == StanceEntry.DRIFT_UPDATE and not is_bias:
        from_value, to_value = _drift_event_values(inputs, ps)
        values["old_value"] = str(from_value)
        values["value"] = str(to_value)
    elif entry == StanceEntry.REVEALED:
        assert ps.trade_idea_id is not None
        idea = inputs.ideas[ps.trade_idea_id]
        if is_bias:
            values["instrument"] = idea.instrument_id
            values["entry"] = format_level(idea.entry_level)
            values["target"] = format_level(idea.target_level)
            values["stop"] = format_level(idea.stop_level)
        else:
            values["instrument"] = idea.instrument_id
            values["value"] = str(inputs.value_at(trait.trait_id, session_date))
    # DRIFT_DORMANT and DRIFT_REVIVE (bias only) and bias DRIFT_UPDATE take no slots.

    slots = {name: value for name, value in values.items() if name in allowed}

    if entry == StanceEntry.REVEALED and is_bias:
        assert ps.carrier is not None
        lines = catalogue.stances.revealed_lines(trait.param, ps.carrier.pattern, asset_class)
    else:
        lines = catalogue.stances.lines(key, entry, asset_class)
    line = _draw_line(lines, rng)

    stance = Stance(
        signal_id=ps.signal_id,
        trait_id=trait.trait_id,
        mode=ps.planned.mode,
        entry=entry,
        stance=render_stance(line, slots),
    )
    return stance, violation_value


def render_skeletons(
    inputs: PlanInputs, assembly: Assembly, catalogue: Catalogue, rng: np.random.Generator
) -> list[Skeleton]:
    """One `Skeleton` per planned session, in session order."""
    pm_id = inputs.persona.pm_id
    asset_class = inputs.persona.mandate.asset_class
    forbidden_trait_ids, forbidden_pref_params = forbidden_sets(inputs, catalogue)
    pref_groups = {entry.param: entry.group for entry in catalogue.preferences}
    traits_by_id = {trait.trait_id: trait for trait in inputs.traits}

    skeletons = []
    for session in assembly.sessions:
        if session.kind == SessionKind.SILENCE:
            skeletons.append(
                Skeleton(
                    session_id=session.session_id,
                    pm_id=pm_id,
                    date=session.date,
                    kind=session.kind,
                    trade_idea_ids=session.trade_idea_ids,
                    stances=(),
                    advisor_violation=None,
                    forbidden_trait_ids=forbidden_trait_ids,
                    forbidden_pref_params=forbidden_pref_params,
                )
            )
            continue

        stances: list[Stance] = []
        violation_group: PreferenceGroup | None = None
        violation_value: str | None = None

        for ps in session.claims:
            trait = traits_by_id[ps.planned.trait_id]
            stances.append(_claim_stance(ps, trait, catalogue, asset_class, rng))

        for ps in session.signals:
            trait = traits_by_id[ps.planned.trait_id]
            stance, value = _signal_stance(
                inputs, catalogue, trait, ps, session.date, asset_class, pref_groups, rng
            )
            stances.append(stance)
            if stance.entry == StanceEntry.REVEALED_REACTION:
                violation_group = pref_groups[trait.param]
                violation_value = value

        advisor_violation = None
        if violation_group is not None:
            assert violation_value is not None
            lines = catalogue.stances.lines(violation_group, StanceEntry.VIOLATION, asset_class)
            line = _draw_line(lines, rng)
            advisor_violation = render_stance(line, {"value": violation_value})

        skeletons.append(
            Skeleton(
                session_id=session.session_id,
                pm_id=pm_id,
                date=session.date,
                kind=session.kind,
                trade_idea_ids=session.trade_idea_ids,
                stances=tuple(stances),
                advisor_violation=advisor_violation,
                forbidden_trait_ids=forbidden_trait_ids,
                forbidden_pref_params=forbidden_pref_params,
            )
        )
    return skeletons
