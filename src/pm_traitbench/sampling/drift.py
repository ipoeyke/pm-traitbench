"""Drift sampling: dated events where a PM's trait changes over the horizon."""

from numpy.random import Generator

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.enums import DriftEventType, Kind
from pm_traitbench.errors import SamplingError
from pm_traitbench.tables.schema import DriftEvent, Trait
from pm_traitbench.timeline import Timeline


def sample_drift(
    pm_id: str,
    traits: list[Trait],
    config: Config,
    catalogue: Catalogue,
    timeline: Timeline,
    rng: Generator,
) -> list[DriftEvent]:
    """Sample one PM's drift events: a bias update, and maybe a preference
    replacement and a dormant/revive pair on another active bias.

    Draw order is fixed for stream alignment: the bias update, then u1 and the
    preference branch, then u2 and the dormant/revive branch. u1 and u2 are
    always drawn even when their branch cannot apply.
    """
    drift = config.drift
    active_bias_traits = [t for t in traits if t.kind == Kind.BIAS and t.active]
    if not active_bias_traits:
        raise SamplingError(f"pm '{pm_id}' has no active bias trait to drift")
    preference_traits = [t for t in traits if t.kind == Kind.PREFERENCE]

    bias_event = _sample_bias_update(pm_id, active_bias_traits, config, timeline, rng)
    events: list[DriftEvent] = [bias_event]

    u1 = rng.random()
    if u1 < drift.p_preference_update and preference_traits:
        event = _sample_preference_update(
            pm_id, preference_traits, config, catalogue, timeline, rng
        )
        if event is not None:
            events.append(event)

    u2 = rng.random()
    others = [t for t in active_bias_traits if t.trait_id != bias_event.trait_id]
    if u2 < drift.p_dormant_revive and others:
        events.extend(_sample_dormant_revive(pm_id, others, config, timeline, rng))

    return sorted(events, key=lambda e: (e.date, e.trait_id, e.event.value))


def _sample_bias_update(
    pm_id: str, active_bias_traits: list[Trait], config: Config, timeline: Timeline, rng: Generator
) -> DriftEvent:
    updated = active_bias_traits[rng.integers(len(active_bias_traits))]
    dates = timeline.weekdays_in_weeks(*config.drift.bias_update_weeks)
    date = dates[rng.integers(len(dates))]
    remaining = rng.uniform(*config.drift.bias_update_remaining)

    neutral = config.biases.params[updated.param].neutral.median_value()
    from_value = float(updated.value)
    to_value = round(neutral + remaining * (from_value - neutral), 4)

    return DriftEvent(
        pm_id=pm_id,
        date=date,
        event=DriftEventType.UPDATE,
        trait_id=updated.trait_id,
        from_value=from_value,
        to_value=to_value,
    )


def _sample_preference_update(
    pm_id: str,
    preference_traits: list[Trait],
    config: Config,
    catalogue: Catalogue,
    timeline: Timeline,
    rng: Generator,
) -> DriftEvent | None:
    trait = preference_traits[rng.integers(len(preference_traits))]
    dates = timeline.weekdays_in_weeks(*config.drift.preference_update_weeks)
    date = dates[rng.integers(len(dates))]

    entry = next((e for e in catalogue.preferences if e.param == trait.param), None)
    other_values = [v for v in entry.values if v != trait.value] if entry is not None else []
    if not other_values:
        return None

    to_value = other_values[rng.integers(len(other_values))]
    return DriftEvent(
        pm_id=pm_id,
        date=date,
        event=DriftEventType.UPDATE,
        trait_id=trait.trait_id,
        from_value=trait.value,
        to_value=to_value,
    )


def _sample_dormant_revive(
    pm_id: str, others: list[Trait], config: Config, timeline: Timeline, rng: Generator
) -> list[DriftEvent]:
    trait = others[rng.integers(len(others))]
    dormant_dates = timeline.weekdays_in_weeks(*config.drift.dormant_weeks)
    dormant_date = dormant_dates[rng.integers(len(dormant_dates))]
    revive_dates = timeline.weekdays_in_weeks(*config.drift.revive_weeks)
    revive_date = revive_dates[rng.integers(len(revive_dates))]

    return [
        DriftEvent(
            pm_id=pm_id,
            date=dormant_date,
            event=DriftEventType.DORMANT,
            trait_id=trait.trait_id,
            from_value=None,
            to_value=None,
        ),
        DriftEvent(
            pm_id=pm_id,
            date=revive_date,
            event=DriftEventType.REVIVE,
            trait_id=trait.trait_id,
            from_value=None,
            to_value=None,
        ),
    ]
