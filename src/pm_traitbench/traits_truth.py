"""Ground truth of a PM's traits as of a date, from the traits table and the drift schedule."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from pm_traitbench.catalogues.models import PreferenceEntry
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import DriftEventType, Kind
from pm_traitbench.errors import CorpusError
from pm_traitbench.tables.schema import DriftEvent, Trait


@dataclass(frozen=True)
class TraitTruth:
    """One param's ground truth: a bias's true activity, or a preference's true held value."""

    kind: Kind
    trait_id: str | None
    truth_active: bool | None
    truth_value: str | None


def bias_active_at(trait: Trait, drift_events: Sequence[DriftEvent], day: date) -> bool:
    """`trait.active`, or False when a dormant event on or before `day` has no later
    revive on or before it; a revive never activates a trait that started inactive.
    """
    own = [e for e in drift_events if e.trait_id == trait.trait_id and e.date <= day]
    for dormant in own:
        if dormant.event != DriftEventType.DORMANT:
            continue
        revived = any(e.event == DriftEventType.REVIVE and e.date > dormant.date for e in own)
        if not revived:
            return False
    return trait.active


def latest_update(
    trait_id: str, drift_events: Sequence[DriftEvent], day: date
) -> DriftEvent | None:
    """The `update` event of `trait_id` with the greatest date on or before `day`, else None."""
    updates = [
        e
        for e in drift_events
        if e.trait_id == trait_id and e.event == DriftEventType.UPDATE and e.date <= day
    ]
    return max(updates, key=lambda e: e.date) if updates else None


def preference_value_at(trait: Trait, drift_events: Sequence[DriftEvent], day: date) -> str:
    """The preference's value on `day`: its latest update, else the trait's own."""
    latest = latest_update(trait.trait_id, drift_events, day)
    return trait.value if latest is None else latest.to_value


def bias_value_at(trait: Trait, drift_events: Sequence[DriftEvent], day: date) -> float:
    """The bias's value on `day`: its latest update, else the trait's own.

    Dormancy is not applied; ask `bias_active_at` first.
    """
    latest = latest_update(trait.trait_id, drift_events, day)
    return float(trait.value if latest is None else latest.to_value)


def compute_truth(
    traits: Sequence[Trait],
    drift_events: Sequence[DriftEvent],
    last_date: date,
    entries: Sequence[PreferenceEntry],
) -> dict[str, TraitTruth]:
    """The ground truth per candidate param: the eight biases, then `entries`' preferences.

    A bias is active as of `last_date` unless left dormant; a preference's value is the
    latest update on or before `last_date`, else the trait's own, else `None` when not
    held. Raises `CorpusError` when `traits` lacks any of the eight bias params.
    """
    bias_by_param = {trait.param: trait for trait in traits if trait.kind == Kind.BIAS}
    missing = set(BIAS_PARAMS) - set(bias_by_param)
    if missing:
        pm_id = traits[0].pm_id if traits else "?"
        raise CorpusError(f"pm {pm_id}: missing bias trait(s) {sorted(missing)}")

    truth: dict[str, TraitTruth] = {}
    for param in BIAS_PARAMS:
        trait = bias_by_param[param]
        truth[param] = TraitTruth(
            kind=Kind.BIAS,
            trait_id=trait.trait_id,
            truth_active=bias_active_at(trait, drift_events, last_date),
            truth_value=None,
        )

    pref_by_param = {trait.param: trait for trait in traits if trait.kind == Kind.PREFERENCE}
    for entry in entries:
        held = pref_by_param.get(entry.param)
        truth_value = (
            preference_value_at(held, drift_events, last_date) if held is not None else None
        )
        truth[entry.param] = TraitTruth(
            kind=Kind.PREFERENCE,
            trait_id=held.trait_id if held is not None else None,
            truth_active=None,
            truth_value=truth_value,
        )
    return truth
