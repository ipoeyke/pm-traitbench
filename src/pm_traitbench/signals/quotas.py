"""Per-trait signal quotas: how many signals of each kind a PM's plan needs, and in what
date window, before task 7 places them on dated sessions.

A trait's trading days split into segments at every value change (an `update` drift
event) and around every dormant window, so a drifted trait's signals can be spread
across its distinct value-holding periods instead of clustering before or after the
change.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import PlanConfig
from pm_traitbench.enums import DriftEventType, Kind, Ownership, SignalMode, StanceEntry, Valence
from pm_traitbench.signals.carriers import Carrier
from pm_traitbench.signals.inputs import PlanInputs
from pm_traitbench.tables.schema import Trait

_NOTE_ENTRY: dict[DriftEventType, StanceEntry] = {
    DriftEventType.UPDATE: StanceEntry.DRIFT_UPDATE,
    DriftEventType.DORMANT: StanceEntry.DRIFT_DORMANT,
    DriftEventType.REVIVE: StanceEntry.DRIFT_REVIVE,
}


@dataclass(frozen=True)
class DateWindow:
    """An inclusive date range a signal may be placed within."""

    first: date
    last: date


@dataclass(frozen=True)
class PlannedSignal:
    """One signal a trait needs, before task 7 gives it a date and a session."""

    trait_id: str
    mode: SignalMode
    valence: Valence
    ownership: Ownership
    entry: StanceEntry
    window: DateWindow
    needs_carrier: bool
    third_party_value: str | None = None
    drift_date: date | None = None


def round_half_up(x: float) -> int:
    """Round to the nearest integer, ties rounding up (Python's `round` rounds ties to even)."""
    return math.floor(x + 0.5)


def largest_remainder[K](total: int, weights: Sequence[tuple[K, float]]) -> dict[K, int]:
    """Split `total` across `weights` in proportion to weight, largest fractional part first.

    Ties go to the earlier item in `weights` order. Zero-weight items always get 0, even
    when every fractional part left to distribute is itself 0.
    """
    keys = [key for key, _ in weights]
    values = [max(0.0, float(weight)) for _, weight in weights]
    total_weight = sum(values)
    if total <= 0 or total_weight <= 0:
        return dict.fromkeys(keys, 0)

    raw = [total * value / total_weight for value in values]
    floors = [math.floor(x) for x in raw]
    counts = list(floors)
    remainder = total - sum(floors)

    positive = [i for i, value in enumerate(values) if value > 0]
    order = sorted(positive, key=lambda i: (-(raw[i] - floors[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    return dict(zip(keys, counts, strict=True))


def _trait_segments(inputs: PlanInputs, trait_id: str) -> list[tuple[date, ...]]:
    """Maximal runs of trading days split at every `update` date, dormant days removed."""
    update_dates = {
        event.date
        for event in inputs.drift_events
        if event.trait_id == trait_id and event.event == DriftEventType.UPDATE
    }
    segments: list[list[date]] = []
    current: list[date] = []
    for day in inputs.trading_days:
        if inputs.is_dormant(trait_id, day):
            if current:
                segments.append(current)
                current = []
            continue
        if day in update_dates and current:
            segments.append(current)
            current = []
        current.append(day)
    if current:
        segments.append(current)
    return [tuple(segment) for segment in segments]


def _segment_for(segments: Sequence[tuple[date, ...]], event_date: date) -> int | None:
    """Index of the first segment starting on or after `event_date`, or `None` if none does."""
    for i, segment in enumerate(segments):
        if segment[0] >= event_date:
            return i
    return None


def _notes_per_segment(
    inputs: PlanInputs, trait_id: str, segments: Sequence[tuple[date, ...]]
) -> list[int]:
    counts = [0] * len(segments)
    for event in inputs.drift_events:
        if event.trait_id != trait_id:
            continue
        if event.event not in (DriftEventType.UPDATE, DriftEventType.REVIVE):
            continue
        index = _segment_for(segments, event.date)
        if index is not None:
            counts[index] += 1
    return counts


def _segment_needs(notes_per_segment: Sequence[int], drift_min_per_side: int) -> list[int]:
    """Each segment's deficit: `drift_min_per_side` minus the notes already counted in it."""
    return [max(0, drift_min_per_side - count) for count in notes_per_segment]


def _drift_notes(inputs: PlanInputs) -> list[PlannedSignal]:
    """One stated, self-owned announcement per drift event, running from its date to the
    PM's last trading day.
    """
    last_day = inputs.trading_days[-1]
    notes = []
    for event in inputs.drift_events:
        notes.append(
            PlannedSignal(
                trait_id=event.trait_id,
                mode=SignalMode.STATED,
                valence=Valence.CONFIRM,
                ownership=Ownership.SELF,
                entry=_NOTE_ENTRY[event.event],
                window=DateWindow(event.date, last_day),
                needs_carrier=False,
                drift_date=event.date,
            )
        )
    return notes


def _confirm_count(
    trait: Trait,
    segments: Sequence[tuple[date, ...]],
    needs: Sequence[int],
    knobs: PlanConfig,
    rng: np.random.Generator,
) -> int:
    """The trait's confirm-signal count: sampled for a bias, fixed for a preference, raised
    to cover every segment's drift deficit when the trait has more than one segment.
    """
    if trait.kind == Kind.BIAS:
        n = int(rng.integers(knobs.bias_signals_min, knobs.bias_signals_max + 1))
    else:
        n = knobs.pref_signals
    if len(segments) > 1:
        n = max(n, sum(needs))
    return n


def _allocate_segments(
    n: int, segments: Sequence[tuple[date, ...]], needs: Sequence[int], rng: np.random.Generator
) -> list[int]:
    """Each segment's need, plus a share of what is left over in proportion to its length."""
    remaining = n - sum(needs)
    lengths = np.array([len(segment) for segment in segments], dtype=float)
    extra = rng.multinomial(remaining, lengths / lengths.sum())
    return [need + int(more) for need, more in zip(needs, extra, strict=True)]


def _mode_sequence(
    n: int, weights: Sequence[tuple[SignalMode, float]], rng: np.random.Generator
) -> list[SignalMode]:
    """The trait's `n` modes, largest-remainder split by `weights` then shuffled so each
    segment gets a mix rather than every signal of one mode landing together.
    """
    counts = largest_remainder(n, weights)
    modes: list[SignalMode] = []
    for mode, _ in weights:
        modes.extend([mode] * counts[mode])
    order = rng.permutation(n)
    return [modes[i] for i in order]


def _bias_entry(mode: SignalMode) -> tuple[StanceEntry, bool]:
    if mode in (SignalMode.REVEALED, SignalMode.CONTRADICTION):
        return StanceEntry.REVEALED, True
    return StanceEntry.STATED, False


def _pref_entry(mode: SignalMode, has_pool: bool) -> tuple[StanceEntry, bool]:
    if mode == SignalMode.STATED:
        return StanceEntry.STATED, False
    if has_pool:
        return StanceEntry.REVEALED, True
    return StanceEntry.REVEALED_REACTION, False


def _trait_confirm_signals(
    inputs: PlanInputs,
    trait: Trait,
    pools: Mapping[str, Sequence[Carrier]],
    knobs: PlanConfig,
    rng: np.random.Generator,
) -> list[PlannedSignal]:
    """One trait's confirm signals: sized, split across its segments, and given a mode,
    entry and carrier need.
    """
    segments = _trait_segments(inputs, trait.trait_id)
    if len(segments) > 1:
        notes_per_segment = _notes_per_segment(inputs, trait.trait_id, segments)
        needs = _segment_needs(notes_per_segment, knobs.drift_min_per_side)
    else:
        needs = [0]
    n = _confirm_count(trait, segments, needs, knobs, rng)
    counts = _allocate_segments(n, segments, needs, rng)

    is_bias = trait.kind == Kind.BIAS
    weights: list[tuple[SignalMode, float]] = (
        [
            (SignalMode.REVEALED, knobs.bias_revealed_weight),
            (SignalMode.STATED, knobs.bias_stated_weight),
            (SignalMode.CONTRADICTION, knobs.bias_contradiction_weight),
        ]
        if is_bias
        else [
            (SignalMode.STATED, knobs.pref_stated_weight),
            (SignalMode.REVEALED, knobs.pref_revealed_weight),
        ]
    )
    modes = _mode_sequence(n, weights, rng)
    has_pool = bool(pools.get(trait.trait_id))

    signals = []
    mode_iter = iter(modes)
    for segment, count in zip(segments, counts, strict=True):
        window = DateWindow(segment[0], segment[-1])
        for _ in range(count):
            mode = next(mode_iter)
            entry, needs_carrier = _bias_entry(mode) if is_bias else _pref_entry(mode, has_pool)
            signals.append(
                PlannedSignal(
                    trait_id=trait.trait_id,
                    mode=mode,
                    valence=Valence.CONFIRM,
                    ownership=Ownership.SELF,
                    entry=entry,
                    window=window,
                    needs_carrier=needs_carrier,
                )
            )
    return signals


def _cycled_targets(rng: np.random.Generator, targets: Sequence[str], count: int) -> list[str]:
    """`count` targets drawn by cycling through `targets` in a random order."""
    if not targets or count <= 0:
        return []
    order = rng.permutation(len(targets))
    return [targets[order[i % len(targets)]] for i in range(count)]


def _random_ownership(rng: np.random.Generator) -> Ownership:
    return Ownership.COLLEAGUE if rng.integers(0, 2) == 0 else Ownership.CLIENT


def _retracted_signals(
    inputs: PlanInputs, total_confirm: int, knobs: PlanConfig, rng: np.random.Generator
) -> list[PlannedSignal]:
    """Extra stated-then-corrected rows, additional to the confirm count, spread over the
    PM's active biases and preferences.
    """
    r = round_half_up(knobs.retracted_share * total_confirm)
    targets = [
        trait.trait_id
        for trait in inputs.traits
        if (trait.kind == Kind.BIAS and trait.active) or trait.kind == Kind.PREFERENCE
    ]
    window = DateWindow(inputs.trading_days[0], inputs.trading_days[-1])
    assigned = _cycled_targets(rng, targets, r)
    return [
        PlannedSignal(
            trait_id=trait_id,
            mode=SignalMode.STATED,
            valence=Valence.RETRACTED,
            ownership=Ownership.SELF,
            entry=StanceEntry.RETRACT,
            window=window,
            needs_carrier=False,
        )
        for trait_id in assigned
    ]


def _preference_third_party_targets(
    inputs: PlanInputs, catalogue: Catalogue
) -> dict[str, tuple[str, ...]]:
    """Preference trait ids mapped to the catalogue values the PM never held: not its current
    value and not a `from` or `to` value of any of its drift events.
    """
    entries = {entry.param: entry for entry in catalogue.preferences}
    targets: dict[str, tuple[str, ...]] = {}
    for trait in inputs.traits:
        if trait.kind != Kind.PREFERENCE:
            continue
        entry = entries.get(trait.param)
        if entry is None:
            continue
        held = {trait.value}
        for event in inputs.drift_events:
            if event.trait_id != trait.trait_id:
                continue
            if event.from_value is not None:
                held.add(event.from_value)
            if event.to_value is not None:
                held.add(event.to_value)
        never_held = tuple(sorted(value for value in entry.values if value not in held))
        if never_held:
            targets[trait.trait_id] = never_held
    return targets


def _third_party_signals(
    inputs: PlanInputs,
    catalogue: Catalogue,
    total_confirm: int,
    retracted_count: int,
    knobs: PlanConfig,
    rng: np.random.Generator,
) -> list[PlannedSignal]:
    """Extra rows attributed to a colleague or client instead of the PM: half (rounded up)
    to the PM's inactive biases and the rest to preferences the PM has never held every
    value of, unless one side has no target, in which case the other side takes them all.
    """
    k = round_half_up(knobs.third_party_share * (total_confirm + retracted_count))
    inactive_biases = [
        trait.trait_id for trait in inputs.traits if trait.kind == Kind.BIAS and not trait.active
    ]
    pref_targets = _preference_third_party_targets(inputs, catalogue)

    if inactive_biases:
        k_bias = math.ceil(k / 2) if pref_targets else k
    else:
        k_bias = 0
    k_pref = k - k_bias if pref_targets else 0

    window = DateWindow(inputs.trading_days[0], inputs.trading_days[-1])
    signals = []
    for trait_id in _cycled_targets(rng, inactive_biases, k_bias):
        signals.append(
            PlannedSignal(
                trait_id=trait_id,
                mode=SignalMode.STATED,
                valence=Valence.CONFIRM,
                ownership=_random_ownership(rng),
                entry=StanceEntry.THIRD_PARTY,
                window=window,
                needs_carrier=False,
            )
        )
    pref_ids = list(pref_targets)
    for trait_id in _cycled_targets(rng, pref_ids, k_pref):
        value = str(rng.choice(pref_targets[trait_id]))
        signals.append(
            PlannedSignal(
                trait_id=trait_id,
                mode=SignalMode.STATED,
                valence=Valence.CONFIRM,
                ownership=_random_ownership(rng),
                entry=StanceEntry.THIRD_PARTY,
                window=window,
                needs_carrier=False,
                third_party_value=value,
            )
        )
    return signals


def plan_quotas(
    inputs: PlanInputs,
    pools: Mapping[str, Sequence[Carrier]],
    catalogue: Catalogue,
    knobs: PlanConfig,
    rng: np.random.Generator,
) -> list[PlannedSignal]:
    """The signals one PM's plan needs: how many of each trait, in which mode, and where."""
    confirm_signals: list[PlannedSignal] = []
    for trait in inputs.traits:
        if trait.kind == Kind.BIAS and not trait.active:
            continue
        confirm_signals.extend(_trait_confirm_signals(inputs, trait, pools, knobs, rng))

    notes = _drift_notes(inputs)
    retracted = _retracted_signals(inputs, len(confirm_signals), knobs, rng)
    third_party = _third_party_signals(
        inputs, catalogue, len(confirm_signals), len(retracted), knobs, rng
    )
    return [*confirm_signals, *notes, *retracted, *third_party]
