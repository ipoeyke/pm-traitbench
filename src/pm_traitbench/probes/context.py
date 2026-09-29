"""The context a copilot has seen by a checkpoint, and the signals that support a trait in it."""

from collections.abc import Collection, Sequence
from datetime import date

from pm_traitbench.enums import Kind, Ownership, Valence
from pm_traitbench.gates.gate2.transcript import render_pm
from pm_traitbench.tables.schema import DriftEvent, Session, Signal, Trait
from pm_traitbench.traits_truth import latest_update


def sessions_until(sessions: Sequence[Session], day: date) -> tuple[Session, ...]:
    """Sessions dated on or before `day`, in date-then-id order."""
    return tuple(
        sorted((s for s in sessions if s.date <= day), key=lambda s: (s.date, s.session_id))
    )


def context_chars(sessions: Sequence[Session]) -> int:
    """Length of the rendered transcript of `sessions`."""
    return len(render_pm(sessions))


def in_context(
    signals: Sequence[Signal], session_ids: Collection[str], day: date
) -> tuple[Signal, ...]:
    """Signals on a kept session and dated on or before `day`.

    A dropped session voids its signals, since a signal is only visible through its session.
    """
    return tuple(s for s in signals if s.session_id in session_ids and s.date <= day)


def own_confirm_ids(
    trait_id: str, signals: Sequence[Signal], since: date | None = None
) -> tuple[str, ...]:
    """Sorted ids of the PM's own confirming signals of `trait_id`, on or after `since`."""
    return tuple(
        sorted(
            s.signal_id
            for s in signals
            if s.trait_id == trait_id
            and s.ownership == Ownership.SELF
            and s.valence == Valence.CONFIRM
            and (since is None or s.date >= since)
        )
    )


def supporting_ids(
    trait: Trait, signals: Sequence[Signal], drift_events: Sequence[DriftEvent], day: date
) -> tuple[str, ...]:
    """Sorted ids of signals in context that support the trait's value on `day`.

    For an updated preference, signals dated before the update express the old value.
    """
    since = None
    if trait.kind == Kind.PREFERENCE:
        update = latest_update(trait.trait_id, drift_events, day)
        since = update.date if update is not None else None
    return own_confirm_ids(trait.trait_id, signals, since)


def third_party_signals(trait_id: str, signals: Sequence[Signal]) -> tuple[Signal, ...]:
    """Signals of `trait_id` attributed to someone other than the PM, in id order."""
    return tuple(
        sorted(
            (s for s in signals if s.trait_id == trait_id and s.ownership != Ownership.SELF),
            key=lambda s: s.signal_id,
        )
    )
