"""Shared probes test fixtures: the gate 2 row builders plus timeline, drift, regime and
third-party signal builders.
"""

from collections.abc import Sequence
from datetime import date, timedelta

from pm_traitbench.enums import (
    DriftEventType,
    Ownership,
    Regime,
    SignalMode,
    Valence,
)
from pm_traitbench.tables.schema import DriftEvent, RegimeSpan, Signal
from pm_traitbench.timeline import Timeline
from tests.gates.gate2.fixtures import PM_A, session_of, signal, trait

__all__ = [
    "PM_A",
    "TIMELINE",
    "drift",
    "session_of",
    "signal",
    "spans",
    "third_party",
    "trait",
    "weekdays",
]

TIMELINE = Timeline(date(2026, 1, 5), 52)

_REGIME_CYCLE = (Regime.RANGE, Regime.RISK_OFF, Regime.RISK_ON)


def weekdays(timeline: Timeline) -> tuple[date, ...]:
    """Every Monday-Friday date of the timeline."""
    return tuple(timeline.weekdays_in_weeks(1, timeline.n_weeks))


def drift(
    pm_id: str,
    day: date,
    event: DriftEventType,
    trait_id: str,
    from_value: float | str | None = None,
    to_value: float | str | None = None,
) -> DriftEvent:
    """A `DriftEvent` on `trait_id` dated `day`."""
    return DriftEvent(
        pm_id=pm_id,
        date=day,
        event=event,
        trait_id=trait_id,
        from_value=from_value,
        to_value=to_value,
    )


def third_party(
    pm_id: str, session_id: str, day: date, trait_id: str, value: str, signal_id: str
) -> Signal:
    """A colleague's stated confirmation of `trait_id` carrying `value`."""
    return Signal(
        signal_id=signal_id,
        pm_id=pm_id,
        session_id=session_id,
        date=day,
        trait_id=trait_id,
        mode=SignalMode.STATED,
        trade_idea_id=None,
        valence=Valence.CONFIRM,
        ownership=Ownership.COLLEAGUE,
        third_party_value=value,
        claim_session_id=None,
    )


def spans(seed: str, starts: Sequence[date], end: date) -> list[RegimeSpan]:
    """Consecutive regime spans opening on `starts` and the last closing on `end`, with
    regimes cycling range, risk-off, risk-on.
    """
    closes = [start - timedelta(days=1) for start in starts[1:]] + [end]
    return [
        RegimeSpan(
            seed=seed,
            regime=_REGIME_CYCLE[i % len(_REGIME_CYCLE)],
            date_start=start,
            date_end=close,
        )
        for i, (start, close) in enumerate(zip(starts, closes, strict=True))
    ]
