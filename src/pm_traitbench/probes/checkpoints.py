"""Checkpoint dates at which a PM's probes are asked."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from pm_traitbench.enums import CheckpointLabel
from pm_traitbench.tables.schema import DriftEvent, RegimeSpan
from pm_traitbench.timeline import Timeline


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint date, its winning label, and every label whose candidate landed on it."""

    label: CheckpointLabel
    day: date
    index: int
    covers: frozenset[CheckpointLabel]


# The drift labels are rarer and are what the drift analysis slices on, so they win a shared date.
LABEL_PRECEDENCE: tuple[CheckpointLabel, ...] = (
    CheckpointLabel.PRE_DRIFT,
    CheckpointLabel.POST_DRIFT,
    CheckpointLabel.REGIME_SHIFT,
    CheckpointLabel.WEEK4,
    CheckpointLabel.WEEK13,
    CheckpointLabel.WEEK52,
)


def week_of(day: date, timeline: Timeline) -> int:
    """The 1-indexed week of `timeline` containing `day`."""
    return (day - timeline.start).days // 7 + 1


def last_trading_day(week: int, timeline: Timeline, trading_days: Sequence[date]) -> date | None:
    """The latest trading day within `week`, or None when the week has none."""
    monday = timeline.week_start(week)
    in_week = [d for d in trading_days if monday <= d <= monday + timedelta(days=6)]
    return max(in_week) if in_week else None


def checkpoints_for(
    drift_events: Sequence[DriftEvent],
    regimes: Sequence[RegimeSpan],
    timeline: Timeline,
    trading_days: Sequence[date],
    post_drift_weeks: int,
) -> tuple[Checkpoint, ...]:
    """A PM's checkpoints in date order, one per date.

    `regimes` are the spans of the PM's own market seed. A date shared by several
    candidate labels keeps the one earliest in `LABEL_PRECEDENCE` and covers them all.
    """
    candidates: list[tuple[CheckpointLabel, int]] = [
        (CheckpointLabel.WEEK4, 4),
        (CheckpointLabel.WEEK13, 13),
        (CheckpointLabel.WEEK52, timeline.n_weeks),
    ]
    for event_day in sorted({e.date for e in drift_events}):
        week = week_of(event_day, timeline)
        candidates.append((CheckpointLabel.PRE_DRIFT, week - 1))
        candidates.append((CheckpointLabel.POST_DRIFT, week + post_drift_weeks))
    for span in sorted(regimes, key=lambda s: s.date_start)[1:]:
        candidates.append((CheckpointLabel.REGIME_SHIFT, week_of(span.date_start, timeline) + 1))

    by_day: dict[date, set[CheckpointLabel]] = {}
    for label, week in candidates:
        if not 1 <= week <= timeline.n_weeks:
            continue
        day = last_trading_day(week, timeline, trading_days)
        if day is not None:
            by_day.setdefault(day, set()).add(label)
    return tuple(
        Checkpoint(
            label=min(by_day[day], key=LABEL_PRECEDENCE.index),
            day=day,
            index=i,
            covers=frozenset(by_day[day]),
        )
        for i, day in enumerate(sorted(by_day))
    )
