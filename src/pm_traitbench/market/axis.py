"""Simulation axis: burn-in warm-up days followed by the published horizon.

Burn-in lets processes reach a stationary state before the first published
day, without ever emitting a price for it.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

from pm_traitbench.timeline import Timeline


@dataclass(frozen=True)
class SimAxis:
    """Ascending weekday dates: burn-in first, then the published horizon."""

    dates: tuple[date, ...]
    n_burn: int
    _index: dict[date, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {day: i for i, day in enumerate(self.dates)})

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def horizon(self) -> slice:
        """Slice into `dates` selecting the published horizon, excluding burn-in."""
        return slice(self.n_burn, self.n_days)

    def index(self, day: date) -> int:
        """Return the axis position of `day`; raise ValueError if it is off the axis."""
        try:
            return self._index[day]
        except KeyError:
            raise ValueError(f"date {day} is not on the simulation axis") from None


def build_axis(timeline: Timeline, burn_in_days: int) -> SimAxis:
    """Build the axis: `burn_in_days` weekdays before `timeline.start`, then the horizon."""
    burn_in: list[date] = []
    day = timeline.start - timedelta(days=1)
    while len(burn_in) < burn_in_days:
        if day.weekday() < 5:
            burn_in.append(day)
        day -= timedelta(days=1)
    burn_in.reverse()

    horizon = timeline.weekdays_in_weeks(1, timeline.n_weeks)
    return SimAxis(dates=tuple(burn_in) + tuple(horizon), n_burn=len(burn_in))
