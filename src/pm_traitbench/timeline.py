"""Week-numbered timeline over a Monday-to-Monday simulation horizon."""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Timeline:
    start: date
    n_weeks: int

    def __post_init__(self) -> None:
        if self.start.weekday() != 0:
            raise ValueError("start must be a Monday")
        if self.n_weeks < 1:
            raise ValueError("n_weeks must be positive")

    def week_start(self, week: int) -> date:
        """Return the Monday of the given 1-indexed week."""
        if not 1 <= week <= self.n_weeks:
            raise ValueError(f"week must be between 1 and {self.n_weeks}")
        return self.start + timedelta(weeks=week - 1)

    def weekdays_in_weeks(self, first: int, last: int) -> list[date]:
        """Return every Mon-Fri date across the 1-indexed week range [first, last]."""
        if first > last:
            raise ValueError("first must not be greater than last")
        days = []
        for week in range(first, last + 1):
            monday = self.week_start(week)
            days.extend(monday + timedelta(days=offset) for offset in range(5))
        return days
