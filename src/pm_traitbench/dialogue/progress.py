"""Progress lines for the API-billed stages, printed to stderr as each unit finishes."""

import sys
import time
from collections.abc import Callable

from pm_traitbench.dialogue.client import UsageTotals
from pm_traitbench.errors import PmTraitbenchError

_SECONDS_PER_HOUR = 3600
_MINUTES_PER_HOUR = 60


def format_count(n: int) -> str:
    """`n` as is below 1,000, else whole thousands (`182k`), else millions (`5.0M`).

    Thousands switch to millions once they would round to `1000k`.
    """
    if n < 1_000:
        return str(n)
    thousands = (n + 500) // 1_000
    if thousands < 1_000:
        return f"{thousands}k"
    return f"{n / 1_000_000:.1f}M"


def format_elapsed(seconds: float) -> str:
    """`MmSSs` under an hour, `HhMMm` from an hour on."""
    whole = int(seconds)
    if whole < _SECONDS_PER_HOUR:
        return f"{whole // 60}m{whole % 60:02d}s"
    minutes = whole // 60
    return f"{minutes // _MINUTES_PER_HOUR}h{minutes % _MINUTES_PER_HOUR:02d}m"


def format_line(
    label: str,
    unit: str,
    done: int,
    total: int,
    failed: int,
    totals: UsageTotals,
    budget: int | None,
    elapsed: float,
) -> str:
    """One progress line; tokens are fresh only, and the budget is shown when there is one."""
    spent = totals.input_tokens + totals.output_tokens
    tokens = (
        f"{format_count(totals.input_tokens)} in, {format_count(totals.output_tokens)} out tokens"
    )
    if budget is not None:
        tokens += f" ({format_count(spent)}/{format_count(budget)} budget)"
    return (
        f"[{label}] {done}/{total} {unit} ({failed} failed) "
        f"| {totals.calls} calls, {totals.cache_hits} cached "
        f"| {tokens} | {format_elapsed(elapsed)}"
    )


def format_failure(label: str, unit_name: str, error: BaseException) -> str:
    """One failure line naming the unit and its reason, stripped of a repeated unit prefix.

    An error outside the project's own hierarchy is led by its type, since its message
    alone (a bare `KeyError`'s key, say) rarely says what went wrong.
    """
    message = str(error)
    prefix = f"{unit_name}: "
    if message.startswith(prefix):
        message = message[len(prefix) :]
    if not isinstance(error, PmTraitbenchError):
        message = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return f"[{label}] {unit_name} failed: {message}"


class Progress:
    """Prints a start line, then one line per finished unit, flushed to stderr."""

    def __init__(
        self,
        label: str,
        unit: str,
        total: int,
        totals: UsageTotals,
        budget: int | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._label = label
        self._unit = unit
        self._total = total
        self._totals = totals
        self._budget = budget
        self._clock = clock
        self._started = 0.0
        self._done = 0
        self._failed = 0

    def start(self, max_concurrency: int) -> None:
        """Record the start time and print the start line."""
        self._started = self._clock()
        self._print(f"[{self._label}] {self._total} {self._unit}, concurrency {max_concurrency}")

    def finish(self, unit_name: str, error: BaseException | None = None) -> None:
        """Count one finished unit and print its line, led by a failure line if it raised."""
        self._done += 1
        if error is not None:
            self._failed += 1
            self._print(format_failure(self._label, unit_name, error))
        self._print(
            format_line(
                self._label,
                self._unit,
                self._done,
                self._total,
                self._failed,
                self._totals,
                self._budget,
                self._clock() - self._started,
            )
        )

    @staticmethod
    def _print(line: str) -> None:
        # `sys.stderr` is looked up per call so a test's capture sees it.
        print(line, file=sys.stderr, flush=True)
