"""Disposition effect: realising gains readily and holding losers, after Odean (1998)."""

from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import PnlState, PositionAction
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs
from pm_traitbench.tables.schema import PositionDay

_REALISED = frozenset({PositionAction.CUT, PositionAction.TRIM, PositionAction.EXIT})


def sell_day_rows(inputs: PmInputs, days: frozenset[date]) -> list[PositionDay]:
    """Position-days in `days` dated on one of the PM's sell dates, gain, loss and flat alike."""
    return [
        row for row in inputs.position_days if row.date in days and row.date in inputs.sell_dates
    ]


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Odean's (1998) proportion-of-gains-realised over proportion-of-losses-realised ratio.

    PGR and PLR are each realised share of their pnl-state's sell-day rows; `n`
    counts every sell-day row including flat ones, which sit in neither ratio.
    `None` when there are no gain rows, no loss rows, or PLR is zero.
    """
    rows = sell_day_rows(inputs, days)
    n = len(rows)
    gains = [row for row in rows if row.pnl_state == PnlState.GAIN]
    losses = [row for row in rows if row.pnl_state == PnlState.LOSS]
    if not gains or not losses:
        return Estimate(value=None, n=n)
    pgr = sum(1 for row in gains if row.action in _REALISED) / len(gains)
    plr = sum(1 for row in losses if row.action in _REALISED) / len(losses)
    if plr == 0:
        return Estimate(value=None, n=n)
    return Estimate(value=pgr / plr, n=n)


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
