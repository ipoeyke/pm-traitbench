"""Loss aversion: the share of loss-side opportunities where a PM adds instead of cutting."""

from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import PnlState, PositionAction
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs
from pm_traitbench.tables.schema import PositionDay


def opportunities(inputs: PmInputs, days: frozenset[date]) -> list[PositionDay]:
    """Position-days in `days` where averaging down was on the table.

    A loss, not mid-trigger-response, not a roll, and not the last horizon date
    (which has no next-day mark to act on).
    """
    return [
        row
        for row in inputs.position_days
        if row.date in days
        and row.pnl_state == PnlState.LOSS
        and not row.trigger_pending
        and row.action != PositionAction.ROLL
        and row.date != inputs.last_date
    ]


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Share of loss-side opportunities in `days` where the PM adds to the position."""
    rows = opportunities(inputs, days)
    n = len(rows)
    if n == 0:
        return Estimate(value=None, n=0)
    adds = sum(1 for row in rows if row.action == PositionAction.ADD)
    return Estimate(value=adds / n, n=n)


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
