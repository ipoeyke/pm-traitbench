"""Anchoring: the share of anchor crossings the PM exits on.

An idea's anchor is a round level fixed at entry (`anchor_level` on its
position-day rows), a salient prior level after Northcraft and Neale (1987).
Among ideas whose tracked level reaches that anchor, the share exited on the
crossing day estimates rho plus the background sell hazard that would exit
some ideas there anyway.
"""

from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.engine.series import Series
from pm_traitbench.enums import PositionAction
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs
from pm_traitbench.gates.gate1.side_sign import SIDE_SIGN
from pm_traitbench.tables.schema import Idea, PositionDay


def _crossing_row(idea: Idea, rows: list[PositionDay], series: Series) -> PositionDay | None:
    """First row after entry whose tracked level reaches the anchor toward the target."""
    if not rows or rows[0].anchor_level is None:
        return None
    anchor = rows[0].anchor_level
    fav = 1.0 if idea.target_level >= idea.entry_level else -1.0
    bullish = series.bullish_sign * SIDE_SIGN[idea.side]
    for row in rows:
        if row.date <= idea.entry_date:
            continue
        level = idea.entry_level + bullish * row.pnl_unit
        if (level - anchor) * fav >= 0:
            return row
    return None


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Hits (exit on the crossing day) over opportunities (a crossing before the horizon ends).

    A crossing on `inputs.last_date` is not an opportunity - every open position closes
    out that day regardless of the PM's choice. A crossing day whose exit followed an
    acted rule event is not a hit, since the PM did not choose it.
    """
    rows_by_idea: dict[str, list[PositionDay]] = {}
    for row in inputs.position_days:
        rows_by_idea.setdefault(row.trade_idea_id, []).append(row)

    opportunities = 0
    hits = 0
    for idea in inputs.ideas:
        if idea.entry_date not in days:
            continue
        rows = sorted(rows_by_idea.get(idea.trade_idea_id, ()), key=lambda row: row.date)
        crossing = _crossing_row(idea, rows, inputs.series[idea.trade_idea_id])
        if crossing is None or crossing.date == inputs.last_date:
            continue
        opportunities += 1
        acted_that_day = (idea.trade_idea_id, crossing.date) in inputs.acted
        if crossing.action == PositionAction.EXIT and not acted_that_day:
            hits += 1

    if opportunities == 0:
        return Estimate(value=None, n=0)
    return Estimate(value=hits / opportunities, n=opportunities)


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
