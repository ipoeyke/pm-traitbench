"""Regime schedule: which of the three regimes applies on each simulated week.

Each market seed cycles range, risk-off and risk-on across three spans at
fixed week boundaries.
"""

from datetime import timedelta

from pm_traitbench.config import Config
from pm_traitbench.tables.schema import RegimeSpan
from pm_traitbench.timeline import Timeline


def build_schedule(config: Config, seed: str, timeline: Timeline) -> list[RegimeSpan]:
    """Build the three regime spans for one market seed's week boundaries."""
    order = config.market.seeds[seed]
    b1, b2 = config.market.boundary_weeks
    week_ranges = ((1, b1), (b1 + 1, b2), (b2 + 1, timeline.n_weeks))
    spans = []
    for regime, (first, last) in zip(order, week_ranges, strict=True):
        date_start = timeline.week_start(first)
        date_end = timeline.week_start(last) + timedelta(days=4)
        spans.append(RegimeSpan(seed=seed, regime=regime, date_start=date_start, date_end=date_end))
    return spans
