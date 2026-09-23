"""One market seed's full simulated state, and its conversion into row-model tables."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from pm_traitbench.enums import EventType
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import RngFor
from pm_traitbench.market.consensus import ConsensusResult
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.regimes import RegimePath
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import CalendarEvent, ConsensusRow, CurvePoint, Price, RegimeSpan


@dataclass(frozen=True)
class SeedMarket:
    """One market seed's full simulated state, before conversion to rows."""

    seed: str
    axis: SimAxis
    schedule: list[RegimeSpan]
    path: RegimePath
    z: np.ndarray
    output: ProcessOutput
    calendar: list[CalendarEvent]
    drawn_events: dict[EventType, int]
    consensus: ConsensusResult
    fills: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketRows:
    """One market seed's simulated state, converted into row-model tables."""

    prices: list[Price]
    curves: list[CurvePoint]
    consensus: list[ConsensusRow]
    calendar: list[CalendarEvent]
    regimes: list[RegimeSpan]


def market_rng(root_seed: int) -> RngFor:
    """Return the market stream factory shared by every draw in the market stage."""
    return lambda *keys: stream(root_seed, "market", *keys)


def to_rows(market: SeedMarket) -> MarketRows:
    """Convert one seed's simulated state into row-model tables, horizon days only."""
    axis = market.axis
    seed = market.seed
    horizon_offsets = [(t, axis.dates[t]) for t in range(axis.n_burn, axis.n_days)]

    prices = []
    for instrument_id, series in market.output.prices.items():
        spreads = market.output.spreads.get(instrument_id)
        for t, day in horizon_offsets:
            prices.append(
                Price(
                    seed=seed,
                    date=day,
                    instrument_id=instrument_id,
                    price=float(series[t]),
                    spread_bp=float(spreads[t]) if spreads is not None else None,
                )
            )

    curves = []
    for (curve_id, tenor), series in market.output.curves.items():
        for t, day in horizon_offsets:
            curves.append(
                CurvePoint(
                    seed=seed, date=day, curve_id=curve_id, tenor=tenor, level=float(series[t])
                )
            )

    return MarketRows(
        prices=prices,
        curves=curves,
        consensus=market.consensus.rows,
        calendar=market.calendar,
        regimes=market.schedule,
    )
