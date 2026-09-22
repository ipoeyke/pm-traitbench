"""Per-seed market generation: simulate one market seed's full state and
convert it into the row-model tables the market stage writes out.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import EventType
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import (
    RngFor,
    build_jumps,
    event_day_indices,
    generated_rows,
    sample_events,
)
from pm_traitbench.market.consensus import ConsensusResult, build_consensus
from pm_traitbench.market.drivers import DriverShocks, seed_driver
from pm_traitbench.market.processes import commodities, credit, equities, fx, rates
from pm_traitbench.market.processes.common import ProcessInputs, ProcessOutput
from pm_traitbench.market.regimes import RegimeLookup, RegimePath, build_schedule, regime_path
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import (
    CalendarEvent,
    ConsensusRow,
    CurvePoint,
    Instrument,
    Price,
    RegimeSpan,
)


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


def _calendar_sort_key(row: CalendarEvent) -> tuple:
    return (row.date, row.instrument_id or "", row.event)


def generate_seed(
    config: Config,
    seed: str,
    instruments: Sequence[Instrument],
    shocks: DriverShocks,
    axis: SimAxis,
) -> SeedMarket:
    """Simulate one market seed: regime path, driver, every family's process, calendar
    and consensus.
    """
    instruments = tuple(instruments)
    rng_for = market_rng(config.seed.root)

    schedule = build_schedule(config, seed, config.timeline())
    lookup = RegimeLookup(schedule, schedule[0].regime)
    path = regime_path(axis, lookup, config)

    sampled = sample_events(instruments, axis, config, rng_for, seed)
    generated = generated_rows(instruments, axis, seed)
    jumps = build_jumps(sampled.rows, axis, config)
    z = seed_driver(shocks, path, jumps.macro)

    inputs = ProcessInputs(
        instruments=instruments,
        axis=axis,
        path=path,
        z=z,
        group_shocks=shocks.groups,
        jumps=jumps,
        market=config.market,
        rng_for=rng_for,
    )

    rates_output = rates.simulate(inputs)
    output = (
        equities.simulate(inputs)
        .merge(rates_output)
        .merge(credit.simulate(inputs, rates_output))
        .merge(commodities.simulate(inputs))
        .merge(fx.simulate(inputs))
    )

    event_days = event_day_indices(sampled.rows, axis)
    consensus = build_consensus(instruments, axis, output, event_days, config.market, rng_for, seed)

    calendar = sorted([*sampled.rows, *generated, *consensus.flips], key=_calendar_sort_key)

    return SeedMarket(
        seed=seed,
        axis=axis,
        schedule=schedule,
        path=path,
        z=z,
        output=output,
        calendar=calendar,
        drawn_events=sampled.drawn,
        consensus=consensus,
    )


def to_rows(market: SeedMarket) -> MarketRows:
    """Convert one seed's simulated state into row-model tables, horizon days only."""
    axis = market.axis
    seed = market.seed
    horizon_offsets = [(t, day) for t, day in enumerate(axis.dates) if t >= axis.n_burn]

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
