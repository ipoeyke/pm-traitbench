"""Per-seed market generation: simulate one market seed's full state from the
regime schedule, driver shocks and every family's price process.
"""

from collections.abc import Sequence

from pm_traitbench.config import Config
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import event_day_indices, generated_rows, row_sort_key
from pm_traitbench.market.consensus import build_consensus
from pm_traitbench.market.regimes import RegimeLookup, regime_path
from pm_traitbench.market.seed import SeedMarket, market_rng
from pm_traitbench.market.synthetic.drivers import DriverShocks, seed_driver
from pm_traitbench.market.synthetic.events import build_jumps, sample_events
from pm_traitbench.market.synthetic.processes import commodities, credit, equities, fx, rates
from pm_traitbench.market.synthetic.processes.common import ProcessInputs
from pm_traitbench.market.synthetic.schedule import build_schedule
from pm_traitbench.tables.schema import Instrument


def build_seed(
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

    calendar = sorted([*sampled.rows, *generated, *consensus.flips], key=row_sort_key)

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
