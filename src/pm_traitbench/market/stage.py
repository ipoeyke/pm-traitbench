"""Market stage: simulate the instrument universe and every configured market seed's
full state, check each against what the config implies, then write the six market tables.
"""

from typing import Any

from pm_traitbench.config import Config
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.seed import to_rows
from pm_traitbench.market.synthetic.build import build_seed
from pm_traitbench.market.synthetic.check import check_market
from pm_traitbench.market.synthetic.drivers import draw_shocks
from pm_traitbench.market.synthetic.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import CalendarEvent, ConsensusRow, CurvePoint, Price, RegimeSpan
from pm_traitbench.tables.specs import (
    MARKET_CALENDAR,
    MARKET_CONSENSUS,
    MARKET_CURVES,
    MARKET_INSTRUMENTS,
    MARKET_PRICES,
    MARKET_REGIMES,
    MARKET_TABLES,
)
from pm_traitbench.tables.store import DataStore


def referenced_seeds(config: Config) -> list[str]:
    """Every market seed the population grid uses, pilot seeds first, without repeats."""
    seen: dict[str, None] = {}
    for seed in (*config.population.pilot_market_seeds, *config.population.market_seeds):
        seen.setdefault(seed, None)
    return list(seen)


def run(config: Config, store: DataStore) -> dict[str, Any]:
    """Simulate every referenced market seed and write the six market tables.

    The universe, axis and driver shocks are shared across every synthetic
    seed and built only if one is referenced. Each seed is generated and
    checked in turn; a failing seed raises MarketCheckError before any table
    is written. A real seed raises StageIOError, since fetching real market
    data is not wired in yet.
    """
    seeds = referenced_seeds(config)
    real_seeds = config.market.real.seeds

    axis = instruments = shocks = None
    if any(seed not in real_seeds for seed in seeds):
        axis = build_axis(config.timeline(), config.market.burn_in_days)
        instruments = build_universe(config, stream(config.seed.root, "market", "universe"))
        shocks = draw_shocks(config.seed.root, axis.n_days)

    prices: list[Price] = []
    curves: list[CurvePoint] = []
    consensus: list[ConsensusRow] = []
    calendar: list[CalendarEvent] = []
    regimes: list[RegimeSpan] = []
    reports: dict[str, Any] = {}

    for seed in seeds:
        if seed in real_seeds:
            raise StageIOError(
                f"real market seed '{seed}' needs a raw cache; run fetch-market first"
            )

        market = build_seed(config, seed, instruments, shocks, axis)
        report = check_market(market, instruments, config)
        reports[seed] = report.to_dict()

        rows = to_rows(market)
        prices.extend(rows.prices)
        curves.extend(rows.curves)
        consensus.extend(rows.consensus)
        calendar.extend(rows.calendar)
        regimes.extend(rows.regimes)

    store.write(MARKET_INSTRUMENTS, instruments)
    store.write(MARKET_PRICES, prices)
    store.write(MARKET_CURVES, curves)
    store.write(MARKET_CONSENSUS, consensus)
    store.write(MARKET_CALENDAR, calendar)
    store.write(MARKET_REGIMES, regimes)

    return {"check": reports}


MARKET_STAGE = Stage(
    number=2,
    name="market",
    help="simulate prices, curves, consensus, calendar and regimes per market seed",
    run=run,
    reads=(),
    writes=MARKET_TABLES,
)
