"""Market stage: build the instrument universe and every configured market seed's
full state, real or simulated, check each against what the config implies, then
write the six market tables.
"""

from datetime import date
from typing import Any

from pm_traitbench.config import Config, referenced_seeds
from pm_traitbench.errors import MarketCheckError, StageIOError
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.real.build import build_seed as build_real_seed
from pm_traitbench.market.real.check import check_real_market
from pm_traitbench.market.real.fetch import RawCache, fetch_range
from pm_traitbench.market.real.universe import build_real_universe
from pm_traitbench.market.seed import to_rows
from pm_traitbench.market.synthetic.build import build_seed as build_synthetic_seed
from pm_traitbench.market.synthetic.check import check_market
from pm_traitbench.market.synthetic.drivers import draw_shocks
from pm_traitbench.market.synthetic.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import (
    CalendarEvent,
    ConsensusRow,
    CurvePoint,
    Instrument,
    Price,
    RegimeSpan,
)
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


def _merge_instruments(synthetic: list[Instrument], real: list[Instrument]) -> list[Instrument]:
    """Combine both universes into the one instruments table: an id shared between
    them (a commodity, an FX pair or the USD curve) is kept once when the two
    rows are equal, otherwise the mismatch fails the run.
    """
    merged = list(synthetic)
    by_id = {inst.instrument_id: inst for inst in merged}
    for inst in real:
        existing = by_id.get(inst.instrument_id)
        if existing is None:
            merged.append(inst)
            by_id[inst.instrument_id] = inst
        elif existing != inst:
            raise MarketCheckError(
                f"instrument '{inst.instrument_id}' differs between real and synthetic universes"
            )
    return merged


def run(config: Config, store: DataStore) -> dict[str, Any]:
    """Build or simulate every referenced market seed and write the six market tables.

    The synthetic universe, axis and driver shocks are shared across every
    synthetic seed and built only if one is referenced; the real universe is
    opened from the raw cache and built only if a real seed is referenced. A
    real seed is built and checked against its own real universe, a synthetic
    seed against its own synthetic universe; the written instruments table is
    the two universes merged. Every seed is built and checked before any
    table is written.
    """
    seeds = referenced_seeds(config)
    real_seeds = config.market.real.seeds

    axis = build_axis(config.timeline(), config.market.burn_in_days)

    synthetic_instruments: list[Instrument] = []
    shocks = None
    if any(seed not in real_seeds for seed in seeds):
        synthetic_instruments = build_universe(
            config, stream(config.seed.root, "market", "universe")
        )
        shocks = draw_shocks(config.seed.root, axis.n_days)

    real_instruments: list[Instrument] = []
    cache: RawCache | None = None
    if any(seed in real_seeds for seed in seeds):
        cache = RawCache.open(store.data_dir)
        cache_start = date.fromisoformat(cache.manifest.window[0])
        cache_end = date.fromisoformat(cache.manifest.window[1])
        needed_start, needed_end = fetch_range(config)
        if cache_start > needed_start or cache_end < needed_end:
            raise StageIOError(
                f"raw market cache covers {cache_start} to {cache_end}, the config needs "
                f"{needed_start} to {needed_end}; run fetch-market again"
            )
        real_instruments = build_real_universe(config, cache, axis)

    instruments = _merge_instruments(synthetic_instruments, real_instruments)

    prices: list[Price] = []
    curves: list[CurvePoint] = []
    consensus: list[ConsensusRow] = []
    calendar: list[CalendarEvent] = []
    regimes: list[RegimeSpan] = []
    reports: dict[str, Any] = {}

    for seed in seeds:
        if seed in real_seeds:
            market = build_real_seed(config, seed, real_instruments, cache, axis)
            report = check_real_market(market, real_instruments, config)
        else:
            market = build_synthetic_seed(config, seed, synthetic_instruments, shocks, axis)
            report = check_market(market, synthetic_instruments, config)
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

    extras: dict[str, Any] = {"check": reports}
    if cache is not None:
        extras["raw_manifest"] = {"files": len(cache.manifest.entries)}
    return extras


MARKET_STAGE = Stage(
    number=2,
    name="market",
    help="build or simulate prices, curves, consensus, calendar and regimes per market seed",
    run=run,
    reads=(),
    writes=MARKET_TABLES,
)
