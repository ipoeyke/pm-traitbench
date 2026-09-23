"""Real-market instrument universe: the registry's fixed identities, priced
against the fetched raw cache, in the same family order as the synthetic
universe (equities, credit, curve, commodities, FX).
"""

from collections.abc import Mapping
from datetime import date

import numpy as np

from pm_traitbench.config import Config, RealSeedSpec, referenced_seeds
from pm_traitbench.enums import Family, InstrumentKind
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.constants import FX_PAIRS
from pm_traitbench.market.real.align import aligned_series
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.sources import (
    CREDIT_DURATION_YEARS,
    DERIVED_FX_PAIRS,
    REAL_INSTRUMENTS,
    REFERENCE_EQUITY,
)
from pm_traitbench.tables.schema import Instrument


def real_axis_dates(spec: RealSeedSpec, axis: SimAxis, calendar_start: date) -> list[date]:
    """Map every axis day, burn-in included, onto its real historical date.

    `window_start` and `calendar_start` are both Mondays, so the offset
    between them is a whole number of weeks and every mapped date keeps the
    weekday of its axis day.
    """
    offset = spec.window_start - calendar_start
    return [day + offset for day in axis.dates]


def _referenced_real_specs(config: Config) -> dict[str, RealSeedSpec]:
    used = set(referenced_seeds(config))
    return {name: spec for name, spec in config.market.real.seeds.items() if name in used}


def _log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))


def _pooled_log_returns(
    cache: RawCache,
    ticker: str,
    specs: Mapping[str, RealSeedSpec],
    axis: SimAxis,
    calendar_start: date,
) -> np.ndarray:
    """Log returns for `ticker`, computed within each real seed's own window and
    concatenated across seeds: no return spans the gap between two windows.
    """
    parts = []
    for spec in specs.values():
        dates = real_axis_dates(spec, axis, calendar_start)
        prices, _ = aligned_series(cache.yahoo(ticker), dates, name=ticker)
        parts.append(_log_returns(prices))
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _beta(spy_returns: np.ndarray, equity_returns: np.ndarray) -> float:
    slope, _ = np.polyfit(spy_returns, equity_returns, 1)
    return round(float(slope), 4)


def _build_equities(
    cache: RawCache, specs: Mapping[str, RealSeedSpec], axis: SimAxis, calendar_start: date
) -> list[Instrument]:
    spy_returns = _pooled_log_returns(cache, REFERENCE_EQUITY, specs, axis, calendar_start)

    instruments = []
    for inst in REAL_INSTRUMENTS:
        if inst.family != Family.EQUITIES:
            continue
        equity_returns = _pooled_log_returns(cache, inst.series, specs, axis, calendar_start)
        beta = _beta(spy_returns, equity_returns)
        instruments.append(
            Instrument(
                instrument_id=inst.instrument_id,
                family=Family.EQUITIES,
                kind=InstrumentKind.EQUITY,
                name=inst.name,
                currency=inst.currency,
                sector=inst.sector,
                rating_band=None,
                commodity_group=None,
                duration_years=None,
                beta=beta,
                expiry_rule=None,
            )
        )
    return instruments


def _build_credit() -> list[Instrument]:
    return [
        Instrument(
            instrument_id=inst.instrument_id,
            family=Family.CREDIT,
            kind=InstrumentKind.CREDIT_ISSUER,
            name=inst.name,
            currency=inst.currency,
            sector=inst.sector,
            rating_band=inst.rating_band,
            commodity_group=None,
            duration_years=CREDIT_DURATION_YEARS,
            beta=None,
            expiry_rule=None,
        )
        for inst in REAL_INSTRUMENTS
        if inst.family == Family.CREDIT
    ]


def _build_curve() -> list[Instrument]:
    return [
        Instrument(
            instrument_id=inst.instrument_id,
            family=Family.RATES,
            kind=InstrumentKind.SOVEREIGN_CURVE,
            name=inst.name,
            currency=inst.currency,
            sector=None,
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=None,
            expiry_rule=None,
        )
        for inst in REAL_INSTRUMENTS
        if inst.family == Family.RATES
    ]


def _build_commodities(config: Config) -> list[Instrument]:
    rule = config.market.universe.expiry_rule
    return [
        Instrument(
            instrument_id=inst.instrument_id,
            family=Family.COMMODITIES,
            kind=InstrumentKind.COMMODITY,
            name=inst.name,
            currency=inst.currency,
            sector=None,
            rating_band=None,
            commodity_group=inst.commodity_group,
            duration_years=None,
            beta=None,
            expiry_rule=rule,
        )
        for inst in REAL_INSTRUMENTS
        if inst.family == Family.COMMODITIES
    ]


def _build_fx() -> list[Instrument]:
    instruments = [
        Instrument(
            instrument_id=inst.instrument_id,
            family=Family.FX,
            kind=InstrumentKind.FX_PAIR,
            name=inst.name,
            currency=inst.currency,
            sector=None,
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=None,
            expiry_rule=None,
        )
        for inst in REAL_INSTRUMENTS
        if inst.family == Family.FX
    ]
    for pair in DERIVED_FX_PAIRS:
        _, quote = FX_PAIRS[pair]
        instruments.append(
            Instrument(
                instrument_id=f"FX-{pair}",
                family=Family.FX,
                kind=InstrumentKind.FX_PAIR,
                name=pair,
                currency=quote,
                sector=None,
                rating_band=None,
                commodity_group=None,
                duration_years=None,
                beta=None,
                expiry_rule=None,
            )
        )
    return instruments


def build_real_universe(config: Config, cache: RawCache, axis: SimAxis) -> list[Instrument]:
    """Build the real instrument universe: equities, credit, curve, commodities,
    FX (the registry's USD pairs, then the derived cross pairs).
    """
    specs = _referenced_real_specs(config)
    instruments: list[Instrument] = []
    instruments.extend(_build_equities(cache, specs, axis, config.calendar.start))
    instruments.extend(_build_credit())
    instruments.extend(_build_curve())
    instruments.extend(_build_commodities(config))
    instruments.extend(_build_fx())
    return instruments
