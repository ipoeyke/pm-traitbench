"""Instrument universe: the fixed set of tradable instruments per market seed.

Every family's instruments are built in a deterministic order, and random
draws within a family happen in one fixed sequence, so a given seed always
reproduces the same universe.
"""

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import HY_BANDS, CommodityGroup, Family, InstrumentKind, RatingBand
from pm_traitbench.market.constants import (
    COMMODITIES,
    CREDIT_BAND_ORDER,
    FX_PAIRS,
    largest_remainder,
    sector_label,
)
from pm_traitbench.tables.schema import Instrument


def _build_equities(config: Config, rng: np.random.Generator) -> list[Instrument]:
    universe = config.market.universe
    n = universe.n_equities
    lo, hi = universe.equity_beta_range
    betas = rng.uniform(lo, hi, size=n)

    instruments = []
    for i in range(n):
        num = i + 1
        sector_num = (i % universe.n_sectors) + 1
        instruments.append(
            Instrument(
                instrument_id=f"EQ-{num:04d}",
                family=Family.EQUITIES,
                kind=InstrumentKind.EQUITY,
                name=f"Equity {num:04d}",
                currency="USD",
                sector=sector_label(sector_num),
                rating_band=None,
                commodity_group=None,
                duration_years=None,
                beta=float(betas[i]),
                expiry_rule=None,
            )
        )
    return instruments


def _build_credit(config: Config, rng: np.random.Generator) -> list[Instrument]:
    universe = config.market.universe
    n = universe.n_credit_issuers
    counts = largest_remainder(n, universe.credit_band_shares, CREDIT_BAND_ORDER)

    issuers: list[tuple[str, str, RatingBand]] = []
    ig_counter = 0
    hy_counter = 0
    for band in CREDIT_BAND_ORDER:
        for _ in range(counts[band]):
            if band in HY_BANDS:
                hy_counter += 1
                issuers.append((f"CR-HY-{hy_counter:03d}", f"Issuer HY {hy_counter:03d}", band))
            else:
                ig_counter += 1
                issuers.append((f"CR-IG-{ig_counter:03d}", f"Issuer IG {ig_counter:03d}", band))

    curves = universe.curves
    dur_lo, dur_hi = universe.credit_duration_range
    sector_idx = rng.integers(0, universe.n_sectors, size=n)
    currency_idx = rng.integers(0, len(curves), size=n)
    durations = rng.uniform(dur_lo, dur_hi, size=n)

    instruments = []
    for i, (instrument_id, name, band) in enumerate(issuers):
        instruments.append(
            Instrument(
                instrument_id=instrument_id,
                family=Family.CREDIT,
                kind=InstrumentKind.CREDIT_ISSUER,
                name=name,
                currency=curves[currency_idx[i]],
                sector=sector_label(int(sector_idx[i]) + 1),
                rating_band=band,
                commodity_group=None,
                duration_years=float(durations[i]),
                beta=None,
                expiry_rule=None,
            )
        )
    return instruments


def _build_curves(config: Config) -> list[Instrument]:
    instruments = []
    for ccy in config.market.universe.curves:
        instruments.append(
            Instrument(
                instrument_id=f"RT-{ccy}",
                family=Family.RATES,
                kind=InstrumentKind.SOVEREIGN_CURVE,
                name=f"{ccy} sovereign curve",
                currency=ccy,
                sector=None,
                rating_band=None,
                commodity_group=None,
                duration_years=None,
                beta=None,
                expiry_rule=None,
            )
        )
    return instruments


def _build_commodities(config: Config) -> list[Instrument]:
    universe = config.market.universe
    instruments = []
    for group in CommodityGroup:
        n = universe.commodities[group]
        for spec in COMMODITIES[group][:n]:
            instruments.append(
                Instrument(
                    instrument_id=f"CM-{spec.code}",
                    family=Family.COMMODITIES,
                    kind=InstrumentKind.COMMODITY,
                    name=spec.name,
                    currency="USD",
                    sector=None,
                    rating_band=None,
                    commodity_group=group,
                    duration_years=None,
                    beta=None,
                    expiry_rule=universe.expiry_rule,
                )
            )
    return instruments


def _build_fx(config: Config) -> list[Instrument]:
    instruments = []
    for pair in config.market.universe.fx_pairs:
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


def build_universe(config: Config, rng: np.random.Generator) -> list[Instrument]:
    """Build the instrument universe: equities, credit, curves, commodities, FX.

    Random draws happen in one fixed sequence: all equity betas, then all
    credit sectors, then all credit currencies, then all credit durations.
    """
    instruments: list[Instrument] = []
    instruments.extend(_build_equities(config, rng))
    instruments.extend(_build_credit(config, rng))
    instruments.extend(_build_curves(config))
    instruments.extend(_build_commodities(config))
    instruments.extend(_build_fx(config))
    return instruments
