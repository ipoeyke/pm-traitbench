"""Fixed market code tables: instrument identities, not configuration choices.

Config only selects how many entries of each table are used and which are
turned on; the entries themselves (codes, names, starting levels) live here.
"""

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

from pm_traitbench.enums import CommodityGroup, RatingBand

HORIZON_DAYS_PER_YEAR = 260
ANNUALISATION_DAYS = 252

CREDIT_BAND_ORDER: tuple[RatingBand, ...] = (
    RatingBand.AA,
    RatingBand.A,
    RatingBand.BBB,
    RatingBand.BB,
    RatingBand.B,
)


def largest_remainder[K: Hashable](
    n: int, shares: Mapping[K, float], order: Sequence[K]
) -> dict[K, int]:
    """Split `n` items across `order` by `shares`, largest remainder first, ties
    broken by `order`. Config and the universe share this so their counts agree.
    """
    raw = {key: n * shares[key] for key in order}
    counts = {key: math.floor(raw[key]) for key in order}
    remainder = n - sum(counts.values())
    # Round before ranking so float noise cannot turn an exact tie into an order override.
    ranked = sorted(order, key=lambda key: (-round(raw[key] - counts[key], 9), order.index(key)))
    for key in ranked[:remainder]:
        counts[key] += 1
    return counts


@dataclass(frozen=True)
class CommoditySpec:
    name: str
    code: str
    start: float
    multiplier: float


COMMODITIES: dict[CommodityGroup, tuple[CommoditySpec, ...]] = {
    CommodityGroup.ENERGY: (
        CommoditySpec("crude", "CRD", 72, 1000),
        CommoditySpec("brent", "BRN", 76, 1000),
        CommoditySpec("gasoil", "GSO", 700, 100),
        CommoditySpec("gasoline", "GSL", 2.2, 42000),
        CommoditySpec("heating_oil", "HOL", 2.4, 42000),
        CommoditySpec("coal", "COL", 130, 1000),
    ),
    CommodityGroup.INDUSTRIAL_METALS: (
        CommoditySpec("copper", "CPR", 9000, 25),
        CommoditySpec("aluminium", "ALU", 2400, 25),
        CommoditySpec("nickel", "NKL", 16000, 6),
        CommoditySpec("zinc", "ZNC", 2700, 25),
    ),
    CommodityGroup.PRECIOUS: (
        CommoditySpec("gold", "GLD", 2400, 100),
        CommoditySpec("silver", "SLV", 28, 5000),
        CommoditySpec("platinum", "PLT", 950, 50),
    ),
    CommodityGroup.AGRICULTURE: (
        CommoditySpec("wheat", "WHT", 600, 50),
        CommoditySpec("corn", "CRN", 450, 50),
        CommoditySpec("soybeans", "SOY", 1100, 50),
        CommoditySpec("sugar", "SGR", 20, 1120),
        CommoditySpec("coffee", "COF", 250, 375),
        CommoditySpec("cotton", "CTN", 75, 500),
        CommoditySpec("cocoa", "CCO", 8000, 10),
    ),
}

# Currency per one contract per one unit of quoted price, from exchange contract specs.
CONTRACT_MULTIPLIER: dict[str, float] = {
    spec.code: spec.multiplier for specs in COMMODITIES.values() for spec in specs
}

FX_PAIRS: dict[str, tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "AUDUSD": ("AUD", "USD"),
    "USDCHF": ("USD", "CHF"),
    "USDCAD": ("USD", "CAD"),
    "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"),
    "AUDJPY": ("AUD", "JPY"),
}

USD_PAIR: dict[str, str] = {
    "EUR": "EURUSD",
    "GBP": "GBPUSD",
    "JPY": "USDJPY",
    "AUD": "AUDUSD",
    "CHF": "USDCHF",
    "CAD": "USDCAD",
}

SECTOR_LABEL = "sector_{:02d}"


def sector_label(i: int) -> str:
    """Format a 1-indexed sector number as its label."""
    return SECTOR_LABEL.format(i)
