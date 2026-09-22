"""Fixed market code tables: instrument identities, not configuration choices.

Config only selects how many entries of each table are used and which are
turned on; the entries themselves (codes, names, starting levels) live here.
"""

from dataclasses import dataclass

from pm_traitbench.enums import CommodityGroup

HORIZON_DAYS_PER_YEAR = 260
ANNUALISATION_DAYS = 252


@dataclass(frozen=True)
class CommoditySpec:
    name: str
    code: str
    start: float


COMMODITIES: dict[CommodityGroup, tuple[CommoditySpec, ...]] = {
    CommodityGroup.ENERGY: (
        CommoditySpec("crude", "CRD", 72),
        CommoditySpec("brent", "BRN", 76),
        CommoditySpec("gasoil", "GSO", 700),
        CommoditySpec("gasoline", "GSL", 2.2),
        CommoditySpec("heating_oil", "HOL", 2.4),
        CommoditySpec("coal", "COL", 130),
    ),
    CommodityGroup.INDUSTRIAL_METALS: (
        CommoditySpec("copper", "CPR", 9000),
        CommoditySpec("aluminium", "ALU", 2400),
        CommoditySpec("nickel", "NKL", 16000),
        CommoditySpec("zinc", "ZNC", 2700),
    ),
    CommodityGroup.PRECIOUS: (
        CommoditySpec("gold", "GLD", 2400),
        CommoditySpec("silver", "SLV", 28),
        CommoditySpec("platinum", "PLT", 950),
    ),
    CommodityGroup.AGRICULTURE: (
        CommoditySpec("wheat", "WHT", 600),
        CommoditySpec("corn", "CRN", 450),
        CommoditySpec("soybeans", "SOY", 1100),
        CommoditySpec("sugar", "SGR", 20),
        CommoditySpec("coffee", "COF", 250),
        CommoditySpec("cotton", "CTN", 75),
        CommoditySpec("cocoa", "CCO", 8000),
    ),
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
