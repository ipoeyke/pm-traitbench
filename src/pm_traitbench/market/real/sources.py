"""Real-market instrument registry for the pilot's real historical market seed.

Fixed identities, tickers and FRED/Yahoo series ids, in the spirit of the
synthetic code tables in market/constants.py: config only picks which real
seeds run, the registry entries themselves are fixed here.
"""

from dataclasses import dataclass
from typing import Literal

from pm_traitbench.enums import CommodityGroup, Family, InstrumentKind, RatingBand, Tenor
from pm_traitbench.market.constants import COMMODITIES, FX_PAIRS, CommoditySpec, sector_label

Source = Literal["fred", "yahoo"]


@dataclass(frozen=True)
class RealInstrument:
    """One registry instrument: its identity plus where its raw data comes from."""

    instrument_id: str
    family: Family
    kind: InstrumentKind
    name: str
    currency: str
    source: Source
    series: str
    sector: str | None = None
    rating_band: RatingBand | None = None
    commodity_group: CommodityGroup | None = None
    spread_base: str | None = None


REFERENCE_EQUITY = "SPY"
# Moody's seasoned indices hold maturities of 20 years or more.
CREDIT_DURATION_YEARS = 13.0
DERIVED_FX_PAIRS: tuple[str, ...] = ("EURGBP", "EURJPY", "AUDJPY")
CURVE_SERIES: dict[Tenor, str] = {
    Tenor.Y2: "DGS2",
    Tenor.Y5: "DGS5",
    Tenor.Y10: "DGS10",
    Tenor.Y30: "DGS30",
}

_EQUITY_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "INTC", "CSCO",
    "JNJ", "PFE", "MRK", "UNH",
    "JPM", "BAC", "WFC", "GS",
    "AMZN", "HD", "MCD", "NKE",
    "PG", "KO", "PEP", "WMT",
    "XOM", "CVX", "COP", "SLB",
    "BA", "CAT", "HON", "UNP",
    "APD", "ECL", "NEM", "SHW",
    "NEE", "DUK", "SO", "D",
    "VZ", "T", "DIS", "CMCSA",
)  # fmt: skip


def _build_equities() -> list[RealInstrument]:
    instruments = []
    for i, ticker in enumerate(_EQUITY_TICKERS):
        instrument_id = f"EQ-R{i + 1:03d}"
        instruments.append(
            RealInstrument(
                instrument_id=instrument_id,
                family=Family.EQUITIES,
                kind=InstrumentKind.EQUITY,
                name=instrument_id,
                currency="USD",
                source="yahoo",
                series=ticker,
                sector=sector_label(i // 4 + 1),
            )
        )
    return instruments


def _build_credit() -> list[RealInstrument]:
    specs = (
        ("CR-R-IG-001", "DAAA", RatingBand.AA),
        ("CR-R-IG-002", "DBAA", RatingBand.BBB),
    )
    return [
        RealInstrument(
            instrument_id=instrument_id,
            family=Family.CREDIT,
            kind=InstrumentKind.CREDIT_ISSUER,
            name=instrument_id,
            currency="USD",
            source="fred",
            series=series,
            sector=sector_label(1),
            rating_band=band,
            spread_base="DGS20",
        )
        for instrument_id, series, band in specs
    ]


def _build_curve() -> list[RealInstrument]:
    return [
        RealInstrument(
            instrument_id="RT-USD",
            family=Family.RATES,
            kind=InstrumentKind.SOVEREIGN_CURVE,
            name="USD sovereign curve",
            currency="USD",
            source="fred",
            series="DGS10",
        )
    ]


# Yahoo continuous-front-month futures tickers, in registry order.
_COMMODITY_ORDER: tuple[str, ...] = (
    "crude", "brent", "heating_oil", "gasoline",
    "gold", "silver", "platinum", "copper",
    "wheat", "corn", "soybeans", "sugar", "coffee", "cotton", "cocoa",
)  # fmt: skip
_COMMODITY_TICKERS: dict[str, str] = dict(
    zip(
        _COMMODITY_ORDER,
        (
            "CL=F",
            "BZ=F",
            "HO=F",
            "RB=F",
            "GC=F",
            "SI=F",
            "PL=F",
            "HG=F",
            "ZW=F",
            "ZC=F",
            "ZS=F",
            "SB=F",
            "KC=F",
            "CT=F",
            "CC=F",
        ),  # fmt: skip
        strict=True,
    )
)


def _commodity_lookup() -> dict[str, tuple[CommoditySpec, CommodityGroup]]:
    return {spec.name: (spec, group) for group, specs in COMMODITIES.items() for spec in specs}


def _build_commodities() -> list[RealInstrument]:
    lookup = _commodity_lookup()
    instruments = []
    for name in _COMMODITY_ORDER:
        spec, group = lookup[name]
        instruments.append(
            RealInstrument(
                instrument_id=f"CM-{spec.code}",
                family=Family.COMMODITIES,
                kind=InstrumentKind.COMMODITY,
                name=spec.name,
                currency="USD",
                source="yahoo",
                series=_COMMODITY_TICKERS[name],
                commodity_group=group,
            )
        )
    return instruments


_FX_SERIES: dict[str, str] = {
    "EURUSD": "DEXUSEU",
    "GBPUSD": "DEXUSUK",
    "USDJPY": "DEXJPUS",
    "AUDUSD": "DEXUSAL",
    "USDCHF": "DEXSZUS",
    "USDCAD": "DEXCAUS",
}


def _build_fx() -> list[RealInstrument]:
    instruments = []
    for pair, series in _FX_SERIES.items():
        _, quote = FX_PAIRS[pair]
        instruments.append(
            RealInstrument(
                instrument_id=f"FX-{pair}",
                family=Family.FX,
                kind=InstrumentKind.FX_PAIR,
                name=pair,
                currency=quote,
                source="fred",
                series=series,
            )
        )
    return instruments


REAL_INSTRUMENTS: tuple[RealInstrument, ...] = tuple(
    _build_equities() + _build_credit() + _build_curve() + _build_commodities() + _build_fx()
)


def fred_series() -> list[str]:
    """Every FRED series id the fetch needs: registry series, credit spread bases
    and every sovereign curve tenor.
    """
    ids = {inst.series for inst in REAL_INSTRUMENTS if inst.source == "fred"}
    ids |= {inst.spread_base for inst in REAL_INSTRUMENTS if inst.spread_base}
    ids |= set(CURVE_SERIES.values())
    return sorted(ids)


def yahoo_tickers() -> list[str]:
    """Every Yahoo ticker the fetch needs: the reference equity plus every
    registry equity and commodity.
    """
    return [REFERENCE_EQUITY] + [inst.series for inst in REAL_INSTRUMENTS if inst.source == "yahoo"]
