"""Real-market instrument registry for the pilot's real historical market seed.

Fixed identities, tickers and FRED/Yahoo series ids, in the spirit of the
synthetic code tables in market/constants.py: config only picks which real
seeds run, the registry entries themselves are fixed here.
"""

from dataclasses import dataclass
from typing import Literal

from pm_traitbench.enums import CommodityGroup, Family, InstrumentKind, RatingBand, Tenor
from pm_traitbench.market.constants import COMMODITIES, FX_PAIRS, CommoditySpec, sector_label

Source = Literal["fred", "yahoo", "edgar"]


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
    cik: str | None = None


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

# SEC EDGAR CIKs, looked up once from https://www.sec.gov/files/company_tickers.json.
_EQUITY_CIKS: dict[str, str] = {
    "AAPL": "0000320193", "MSFT": "0000789019", "INTC": "0000050863", "CSCO": "0000858877",
    "JNJ": "0000200406", "PFE": "0000078003", "MRK": "0000310158", "UNH": "0000731766",
    "JPM": "0000019617", "BAC": "0000070858", "WFC": "0000072971", "GS": "0000886982",
    "AMZN": "0001018724", "HD": "0000354950", "MCD": "0000063908", "NKE": "0000320187",
    "PG": "0000080424", "KO": "0000021344", "PEP": "0000077476", "WMT": "0000104169",
    "XOM": "0002115436", "CVX": "0000093410", "COP": "0001163165", "SLB": "0000087347",
    "BA": "0000012927", "CAT": "0000018230", "HON": "0000773840", "UNP": "0000100885",
    "APD": "0000002969", "ECL": "0000031462", "NEM": "0001164727", "SHW": "0000089800",
    "NEE": "0000753308", "DUK": "0001326160", "SO": "0000092122", "D": "0000715957",
    "VZ": "0000732712", "T": "0000732717", "DIS": "0001744489", "CMCSA": "0001166691",
}  # fmt: skip


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
                cik=_EQUITY_CIKS[ticker],
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
