"""Tests for the real-market instrument registry."""

from collections import Counter

from pm_traitbench.enums import CommodityGroup, Family, InstrumentKind, RatingBand
from pm_traitbench.market.real.sources import (
    REAL_INSTRUMENTS,
    fred_series,
    yahoo_tickers,
)

_EXPECTED_EQUITY_TICKERS: tuple[str, ...] = (
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

_EXPECTED_COMMODITY_TICKERS: dict[str, str] = {
    "CM-CRD": "CL=F",
    "CM-BRN": "BZ=F",
    "CM-HOL": "HO=F",
    "CM-GSL": "RB=F",
    "CM-GLD": "GC=F",
    "CM-SLV": "SI=F",
    "CM-PLT": "PL=F",
    "CM-CPR": "HG=F",
    "CM-WHT": "ZW=F",
    "CM-CRN": "ZC=F",
    "CM-SOY": "ZS=F",
    "CM-SGR": "SB=F",
    "CM-COF": "KC=F",
    "CM-CTN": "CT=F",
    "CM-CCO": "CC=F",
}

_EXPECTED_COMMODITY_GROUPS: dict[str, CommodityGroup] = {
    "CM-CRD": CommodityGroup.ENERGY,
    "CM-BRN": CommodityGroup.ENERGY,
    "CM-HOL": CommodityGroup.ENERGY,
    "CM-GSL": CommodityGroup.ENERGY,
    "CM-GLD": CommodityGroup.PRECIOUS,
    "CM-SLV": CommodityGroup.PRECIOUS,
    "CM-PLT": CommodityGroup.PRECIOUS,
    "CM-CPR": CommodityGroup.INDUSTRIAL_METALS,
    "CM-WHT": CommodityGroup.AGRICULTURE,
    "CM-CRN": CommodityGroup.AGRICULTURE,
    "CM-SOY": CommodityGroup.AGRICULTURE,
    "CM-SGR": CommodityGroup.AGRICULTURE,
    "CM-COF": CommodityGroup.AGRICULTURE,
    "CM-CTN": CommodityGroup.AGRICULTURE,
    "CM-CCO": CommodityGroup.AGRICULTURE,
}


def test_family_counts_match_the_registry() -> None:
    counts = Counter(inst.family for inst in REAL_INSTRUMENTS)
    assert counts[Family.EQUITIES] == 40
    assert counts[Family.RATES] == 1
    assert counts[Family.CREDIT] == 2
    assert counts[Family.COMMODITIES] == 15
    assert counts[Family.FX] == 6


def test_equity_ticker_to_id_order_matches_the_registry() -> None:
    equities = [inst for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES]
    assert [inst.series for inst in equities] == list(_EXPECTED_EQUITY_TICKERS)
    assert [inst.instrument_id for inst in equities] == [f"EQ-R{i:03d}" for i in range(1, 41)]
    for inst in equities:
        assert inst.name == inst.instrument_id
        assert inst.currency == "USD"
        assert inst.source == "yahoo"


def test_equity_sector_groups_are_four_tickers_each_in_order() -> None:
    equities = [inst for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES]
    sectors = [inst.sector for inst in equities]
    assert sectors == [f"sector_{(i // 4) + 1:02d}" for i in range(40)]
    assert Counter(sectors) == {f"sector_{n:02d}": 4 for n in range(1, 11)}


def test_credit_registry_entries() -> None:
    credit = {inst.instrument_id: inst for inst in REAL_INSTRUMENTS if inst.family == Family.CREDIT}
    assert set(credit) == {"CR-R-IG-001", "CR-R-IG-002"}
    for inst in credit.values():
        assert inst.spread_base == "DGS20"
        assert inst.name == inst.instrument_id
        assert inst.currency == "USD"
        assert inst.sector == "sector_01"
    assert credit["CR-R-IG-001"].series == "DAAA"
    assert credit["CR-R-IG-002"].series == "DBAA"
    assert credit["CR-R-IG-001"].rating_band == RatingBand.AA
    assert credit["CR-R-IG-002"].rating_band == RatingBand.BBB


def test_curve_registry_entry() -> None:
    curves = [inst for inst in REAL_INSTRUMENTS if inst.family == Family.RATES]
    assert len(curves) == 1
    curve = curves[0]
    assert curve.instrument_id == "RT-USD"
    assert curve.family == Family.RATES
    assert curve.kind == InstrumentKind.SOVEREIGN_CURVE
    assert curve.series == "DGS10"
    assert curve.source == "fred"


def test_commodity_ids_and_groups_match_exactly() -> None:
    commodities = {
        inst.instrument_id: inst for inst in REAL_INSTRUMENTS if inst.family == Family.COMMODITIES
    }
    assert set(commodities) == set(_EXPECTED_COMMODITY_TICKERS)
    for instrument_id, ticker in _EXPECTED_COMMODITY_TICKERS.items():
        assert commodities[instrument_id].series == ticker
        assert (
            commodities[instrument_id].commodity_group == _EXPECTED_COMMODITY_GROUPS[instrument_id]
        )
        assert commodities[instrument_id].source == "yahoo"


def test_fx_registry_entries() -> None:
    fx = {inst.instrument_id: inst for inst in REAL_INSTRUMENTS if inst.family == Family.FX}
    assert fx["FX-EURUSD"].series == "DEXUSEU"
    assert fx["FX-GBPUSD"].series == "DEXUSUK"
    assert fx["FX-USDJPY"].series == "DEXJPUS"
    assert fx["FX-AUDUSD"].series == "DEXUSAL"
    assert fx["FX-USDCHF"].series == "DEXSZUS"
    assert fx["FX-USDCAD"].series == "DEXCAUS"
    assert fx["FX-USDJPY"].currency == "JPY"
    assert fx["FX-USDCHF"].currency == "CHF"
    assert fx["FX-USDCAD"].currency == "CAD"
    assert fx["FX-EURUSD"].currency == "USD"
    assert fx["FX-GBPUSD"].currency == "USD"
    assert fx["FX-AUDUSD"].currency == "USD"
    for inst in fx.values():
        assert inst.source == "fred"


def test_fred_series_has_thirteen_ids() -> None:
    ids = fred_series()
    assert len(ids) == 13
    assert len(set(ids)) == 13


def test_yahoo_tickers_has_fifty_six() -> None:
    tickers = yahoo_tickers()
    assert len(tickers) == 56
    assert "SPY" in tickers


def test_no_duplicate_ids_or_series() -> None:
    ids = [inst.instrument_id for inst in REAL_INSTRUMENTS]
    assert len(ids) == len(set(ids))
    series = [inst.series for inst in REAL_INSTRUMENTS]
    assert len(series) == len(set(series))
