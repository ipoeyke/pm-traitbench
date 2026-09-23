"""Tests for the real-market instrument registry."""

from collections import Counter

from pm_traitbench.enums import Family
from pm_traitbench.market.real.sources import (
    REAL_INSTRUMENTS,
    fred_series,
    yahoo_tickers,
)


def test_family_counts_match_the_registry() -> None:
    counts = Counter(inst.family for inst in REAL_INSTRUMENTS)
    assert counts[Family.EQUITIES] == 40
    assert counts[Family.RATES] == 1
    assert counts[Family.CREDIT] == 2
    assert counts[Family.COMMODITIES] == 15
    assert counts[Family.FX] == 6


def test_equities_spread_across_ten_sectors_of_four() -> None:
    sectors = Counter(inst.sector for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES)
    assert len(sectors) == 10
    assert set(sectors.values()) == {4}


def test_equity_id_format() -> None:
    equities = [inst for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES]
    ids = [inst.instrument_id for inst in equities]
    assert ids[0] == "EQ-R001"
    assert ids[-1] == "EQ-R040"
    for inst in equities:
        assert inst.name == inst.instrument_id
        assert inst.currency == "USD"
        assert inst.source == "yahoo"


def test_credit_registry_entries() -> None:
    credit = {inst.instrument_id: inst for inst in REAL_INSTRUMENTS if inst.family == Family.CREDIT}
    assert set(credit) == {"CR-R-IG-001", "CR-R-IG-002"}
    for inst in credit.values():
        assert inst.spread_base == "DGS20"
        assert inst.name == inst.instrument_id
        assert inst.currency == "USD"
    assert credit["CR-R-IG-001"].series == "DAAA"
    assert credit["CR-R-IG-002"].series == "DBAA"


def test_curve_registry_entry() -> None:
    curves = [inst for inst in REAL_INSTRUMENTS if inst.family == Family.RATES]
    assert len(curves) == 1
    curve = curves[0]
    assert curve.instrument_id == "RT-USD"
    assert curve.series == "DGS10"
    assert curve.source == "fred"


def test_commodity_registry_tickers() -> None:
    commodities = {
        inst.series: inst.instrument_id
        for inst in REAL_INSTRUMENTS
        if inst.family == Family.COMMODITIES
    }
    expected = {
        "CL=F", "BZ=F", "HO=F", "RB=F", "GC=F", "SI=F", "PL=F", "HG=F",
        "ZW=F", "ZC=F", "ZS=F", "SB=F", "KC=F", "CT=F", "CC=F",
    }  # fmt: skip
    assert set(commodities) == expected


def test_fx_registry_entries() -> None:
    fx = {inst.instrument_id: inst for inst in REAL_INSTRUMENTS if inst.family == Family.FX}
    assert fx["FX-EURUSD"].series == "DEXUSEU"
    assert fx["FX-GBPUSD"].series == "DEXUSUK"
    assert fx["FX-USDJPY"].series == "DEXJPUS"
    assert fx["FX-AUDUSD"].series == "DEXUSAL"
    assert fx["FX-USDCHF"].series == "DEXSZUS"
    assert fx["FX-USDCAD"].series == "DEXCAUS"
    assert fx["FX-USDJPY"].currency == "JPY"
    assert fx["FX-EURUSD"].currency == "USD"
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
