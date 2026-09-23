"""Tests for the market instrument universe: identities, ids and deterministic order."""

import re
from fractions import Fraction

from pm_traitbench.config import Config
from pm_traitbench.enums import HY_BANDS, CommodityGroup, Family, InstrumentKind, RatingBand
from pm_traitbench.market.constants import COMMODITIES, FX_PAIRS, sector_label
from pm_traitbench.market.universe import build_universe
from pm_traitbench.rng import stream

_CREDIT_BAND_ORDER = (
    RatingBand.AA,
    RatingBand.A,
    RatingBand.BBB,
    RatingBand.BB,
    RatingBand.B,
)

_EQ_ID = re.compile(r"^EQ-\d{4}$")
_CR_ID = re.compile(r"^CR-(IG|HY)-\d{3}$")
_RT_ID = re.compile(r"^RT-[A-Z]{3}$")
_CM_ID = re.compile(r"^CM-[A-Z]{3}$")
_FX_ID = re.compile(r"^FX-[A-Z]{6}$")

_ID_PATTERN_BY_KIND = {
    InstrumentKind.EQUITY: _EQ_ID,
    InstrumentKind.CREDIT_ISSUER: _CR_ID,
    InstrumentKind.SOVEREIGN_CURVE: _RT_ID,
    InstrumentKind.COMMODITY: _CM_ID,
    InstrumentKind.FX_PAIR: _FX_ID,
}


def _build(config: Config | None = None, seed: int = 1):
    config = config or Config()
    rng = stream(seed, "market", "universe")
    return build_universe(config, rng)


def _family_counts(instruments):
    counts = {family: 0 for family in Family}
    for instrument in instruments:
        counts[instrument.family] += 1
    return counts


def test_default_family_counts() -> None:
    instruments = _build()
    counts = _family_counts(instruments)
    assert counts[Family.EQUITIES] == 80
    assert counts[Family.CREDIT] == 48
    assert counts[Family.RATES] == 4
    assert counts[Family.COMMODITIES] == 20
    assert counts[Family.FX] == 9
    assert len(instruments) == 161


def test_instrument_ids_are_unique_and_match_kind_pattern() -> None:
    instruments = _build()
    ids = [instrument.instrument_id for instrument in instruments]
    assert len(set(ids)) == len(ids)
    for instrument in instruments:
        assert _ID_PATTERN_BY_KIND[instrument.kind].match(instrument.instrument_id)


def test_credit_band_counts_match_largest_remainder() -> None:
    instruments = _build()
    credit = [i for i in instruments if i.kind == InstrumentKind.CREDIT_ISSUER]
    counts = {band: 0 for band in RatingBand}
    for instrument in credit:
        counts[instrument.rating_band] += 1
    assert counts[RatingBand.AA] == 7
    assert counts[RatingBand.A] == 12
    assert counts[RatingBand.BBB] == 14
    assert counts[RatingBand.BB] == 10
    assert counts[RatingBand.B] == 5


def _exact_credit_band_counts(n: int, shares: dict[RatingBand, Fraction]) -> dict[RatingBand, int]:
    """Reimplement the largest-remainder split in exact fractions, immune to float noise."""
    raw = {band: n * shares[band] for band in _CREDIT_BAND_ORDER}
    floors = {band: int(raw[band]) for band in _CREDIT_BAND_ORDER}
    remainder = n - sum(floors.values())
    ranked = sorted(
        _CREDIT_BAND_ORDER,
        key=lambda band: (-(raw[band] - floors[band]), _CREDIT_BAND_ORDER.index(band)),
    )
    counts = dict(floors)
    for band in ranked[:remainder]:
        counts[band] += 1
    return counts


def test_credit_band_counts_break_exact_remainder_ties_by_band_order() -> None:
    # 45 x (.35, .15, .15, .20, .15) floors to 15, 6, 6, 9, 6 (sum 42); AA, A,
    # BBB and B tie at a remainder of exactly 0.75, so the 3 spare seats must
    # go to the first three tied bands in band order: AA, A, BBB. The split
    # must round the remainder before ranking, or float noise in 0.35 * 45
    # can separate an exact tie and break the order.
    shares = {
        RatingBand.AA: Fraction(35, 100),
        RatingBand.A: Fraction(15, 100),
        RatingBand.BBB: Fraction(15, 100),
        RatingBand.BB: Fraction(20, 100),
        RatingBand.B: Fraction(15, 100),
    }
    n = 45
    expected = _exact_credit_band_counts(n, shares)
    assert sum(expected.values()) == n

    config = Config.model_validate(
        {
            "market": {
                "universe": {
                    "n_credit_issuers": n,
                    "credit_band_shares": {
                        band.value: float(share) for band, share in shares.items()
                    },
                }
            }
        }
    )
    instruments = _build(config)
    counts = {band: 0 for band in RatingBand}
    for instrument in instruments:
        if instrument.kind == InstrumentKind.CREDIT_ISSUER:
            counts[instrument.rating_band] += 1
    assert counts == expected


def test_credit_currency_is_one_of_the_configured_curves() -> None:
    config = Config()
    instruments = _build(config)
    curves = set(config.market.universe.curves)
    credit = [i for i in instruments if i.kind == InstrumentKind.CREDIT_ISSUER]
    assert credit
    for instrument in credit:
        assert instrument.currency in curves


def test_equity_sectors_cycle_through_sector_labels() -> None:
    config = Config()
    instruments = _build(config)
    equities = [i for i in instruments if i.kind == InstrumentKind.EQUITY]
    n_sectors = config.market.universe.n_sectors
    expected = [sector_label((i % n_sectors) + 1) for i in range(len(equities))]
    assert [instrument.sector for instrument in equities] == expected


def test_ig_ids_only_on_investment_grade_bands() -> None:
    instruments = _build()
    credit = [i for i in instruments if i.kind == InstrumentKind.CREDIT_ISSUER]
    assert credit
    for instrument in credit:
        is_ig_id = "-IG-" in instrument.instrument_id
        is_hy_band = instrument.rating_band in HY_BANDS
        assert is_ig_id != is_hy_band


def test_equity_betas_and_credit_durations_within_configured_range() -> None:
    config = Config()
    instruments = _build(config)
    lo, hi = config.market.universe.equity_beta_range
    dur_lo, dur_hi = config.market.universe.credit_duration_range
    checked_equity = checked_credit = False
    for instrument in instruments:
        if instrument.kind == InstrumentKind.EQUITY:
            assert lo <= instrument.beta <= hi
            checked_equity = True
        elif instrument.kind == InstrumentKind.CREDIT_ISSUER:
            assert dur_lo <= instrument.duration_years <= dur_hi
            checked_credit = True
    assert checked_equity
    assert checked_credit


def test_same_seed_gives_identical_output() -> None:
    first = _build(seed=42)
    second = _build(seed=42)
    assert [i.model_dump() for i in first] == [i.model_dump() for i in second]


def test_demo_sized_config_gives_expected_counts() -> None:
    config = Config.model_validate(
        {"market": {"universe": {"n_equities": 20, "n_sectors": 5, "n_credit_issuers": 12}}}
    )
    instruments = _build(config)
    counts = _family_counts(instruments)
    assert counts[Family.EQUITIES] == 20
    assert counts[Family.CREDIT] == 12
    assert counts[Family.RATES] == 4
    assert counts[Family.COMMODITIES] == 20
    assert counts[Family.FX] == 9
    assert len(instruments) == 65


def test_universe_order_is_equities_credit_curves_commodities_fx() -> None:
    instruments = _build()
    kinds = [instrument.kind for instrument in instruments]
    expected_order = (
        InstrumentKind.EQUITY,
        InstrumentKind.CREDIT_ISSUER,
        InstrumentKind.SOVEREIGN_CURVE,
        InstrumentKind.COMMODITY,
        InstrumentKind.FX_PAIR,
    )
    boundaries = [0]
    for kind in expected_order:
        boundaries.append(boundaries[-1] + sum(1 for k in kinds if k == kind))
    for kind, start, end in zip(expected_order, boundaries, boundaries[1:], strict=False):
        assert kinds[start:end] == [kind] * (end - start)
    assert boundaries[-1] == len(kinds)


def test_curve_instrument_identities() -> None:
    config = Config()
    instruments = _build(config)
    curves = [i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE]
    assert [i.instrument_id for i in curves] == [f"RT-{c}" for c in config.market.universe.curves]
    for instrument, ccy in zip(curves, config.market.universe.curves, strict=True):
        assert instrument.name == f"{ccy} sovereign curve"
        assert instrument.currency == ccy


def test_commodity_instrument_identities() -> None:
    config = Config()
    instruments = _build(config)
    commodities = [i for i in instruments if i.kind == InstrumentKind.COMMODITY]
    expected_ids = []
    for group in CommodityGroup:
        n = config.market.universe.commodities[group]
        expected_ids.extend(f"CM-{spec.code}" for spec in COMMODITIES[group][:n])
    assert [i.instrument_id for i in commodities] == expected_ids
    for instrument in commodities:
        assert instrument.currency == "USD"
        assert instrument.expiry_rule == config.market.universe.expiry_rule


def test_fx_instrument_identities() -> None:
    config = Config()
    instruments = _build(config)
    fx = [i for i in instruments if i.kind == InstrumentKind.FX_PAIR]
    assert [i.instrument_id for i in fx] == [f"FX-{p}" for p in config.market.universe.fx_pairs]
    for instrument, pair in zip(fx, config.market.universe.fx_pairs, strict=True):
        assert instrument.name == pair
        assert instrument.currency == FX_PAIRS[pair][1]
