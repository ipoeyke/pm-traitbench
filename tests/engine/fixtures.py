"""Shared fixtures for behaviour-engine tests: a small deterministic market and a neutral PM."""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pm_traitbench.catalogues.loader import render_template
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.enums import (
    Action,
    AssetClass,
    CommodityGroup,
    EventType,
    ExpiryRule,
    Family,
    InstrumentKind,
    Kind,
    Op,
    Positioning,
    RatingBand,
    Regime,
    RuleScope,
    RuleSource,
    Split,
    StreetView,
    Tenor,
    Typicality,
)
from pm_traitbench.market.calendar import third_friday
from pm_traitbench.tables.schema import (
    CalendarEvent,
    ConsensusRow,
    CurvePoint,
    Instrument,
    Mandate,
    Persona,
    Price,
    RegimeSpan,
    Rule,
    StatedProfile,
    Trait,
)
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    MARKET_CALENDAR,
    MARKET_CONSENSUS,
    MARKET_CURVES,
    MARKET_INSTRUMENTS,
    MARKET_PRICES,
    MARKET_REGIMES,
    PERSONAS,
    RULES,
    TRAITS,
)

_SEED = "T"
_N_DAYS = 60
_SOVEREIGN_TENORS = (Tenor.Y2, Tenor.Y5, Tenor.Y10, Tenor.Y30)
_FUTURES_TENORS = tuple(Tenor(f"M{k}") for k in range(1, 13))


def _weekdays(start: date, n: int) -> list[date]:
    days: list[date] = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _street_view(score: float) -> StreetView:
    if score > 0.1:
        return StreetView.OVERWEIGHT
    if score < -0.1:
        return StreetView.UNDERWEIGHT
    return StreetView.NEUTRAL


def _positioning(pct: float) -> Positioning:
    if pct > 66:
        return Positioning.CROWDED_LONG
    if pct < 34:
        return Positioning.CROWDED_SHORT
    return Positioning.NEUTRAL


@pytest.fixture(scope="module")
def fixture_market() -> dict:
    """A small deterministic market on seed 'T': 60 horizon weekdays from 2026-01-05."""
    rng = np.random.default_rng(7)
    dates = _weekdays(date(2026, 1, 5), _N_DAYS)

    instruments = [
        Instrument(
            instrument_id="EQ-0001",
            family=Family.EQUITIES,
            kind=InstrumentKind.EQUITY,
            name="Equity 0001",
            currency="USD",
            sector="sector_01",
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=1.0,
            expiry_rule=None,
        ),
        Instrument(
            instrument_id="EQ-0002",
            family=Family.EQUITIES,
            kind=InstrumentKind.EQUITY,
            name="Equity 0002",
            currency="USD",
            sector="sector_01",
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=1.0,
            expiry_rule=None,
        ),
        Instrument(
            instrument_id="EQ-0003",
            family=Family.EQUITIES,
            kind=InstrumentKind.EQUITY,
            name="Equity 0003",
            currency="USD",
            sector="sector_01",
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=1.0,
            expiry_rule=None,
        ),
        Instrument(
            instrument_id="EQ-0004",
            family=Family.EQUITIES,
            kind=InstrumentKind.EQUITY,
            name="Equity 0004",
            currency="USD",
            sector="sector_02",
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=1.0,
            expiry_rule=None,
        ),
        Instrument(
            instrument_id="RT-USD",
            family=Family.RATES,
            kind=InstrumentKind.SOVEREIGN_CURVE,
            name="US Treasury Curve",
            currency="USD",
            sector=None,
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=None,
            expiry_rule=None,
        ),
        Instrument(
            instrument_id="CR-IG-001",
            family=Family.CREDIT,
            kind=InstrumentKind.CREDIT_ISSUER,
            name="IG Issuer 001",
            currency="USD",
            sector="sector_01",
            rating_band=RatingBand.AA,
            commodity_group=None,
            duration_years=6.0,
            beta=None,
            expiry_rule=None,
        ),
        Instrument(
            instrument_id="CR-IG-002",
            family=Family.CREDIT,
            kind=InstrumentKind.CREDIT_ISSUER,
            name="IG Issuer 002",
            currency="USD",
            sector="sector_02",
            rating_band=RatingBand.BBB,
            commodity_group=None,
            duration_years=4.0,
            beta=None,
            expiry_rule=None,
        ),
        Instrument(
            instrument_id="CM-CRD",
            family=Family.COMMODITIES,
            kind=InstrumentKind.COMMODITY,
            name="Crude",
            currency="USD",
            sector=None,
            rating_band=None,
            commodity_group=CommodityGroup.ENERGY,
            duration_years=None,
            beta=None,
            expiry_rule=ExpiryRule.MONTHLY_THIRD_FRIDAY,
        ),
        Instrument(
            instrument_id="CM-GLD",
            family=Family.COMMODITIES,
            kind=InstrumentKind.COMMODITY,
            name="Gold",
            currency="USD",
            sector=None,
            rating_band=None,
            commodity_group=CommodityGroup.PRECIOUS,
            duration_years=None,
            beta=None,
            expiry_rule=ExpiryRule.MONTHLY_THIRD_FRIDAY,
        ),
    ]

    equity_ids = ["EQ-0001", "EQ-0002", "EQ-0003", "EQ-0004"]
    commodity_ids = ["CM-CRD", "CM-GLD"]
    credit_spreads_start = {"CR-IG-001": 60.0, "CR-IG-002": 110.0}

    prices: list[Price] = []
    front_prices: dict[str, np.ndarray] = {}

    for instrument_id in equity_ids:
        walk = np.empty(_N_DAYS)
        level = 100.0
        for t in range(_N_DAYS):
            level *= math.exp(rng.normal(0.0, 0.015))
            walk[t] = level
        front_prices[instrument_id] = walk
        for t, day in enumerate(dates):
            prices.append(
                Price(
                    seed=_SEED, date=day, instrument_id=instrument_id, price=walk[t], spread_bp=None
                )
            )

    for instrument_id in commodity_ids:
        walk = np.empty(_N_DAYS)
        level = 50.0
        for t in range(_N_DAYS):
            level *= math.exp(rng.normal(0.0, 0.01))
            walk[t] = level
        front_prices[instrument_id] = walk
        for t, day in enumerate(dates):
            prices.append(
                Price(
                    seed=_SEED, date=day, instrument_id=instrument_id, price=walk[t], spread_bp=None
                )
            )

    for instrument_id, start in credit_spreads_start.items():
        spreads = np.empty(_N_DAYS)
        level = start
        for t in range(_N_DAYS):
            level += rng.normal(0.0, 2.0)
            spreads[t] = level
        for t, day in enumerate(dates):
            prices.append(
                Price(
                    seed=_SEED,
                    date=day,
                    instrument_id=instrument_id,
                    price=100.0,
                    spread_bp=float(spreads[t]),
                )
            )

    curves: list[CurvePoint] = []

    yields: dict[Tenor, np.ndarray] = {}
    for tenor, start in zip(_SOVEREIGN_TENORS, (3.5, 3.8, 4.0, 4.3), strict=True):
        path = np.empty(_N_DAYS)
        level = start
        for t in range(_N_DAYS):
            level += rng.normal(0.0, 0.05)
            path[t] = level
        yields[tenor] = path
    for tenor, path in yields.items():
        for t, day in enumerate(dates):
            curves.append(
                CurvePoint(
                    seed=_SEED, date=day, curve_id="RT-USD", tenor=tenor, level=float(path[t])
                )
            )

    for instrument_id in commodity_ids:
        m1 = front_prices[instrument_id]
        for k, tenor in enumerate(_FUTURES_TENORS, start=1):
            for t, day in enumerate(dates):
                level = float(m1[t]) * (1 + 0.02 * (k - 1) / 12)
                curves.append(
                    CurvePoint(
                        seed=_SEED, date=day, curve_id=instrument_id, tenor=tenor, level=level
                    )
                )

    consensus: list[ConsensusRow] = []
    consensus_ids = [*equity_ids, "CR-IG-001", "CR-IG-002", *commodity_ids]
    for instrument_id in consensus_ids:
        for t, day in enumerate(dates):
            base = 0.5 * math.sin(2 * math.pi * t / 20)
            score = base if t < 30 else -base
            score = max(-1.0, min(1.0, score))
            pct = max(0.0, min(100.0, 50.0 + 40.0 * score))
            consensus.append(
                ConsensusRow(
                    seed=_SEED,
                    date=day,
                    instrument_id=instrument_id,
                    street_score=score,
                    street_view=_street_view(score),
                    positioning_pct=pct,
                    positioning=_positioning(pct),
                )
            )

    calendar: list[CalendarEvent] = []
    calendar.append(
        CalendarEvent(
            seed=_SEED,
            date=dates[10],
            instrument_id="EQ-0001",
            event=EventType.EARNINGS,
            surprise=0.2,
            affected="equities",
        )
    )
    calendar.append(
        CalendarEvent(
            seed=_SEED,
            date=dates[40],
            instrument_id="EQ-0001",
            event=EventType.EARNINGS,
            surprise=-0.15,
            affected="equities",
        )
    )
    calendar.append(
        CalendarEvent(
            seed=_SEED,
            date=dates[20],
            instrument_id="RT-USD",
            event=EventType.CB_MEETING,
            surprise=0.1,
            affected="rates",
        )
    )
    calendar.append(
        CalendarEvent(
            seed=_SEED,
            date=dates[15],
            instrument_id="CM-CRD",
            event=EventType.INVENTORY_REPORT,
            surprise=0.3,
            affected="commodities",
        )
    )
    calendar.append(
        CalendarEvent(
            seed=_SEED,
            date=dates[25],
            instrument_id=None,
            event=EventType.MACRO_PRINT,
            surprise=0.4,
            affected="all",
        )
    )

    months = sorted({(day.year, day.month) for day in dates})
    horizon_set = set(dates)
    for year, month in months:
        expiry = third_friday(year, month)
        if expiry not in horizon_set:
            continue
        for instrument_id in commodity_ids:
            calendar.append(
                CalendarEvent(
                    seed=_SEED,
                    date=expiry,
                    instrument_id=instrument_id,
                    event=EventType.CONTRACT_EXPIRY,
                    surprise=None,
                    affected="commodities",
                )
            )

    flip_affected = {
        "EQ-0001": "equities",
        "EQ-0002": "equities",
        "EQ-0003": "equities",
        "EQ-0004": "equities",
        "CR-IG-001": "credit",
        "CR-IG-002": "credit",
        "CM-CRD": "commodities",
        "CM-GLD": "commodities",
    }
    for instrument_id, affected in flip_affected.items():
        calendar.append(
            CalendarEvent(
                seed=_SEED,
                date=dates[30],
                instrument_id=instrument_id,
                event=EventType.CONSENSUS_FLIP,
                surprise=None,
                affected=affected,
            )
        )

    regimes = [
        RegimeSpan(seed=_SEED, regime=Regime.RANGE, date_start=dates[0], date_end=dates[29]),
        RegimeSpan(seed=_SEED, regime=Regime.RISK_OFF, date_start=dates[30], date_end=dates[59]),
    ]

    return {
        "dates": tuple(dates),
        "instruments": instruments,
        "prices": prices,
        "curves": curves,
        "consensus": consensus,
        "calendar": calendar,
        "regimes": regimes,
    }


@pytest.fixture(scope="module")
def fixture_view(fixture_market: dict) -> MarketView:
    """The fixture market built into a `MarketView`."""
    return MarketView.build(
        seed=_SEED,
        dates=fixture_market["dates"],
        instruments=fixture_market["instruments"],
        prices=fixture_market["prices"],
        curves=fixture_market["curves"],
        consensus=fixture_market["consensus"],
        calendar=fixture_market["calendar"],
        regimes=fixture_market["regimes"],
    )


@pytest.fixture(scope="module")
def fixture_instruments(fixture_market: dict) -> list[Instrument]:
    """The fixture market's instrument universe."""
    return fixture_market["instruments"]


@pytest.fixture
def engine_config() -> Config:
    """Default pipeline configuration; the fixture seed 'T' never needs to exist in it."""
    return Config()


_NEUTRAL_STOP_LOSS_LEVEL: dict[tuple[AssetClass, str | None], float] = {
    (AssetClass.EQUITIES, None): -10.0,
    (AssetClass.COMMODITIES, None): -10.0,
    (AssetClass.RATES_CREDIT, "sovereign_rates"): 20.0,
    (AssetClass.RATES_CREDIT, "long_short_credit"): 30.0,
}


def _stop_loss_level(asset_class: AssetClass, sub_style: str) -> float:
    key = (asset_class, sub_style if asset_class == AssetClass.RATES_CREDIT else None)
    if key not in _NEUTRAL_STOP_LOSS_LEVEL:
        raise ValueError(f"no neutral stop-loss level defined for {asset_class.value}/{sub_style}")
    return _NEUTRAL_STOP_LOSS_LEVEL[key]


def _build_rule(
    pm_id: str, rule_id: str, source: RuleSource, param: str, variant, level: float
) -> Rule:
    template = variant.templates[0]
    return Rule(
        pm_id=pm_id,
        rule_id=rule_id,
        source=source,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param=param,
        field=variant.field,
        op=variant.op,
        level=level,
        unit=variant.unit,
        window=variant.window,
        action=variant.action,
        text=render_template(template, level, variant.unit),
    )


def _build_rules(
    pm_id: str, asset_class: AssetClass, sub_style: str, catalogue: Catalogue
) -> list[Rule]:
    entries_by_param = {entry.param: entry for entry in catalogue.rules.entries}
    rows: list[Rule] = []

    cap_entry = catalogue.rules.mandate_cap
    cap_variant = cap_entry.variant_for(asset_class, sub_style)
    rows.append(_build_rule(pm_id, "r_01", RuleSource.MANDATE, cap_entry.param, cap_variant, 10.0))

    stop_loss = entries_by_param["stop_loss"]
    stop_variant = stop_loss.variant_for(asset_class, sub_style)
    rows.append(
        _build_rule(
            pm_id,
            "r_02",
            RuleSource.SELF,
            stop_loss.param,
            stop_variant,
            _stop_loss_level(asset_class, sub_style),
        )
    )

    trim = entries_by_param["trim_at_target"]
    trim_variant = trim.variant_for(asset_class, sub_style)
    trim_level = trim_variant.level_choices[0]
    rows.append(_build_rule(pm_id, "r_03", RuleSource.SELF, trim.param, trim_variant, trim_level))

    no_add = entries_by_param["no_add_before_trigger"]
    no_add_variant = no_add.variant_for(asset_class, sub_style)
    no_add_level = no_add_variant.level_choices[0]
    rows.append(
        _build_rule(pm_id, "r_04", RuleSource.SELF, no_add.param, no_add_variant, no_add_level)
    )

    min_holding = entries_by_param["min_holding_period"]
    min_holding_variant = min_holding.variant_for(asset_class, sub_style)
    rows.append(
        _build_rule(pm_id, "r_05", RuleSource.SELF, min_holding.param, min_holding_variant, 5.0)
    )

    max_positions = entries_by_param["max_positions"]
    max_positions_variant = max_positions.variant_for(asset_class, sub_style)
    rows.append(
        _build_rule(
            pm_id, "r_06", RuleSource.SELF, max_positions.param, max_positions_variant, 20.0
        )
    )

    if asset_class == AssetClass.COMMODITIES:
        roll = entries_by_param["roll_before_expiry"]
        roll_variant = roll.variant_for(asset_class, sub_style)
        rows.append(_build_rule(pm_id, "r_07", RuleSource.SELF, roll.param, roll_variant, 5.0))

    return rows


@pytest.fixture
def neutral_pm(catalogue: Catalogue):
    """Factory: a neutral PM persona, its eight bias traits at their neutral medians, and rules."""

    def _build(asset_class: AssetClass, sub_style: str) -> tuple[Persona, list[Trait], list[Rule]]:
        pm_id = "pm_001"
        persona = Persona(
            pm_id=pm_id,
            market_seed=_SEED,
            split=Split.PILOT,
            mandate=Mandate(
                asset_class=asset_class,
                sub_style=sub_style,
                book_size=1e8,
                risk_unit="pct_nav",
                benchmark="cash",
            ),
            stated_profile=StatedProfile(self_description="I run a disciplined, rules-based book."),
            typicality=Typicality.TYPICAL,
        )

        config = Config()
        traits = [
            Trait(
                pm_id=pm_id,
                trait_id=f"t_{i:02d}",
                kind=Kind.BIAS,
                param=param,
                value=config.biases.params[param].neutral.median_value(),
                active=False,
                mult_range=1.0,
                mult_risk_off=1.0,
                mult_risk_on=1.0,
            )
            for i, param in enumerate(BIAS_PARAMS, start=1)
        ]

        rules = _build_rules(pm_id, asset_class, sub_style, catalogue)
        return persona, traits, rules

    return _build


# Shared by the engine and gate 1 stage tests: a small population of neutral
# PMs, one per asset class and sub-style, plus a multi-asset PM the engine
# stage skips.
NEUTRAL_PMS: list[tuple[AssetClass, str]] = [
    (AssetClass.EQUITIES, "equity_long_short"),
    (AssetClass.RATES_CREDIT, "sovereign_rates"),
    (AssetClass.RATES_CREDIT, "long_short_credit"),
    (AssetClass.COMMODITIES, "commodity_futures_directional"),
    (AssetClass.COMMODITIES, "curve_and_spread"),
]
MULTI_ASSET_PM_ID = "pm_006"


def stage_config_overrides() -> dict:
    """The overrides behind `stage_config`, for tests that write them to a YAML config file."""
    return {
        "calendar": {"start": "2026-01-05", "n_weeks": 12},
        "population": {
            "asset_classes": ["equities", "rates_credit", "commodities"],
            "market_seeds": ["T"],
            "pilot_market_seeds": ["T"],
            "pilot_per_cell": 1,
            "full_per_cell": 1,
        },
        "market": {
            "seeds": {"T": ["range", "risk_off", "risk_on"]},
            "real": {"seeds": {}},
            "boundary_weeks": [4, 8],
            "burn_in_days": 1,
        },
        "drift": {
            "bias_update_weeks": [2, 4],
            "preference_update_weeks": [2, 10],
            "dormant_weeks": [5, 7],
            "revive_weeks": [8, 9],
        },
        "engine": {"horizon_days": 10},
    }


def stage_config() -> Config:
    """A config whose published horizon exactly covers the 60-day 'T' fixture market."""
    return Config.model_validate(stage_config_overrides())


def _bad_field_rule(pm_id: str) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id="r_99",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="min_holding_period",
        field="not_a_real_field",
        op=Op.GE,
        level=5.0,
        unit=None,
        window=1,
        action=Action.HOLD,
        text="bogus rule",
    )


def write_stage_inputs(store, fixture_market, neutral_pm, *, bad_field: bool = False):
    """Write personas, traits, rules, drift events and the fixture market into `store`.

    One neutral PM per `NEUTRAL_PMS` entry plus a multi-asset PM the engine
    stage skips; `bad_field` appends a rule with an unknown field to the first
    PM, for the unknown-rule-field failure path.
    """
    personas = []
    all_traits = []
    all_rules = []
    for i, (asset_class, sub_style) in enumerate(NEUTRAL_PMS, start=1):
        persona, traits, rules = neutral_pm(asset_class, sub_style)
        pm_id = f"pm_{i:03d}"
        persona = persona.model_copy(update={"pm_id": pm_id})
        traits = [t.model_copy(update={"pm_id": pm_id}) for t in traits]
        rules = [r.model_copy(update={"pm_id": pm_id}) for r in rules]
        if bad_field and i == 1:
            rules.append(_bad_field_rule(pm_id))
        personas.append(persona)
        all_traits.extend(traits)
        all_rules.extend(rules)

    personas.append(
        Persona(
            pm_id=MULTI_ASSET_PM_ID,
            market_seed="T",
            split=Split.PILOT,
            mandate=Mandate(
                asset_class=AssetClass.MULTI_ASSET,
                sub_style="global_macro",
                book_size=1e8,
                risk_unit="pct_nav_sleeve",
                benchmark="cash",
            ),
            stated_profile=StatedProfile(self_description="I mix sleeves across asset classes."),
            typicality=Typicality.TYPICAL,
        )
    )

    store.write(PERSONAS, personas)
    store.write(TRAITS, all_traits)
    store.write(RULES, all_rules)
    store.write(DRIFT_EVENTS, [])
    store.write(MARKET_INSTRUMENTS, fixture_market["instruments"])
    store.write(MARKET_PRICES, fixture_market["prices"])
    store.write(MARKET_CURVES, fixture_market["curves"])
    store.write(MARKET_CONSENSUS, fixture_market["consensus"])
    store.write(MARKET_CALENDAR, fixture_market["calendar"])
    store.write(MARKET_REGIMES, fixture_market["regimes"])
    return personas, all_rules
