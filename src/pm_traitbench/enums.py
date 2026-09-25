"""Shared enumerations for config, row models and sampling.

Kept in one module with no other imports so any part of the pipeline can
depend on these without pulling in unrelated code.
"""

from enum import StrEnum


class AssetClass(StrEnum):
    EQUITIES = "equities"
    RATES_CREDIT = "rates_credit"
    COMMODITIES = "commodities"
    MULTI_ASSET = "multi_asset"


class Split(StrEnum):
    PILOT = "pilot"
    FULL = "full"


class Typicality(StrEnum):
    TYPICAL = "typical"
    ANTI_TYPICAL = "anti_typical"


class Kind(StrEnum):
    BIAS = "bias"
    PREFERENCE = "preference"


class RuleSource(StrEnum):
    MANDATE = "mandate"
    SELF = "self"


class RuleScope(StrEnum):
    PM = "pm"
    IDEA = "idea"


class Op(StrEnum):
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"
    EQ = "=="
    NE = "!="


class Action(StrEnum):
    EXIT = "exit"
    TRIM_HALF = "trim_half"
    NO_ADD = "no_add"
    EXCLUDE = "exclude"
    CAP = "cap"
    TARGET = "target"
    SIGNPOST = "signpost"
    HOLD = "hold"
    ROLL = "roll"


class DriftEventType(StrEnum):
    UPDATE = "update"
    DORMANT = "dormant"
    REVIVE = "revive"


class Regime(StrEnum):
    RANGE = "range"
    RISK_OFF = "risk_off"
    RISK_ON = "risk_on"


class Family(StrEnum):
    EQUITIES = "equities"
    RATES = "rates"
    CREDIT = "credit"
    COMMODITIES = "commodities"
    FX = "fx"


class InstrumentKind(StrEnum):
    EQUITY = "equity"
    CREDIT_ISSUER = "credit_issuer"
    SOVEREIGN_CURVE = "sovereign_curve"
    COMMODITY = "commodity"
    FX_PAIR = "fx_pair"


class EventType(StrEnum):
    EARNINGS = "earnings"
    RATING_DOWNGRADE = "rating_downgrade"
    RATING_UPGRADE = "rating_upgrade"
    CB_MEETING = "cb_meeting"
    INVENTORY_REPORT = "inventory_report"
    CROP_REPORT = "crop_report"
    MACRO_PRINT = "macro_print"
    CONTRACT_EXPIRY = "contract_expiry"
    POSITIONING_REPORT = "positioning_report"
    CONSENSUS_FLIP = "consensus_flip"


class StreetView(StrEnum):
    UNDERWEIGHT = "underweight"
    NEUTRAL = "neutral"
    OVERWEIGHT = "overweight"


class Positioning(StrEnum):
    CROWDED_SHORT = "crowded_short"
    NEUTRAL = "neutral"
    CROWDED_LONG = "crowded_long"


class Tenor(StrEnum):
    Y2 = "2Y"
    Y5 = "5Y"
    Y10 = "10Y"
    Y30 = "30Y"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    M5 = "M5"
    M6 = "M6"
    M7 = "M7"
    M8 = "M8"
    M9 = "M9"
    M10 = "M10"
    M11 = "M11"
    M12 = "M12"


class CommodityGroup(StrEnum):
    ENERGY = "energy"
    INDUSTRIAL_METALS = "industrial_metals"
    PRECIOUS = "precious"
    AGRICULTURE = "agriculture"


class RatingBand(StrEnum):
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"


class ExpiryRule(StrEnum):
    MONTHLY_THIRD_FRIDAY = "monthly_third_friday"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Expression(StrEnum):
    OUTRIGHT = "outright"
    PAIR = "pair"
    CURVE = "curve"
    CALENDAR_SPREAD = "calendar_spread"


class RuleResponse(StrEnum):
    ACTED = "acted"
    ACKED_NO_ACTION = "acked_no_action"
    ADDED = "added"
    OVERRIDDEN = "overridden"


class PositionAction(StrEnum):
    NONE = "none"
    HOLD = "hold"
    ADD = "add"
    CUT = "cut"
    TRIM = "trim"
    EXIT = "exit"
    ROLL = "roll"


class PnlState(StrEnum):
    GAIN = "gain"
    LOSS = "loss"
    FLAT = "flat"


class Gate1Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT = "insufficient"


class Gate1Test(StrEnum):
    """Which rule judged a gate 1 cell: one PM's own statistic, or the pooled population."""

    PER_PM = "per_pm"
    POPULATION = "population"


class Gate1Split(StrEnum):
    ALL = "all"
    REGIME_RANGE = "regime_range"
    REGIME_RISK_OFF = "regime_risk_off"
    REGIME_RISK_ON = "regime_risk_on"
    BEFORE = "before"
    AFTER = "after"


class SeedGroupKind(StrEnum):
    SYNTHETIC_POOL = "synthetic_pool"
    SYNTHETIC_SEED = "synthetic_seed"
    REAL_SEED = "real_seed"


SOVEREIGN_TENORS: tuple[Tenor, ...] = (Tenor.Y2, Tenor.Y5, Tenor.Y10, Tenor.Y30)
FUTURES_TENORS: tuple[Tenor, ...] = (
    Tenor.M1,
    Tenor.M2,
    Tenor.M3,
    Tenor.M4,
    Tenor.M5,
    Tenor.M6,
    Tenor.M7,
    Tenor.M8,
    Tenor.M9,
    Tenor.M10,
    Tenor.M11,
    Tenor.M12,
)
NULL_SURPRISE_EVENTS: frozenset[EventType] = frozenset(
    {EventType.CONTRACT_EXPIRY, EventType.POSITIONING_REPORT, EventType.CONSENSUS_FLIP}
)
MARKET_WIDE_EVENTS: frozenset[EventType] = frozenset(
    {EventType.MACRO_PRINT, EventType.POSITIONING_REPORT}
)
HY_BANDS: frozenset[RatingBand] = frozenset({RatingBand.BB, RatingBand.B})
IG_BANDS: frozenset[RatingBand] = frozenset(RatingBand) - HY_BANDS
MULTI_LEG_FORMS: frozenset[Expression] = frozenset(
    {Expression.PAIR, Expression.CURVE, Expression.CALENDAR_SPREAD}
)
