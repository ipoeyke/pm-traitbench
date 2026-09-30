"""Shared enumerations for config, row models and sampling.

Kept in one module with no other imports so any part of the pipeline can
depend on these without pulling in unrelated code.
"""

from enum import StrEnum

# Sampling: the PM population and its traits


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


# Rules: the condition grammar and what firing one does


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


# Drift


class DriftEventType(StrEnum):
    UPDATE = "update"
    DORMANT = "dormant"
    REVIVE = "revive"


# Market: universe, regimes, events and consensus


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


# Engine: trades, ideas and position days


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


# Gate 1


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


# Gate 2


class Gate2Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT = "insufficient"


class Gate2Slice(StrEnum):
    """A cell's slicing dimension: the whole population, or one grouping within it."""

    ALL = "all"
    HELD = "held"
    KIND = "kind"
    MODE = "mode"
    ASSET_CLASS = "asset_class"
    TYPICALITY = "typicality"
    DRIFT = "drift"


# Derived groupings


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


# Signal plan: signals, sessions and stances


class SignalMode(StrEnum):
    STATED = "stated"
    REVEALED = "revealed"
    CONTRADICTION = "contradiction"


# The modes in which a trait shows through behaviour rather than being stated
REVEALING_MODES: frozenset[SignalMode] = frozenset({SignalMode.REVEALED, SignalMode.CONTRADICTION})


class Valence(StrEnum):
    CONFIRM = "confirm"
    RETRACTED = "retracted"


class Ownership(StrEnum):
    SELF = "self"
    COLLEAGUE = "colleague"
    CLIENT = "client"


class SessionKind(StrEnum):
    DECISION = "decision"
    CHECK_IN = "check_in"
    SILENCE = "silence"


class StanceEntry(StrEnum):
    REVEALED = "revealed"
    STATED = "stated"
    CLAIM = "claim"
    RETRACT = "retract"
    THIRD_PARTY = "third_party"
    DRIFT_UPDATE = "drift_update"
    DRIFT_DORMANT = "drift_dormant"
    DRIFT_REVIVE = "drift_revive"
    REVEALED_REACTION = "revealed_reaction"
    VIOLATION = "violation"


class CarrierSource(StrEnum):
    LEDGER = "ledger"
    POSITION_DAY = "position_day"
    RULE_EVENT = "rule_event"
    IDEA = "idea"


# Dialogue


class TurnRole(StrEnum):
    PM = "pm"
    ADVISOR = "advisor"


class MentionKind(StrEnum):
    TRADE = "trade"
    LEVEL = "level"


# Validate


class ValidationStatus(StrEnum):
    PASS = "pass"
    REGENERATE = "regenerate"
    DROPPED = "dropped"


# Cross-stage helpers and API settings


class DriftStatus(StrEnum):
    STATIC = "static"
    DRIFT = "drift"


class Effort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class AdvisorTool(StrEnum):
    GET_QUOTE = "get_quote"
    GET_CURVE = "get_curve"
    GET_CONSENSUS = "get_consensus"
    GET_CALENDAR = "get_calendar"
    GET_HISTORY = "get_history"


class ProbeType(StrEnum):
    TRAIT_PRESENCE = "trait_presence"
    TRAIT_MCQ = "trait_mcq"
    IN_SITU = "in_situ"
    ROUTINE_QUESTION = "routine_question"
    GOVERNANCE = "governance"


class ProbeForm(StrEnum):
    MCQ = "mcq"
    OPEN = "open"


class CheckpointLabel(StrEnum):
    WEEK4 = "week4"
    WEEK13 = "week13"
    PRE_DRIFT = "pre_drift"
    POST_DRIFT = "post_drift"
    REGIME_SHIFT = "regime_shift"
    WEEK52 = "week52"


class OptionSource(StrEnum):
    CURRENT = "current"
    PRE_UPDATE = "pre_update"
    STATED_PROFILE = "stated_profile"
    THIRD_PARTY = "third_party"
    NONE = "none"


class McqAction(StrEnum):
    """An engine outcome a bias MCQ option stands for; an outcome shared by biases is one member."""

    ADD = "add"
    HOLD = "hold"
    CUT = "cut"
    TRIM_HALF = "trim_half"
    SELL_NOW = "sell_now"
    HOLD_TO_TARGET = "hold_to_target"
    EXIT_AT_ROUND_LEVEL = "exit_at_round_level"
    LEAVE_ON = "leave_on"
    EXIT_PER_STOP = "exit_per_stop"
    FOLLOW_STREET = "follow_street"
    OWN_READ = "own_read"
    STAND_ASIDE = "stand_aside"
    HEDGE = "hedge"
    CHASE_RUN = "chase_run"
    SELL_ON_THESIS = "sell_on_thesis"
    SIZE_DOUBLE = "size_double"
    SIZE_ONE_AND_HALF = "size_one_and_half"
    SIZE_STANDARD = "size_standard"
    SIZE_HALF = "size_half"
    SIZE_OFF_RATING = "size_off_rating"
    SIZE_TO_RATING = "size_to_rating"
    SIZE_FULL = "size_full"
    NO_POSITION = "no_position"


class ProbeSkip(StrEnum):
    NO_SUPPORT = "no_supporting_signal"
    NO_SITUATION = "no_situation"
    NO_HORIZON = "no_horizon"


class Scorer(StrEnum):
    OPTION_LETTER = "option_letter"
    FORMAT = "format"
    JUDGE_OPEN = "judge_open"
    JUDGE_IN_SITU = "judge_in_situ"
    JUDGE_GOVERNANCE = "judge_governance"
    JUDGE_INTRUSION = "judge_intrusion"
    JUDGE_FORMAT = "judge_format"


class Judge(StrEnum):
    OPEN = "judge_open"
    IN_SITU = "judge_in_situ"
    GOVERNANCE = "judge_governance"
    INTRUSION = "judge_intrusion"
    FORMAT = "judge_format"


class FormatOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class CheckKind(StrEnum):
    BULLETS = "bullets"
    PROSE_PARAGRAPH = "prose_paragraph"
    TABLE = "table"
    HEADERS = "headers"
    UNITS_BP = "units_bp"
    UNITS_PERCENT = "units_percent"
    UNITS_BOTH = "units_both"
    ONE_SENTENCE = "one_sentence"
    TWO_TO_THREE_SENTENCES = "two_to_three_sentences"
    SHORT_PAGE = "short_page"
    NO_HEDGES = "no_hedges"
    CONFIDENCE_LEVEL = "confidence_level"
    JUDGE = "judge"


class RunStatus(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"


class EvidenceType(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    MIXED = "mixed"
    NONE = "none"
