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
