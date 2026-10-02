"""Pipeline configuration: sections, binding defaults and YAML loading."""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pm_traitbench.distributions import BetaSpec, Distribution, LogNormalSpec
from pm_traitbench.enums import (
    HY_BANDS,
    AssetClass,
    CommodityGroup,
    DriftStatus,
    Effort,
    EventType,
    ExpiryRule,
    RatingBand,
    Regime,
    SessionKind,
    Split,
    Typicality,
)
from pm_traitbench.errors import ConfigError
from pm_traitbench.market.constants import (
    COMMODITIES,
    CREDIT_BAND_ORDER,
    FX_PAIRS,
    HORIZON_DAYS_PER_YEAR,
    USD_PAIR,
    largest_remainder,
)
from pm_traitbench.market.real.event_dates import EVENT_DATES_COVER
from pm_traitbench.timeline import Timeline

BIAS_PARAMS: tuple[str, ...] = (
    "loss_aversion_lambda",
    "disposition_ratio",
    "anchoring_rho",
    "extrapolation_theta",
    "herding_weight",
    "overconfidence_coverage",
    "conviction_size_miscalibration",
    "exit_deficiency",
)

Basis = Literal["sourced", "design", "guess"]

# One model for narrator, advisor and judge, so model behaviour never confounds recovery.
DEFAULT_MODEL = "claude-opus-5-5"


class BiasSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    neutral: Distribution
    active: Distribution
    higher_is_stronger: bool
    cluster_regime: Regime | None
    basis: Basis
    note: str = Field(min_length=1)


def _default_bias_params() -> dict[str, BiasSpec]:
    return {
        "loss_aversion_lambda": BiasSpec(
            neutral=LogNormalSpec(median=1.1, sigma=0.10),
            active=LogNormalSpec(median=2.0, sigma=0.25, lo=1.5),
            higher_is_stronger=True,
            cluster_regime=Regime.RISK_OFF,
            basis="sourced",
            note=(
                "Brown et al. 2024 meta-analysis of 607 estimates, mean 1.955; Tversky "
                "and Kahneman 1992 report 2.25. Spread is a guess."
            ),
        ),
        "disposition_ratio": BiasSpec(
            neutral=LogNormalSpec(median=1.0, sigma=0.08),
            active=LogNormalSpec(median=1.2, sigma=0.15, lo=1.2),
            higher_is_stronger=True,
            cluster_regime=Regime.RISK_OFF,
            basis="sourced",
            note=(
                "Odean 1998 retail ratio 1.51; Frazzini 2006 and Locke and Mann 2005 "
                "give a professional centre near 1.2."
            ),
        ),
        "anchoring_rho": BiasSpec(
            neutral=BetaSpec(a=2, b=12),
            active=BetaSpec(a=9, b=12),
            higher_is_stronger=True,
            cluster_regime=Regime.RANGE,
            basis="sourced",
            note=(
                "Northcraft and Neale 1987 valuation-anchor correlation 0.41 for "
                "experts; Yee and Koh 2026 benchmark 0.43."
            ),
        ),
        "extrapolation_theta": BiasSpec(
            neutral=BetaSpec(a=2, b=10),
            active=BetaSpec(a=12, b=8),
            higher_is_stronger=True,
            cluster_regime=Regime.RISK_ON,
            basis="sourced",
            note=(
                "Bloomfield and Hales 2002 forecast-past return correlation 0.63; "
                "Greenwood and Shleifer 2014 report 0.57."
            ),
        ),
        "herding_weight": BiasSpec(
            neutral=BetaSpec(a=2, b=10),
            active=BetaSpec(a=7, b=5),
            higher_is_stronger=True,
            cluster_regime=Regime.RISK_ON,
            basis="sourced",
            note=(
                "Anderson and Holt 1997 follow-the-crowd rate 0.68 in cascades; "
                "professional centre lowered to 0.58 as a guess."
            ),
        ),
        "overconfidence_coverage": BiasSpec(
            neutral=BetaSpec(a=16, b=4),
            active=BetaSpec(a=4, b=6),
            higher_is_stronger=False,
            cluster_regime=None,
            basis="sourced",
            note=(
                "Ben-David, Graham and Harvey 2013: CFO 80 percent intervals "
                "contained the outcome 36 percent of the time."
            ),
        ),
        "conviction_size_miscalibration": BiasSpec(
            neutral=BetaSpec(a=2, b=10),
            active=BetaSpec(a=5, b=5),
            higher_is_stronger=True,
            cluster_regime=None,
            basis="guess",
            note=(
                "No source for magnitude; motivated by Cohen, Polk and Silli 2010 on best ideas."
            ),
        ),
        "exit_deficiency": BiasSpec(
            neutral=BetaSpec(a=1, b=15),
            active=BetaSpec(a=4, b=5),
            higher_is_stronger=True,
            cluster_regime=None,
            basis="guess",
            note=(
                "Direction from Akepanidtaworn et al. 2023 on weak selling "
                "decisions; magnitude is a guess."
            ),
        ),
    }


def _default_correlation() -> tuple[tuple[float, ...], ...]:
    n = len(BIAS_PARAMS)
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    index = {name: i for i, name in enumerate(BIAS_PARAMS)}
    pairs = {
        ("loss_aversion_lambda", "disposition_ratio"): 0.42,
        ("overconfidence_coverage", "herding_weight"): -0.31,
        ("loss_aversion_lambda", "herding_weight"): -0.08,
        ("extrapolation_theta", "herding_weight"): 0.05,
        ("exit_deficiency", "disposition_ratio"): 0.30,
        # Product of the two pairs above: exit deficiency relates to loss aversion
        # only through disposition, so their partial correlation is zero.
        ("exit_deficiency", "loss_aversion_lambda"): 0.126,
    }
    for (name_a, name_b), value in pairs.items():
        i, j = index[name_a], index[name_b]
        matrix[i][j] = value
        matrix[j][i] = value
    return tuple(tuple(row) for row in matrix)


class SeedConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root: int = Field(20260105, json_schema_extra={"basis": "design", "note": "root RNG seed"})


class PopulationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_classes: tuple[AssetClass, ...] = Field(
        default=tuple(AssetClass),
        json_schema_extra={"basis": "design", "note": "cover every asset class"},
    )
    market_seeds: tuple[str, ...] = Field(
        default=("A", "B", "C"),
        json_schema_extra={"basis": "design", "note": "three parallel market seeds per cell"},
    )
    pilot_market_seeds: tuple[str, ...] = Field(
        default=("R1",),
        json_schema_extra={
            "basis": "design",
            "note": (
                "the pilot runs on a real historical market so it can finish before "
                "the synthetic full split"
            ),
        },
    )
    pilot_per_cell: int = Field(
        2, ge=0, json_schema_extra={"basis": "design", "note": "small pilot batch per cell"}
    )
    full_per_cell: int = Field(
        3, ge=0, json_schema_extra={"basis": "design", "note": "full batch size per cell"}
    )

    @model_validator(mode="after")
    def _check_grid(self) -> "PopulationConfig":
        # A repeated entry would silently double every cell it belongs to.
        for name, values in (
            ("asset_classes", self.asset_classes),
            ("market_seeds", self.market_seeds),
            ("pilot_market_seeds", self.pilot_market_seeds),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not repeat an entry: {list(values)}")
        for name, seeds in (
            ("market_seeds", self.market_seeds),
            ("pilot_market_seeds", self.pilot_market_seeds),
        ):
            if any(not seed.strip() for seed in seeds):
                raise ValueError(f"{name} must not contain a blank name")
        if self.pilot_per_cell == 0 and self.full_per_cell == 0:
            raise ValueError("pilot_per_cell and full_per_cell cannot both be 0")
        return self


class BiasesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    p_active: float = Field(
        0.35,
        ge=0,
        le=1,
        json_schema_extra={"basis": "guess", "note": "share of PMs with an active bias"},
    )
    min_active: int = Field(
        2,
        ge=0,
        le=8,
        json_schema_extra={"basis": "design", "note": "minimum active biases per activated PM"},
    )
    max_activation_attempts: int = Field(
        1000,
        gt=0,
        json_schema_extra={"basis": "design", "note": "resampling cap for activation draws"},
    )
    p_regime_cluster: float = Field(
        0.5,
        ge=0,
        le=1,
        json_schema_extra={"basis": "guess", "note": "chance a regime-clustered bias is boosted"},
    )
    regime_multiplier: Distribution = Field(
        default_factory=lambda: LogNormalSpec(median=1.15, sigma=0.10, lo=1.0, hi=1.5),
        json_schema_extra={
            "basis": "sourced",
            "note": "Guiso, Sapienza and Zingales 2018 risk aversion shift after crisis periods.",
        },
    )
    params: dict[str, BiasSpec] = Field(default_factory=_default_bias_params)
    correlation: tuple[tuple[float, ...], ...] = Field(
        default_factory=_default_correlation,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "Yee and Koh 2026 bias correlation structure; the two exit deficiency "
                "pairs are guesses."
            ),
        },
    )

    @field_validator("params")
    @classmethod
    def _check_params_keys(cls, value: dict[str, BiasSpec]) -> dict[str, BiasSpec]:
        if set(value) != set(BIAS_PARAMS):
            raise ValueError(f"params keys must be exactly {BIAS_PARAMS}")
        # Rebuild in BIAS_PARAMS order: correlation is indexed positionally by it.
        return {name: value[name] for name in BIAS_PARAMS}

    @field_validator("correlation")
    @classmethod
    def _check_correlation(
        cls, value: tuple[tuple[float, ...], ...]
    ) -> tuple[tuple[float, ...], ...]:
        n = len(BIAS_PARAMS)
        matrix = np.array(value, dtype=float)
        if matrix.shape != (n, n):
            raise ValueError(f"correlation must be {n}x{n}")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(matrix), 1.0):
            raise ValueError("correlation diagonal must be 1")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as e:
            raise ValueError("correlation must be positive-definite") from e
        return value


class MandateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    book_size_min: float = Field(
        50e6, gt=0, json_schema_extra={"basis": "guess", "note": "smallest plausible book size"}
    )
    book_size_max: float = Field(
        2e9, gt=0, json_schema_extra={"basis": "guess", "note": "largest plausible book size"}
    )

    @model_validator(mode="after")
    def _check_range(self) -> "MandateConfig":
        if self.book_size_min > self.book_size_max:
            raise ValueError("book_size_min must not be greater than book_size_max")
        return self


class RulesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_risk_pct_choices: tuple[float, ...] = Field(
        (5.0, 8.0, 10.0, 15.0),
        json_schema_extra={"basis": "guess", "note": "plausible max-risk mandate percentages"},
    )
    n_self_rules_min: int = Field(
        3, ge=0, json_schema_extra={"basis": "guess", "note": "minimum self rules per PM"}
    )
    n_self_rules_max: int = Field(
        5, ge=0, json_schema_extra={"basis": "guess", "note": "maximum self rules per PM"}
    )

    @model_validator(mode="after")
    def _check_range(self) -> "RulesConfig":
        if self.n_self_rules_min > self.n_self_rules_max:
            raise ValueError("n_self_rules_min must not be greater than n_self_rules_max")
        return self


class PreferencesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_min: int = Field(
        4, ge=0, json_schema_extra={"basis": "guess", "note": "minimum preferences per PM"}
    )
    n_max: int = Field(
        8, ge=0, json_schema_extra={"basis": "guess", "note": "maximum preferences per PM"}
    )

    @model_validator(mode="after")
    def _check_range(self) -> "PreferencesConfig":
        if self.n_min > self.n_max:
            raise ValueError("n_min must not be greater than n_max")
        return self


class DriftConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bias_update_weeks: tuple[int, int] = Field(
        (18, 30), json_schema_extra={"basis": "design", "note": "weeks bias updates may occur"}
    )
    bias_update_remaining: tuple[float, float] = Field(
        (0.5, 0.75),
        json_schema_extra={
            "basis": "sourced",
            "note": "Feng and Seasholes 2005 disposition effect decay in remaining position size.",
        },
    )
    p_preference_update: float = Field(
        0.5,
        ge=0,
        le=1,
        json_schema_extra={"basis": "guess", "note": "chance a preference updates in a week"},
    )
    preference_update_weeks: tuple[int, int] = Field(
        (8, 45),
        json_schema_extra={"basis": "design", "note": "weeks preference updates may occur"},
    )
    p_dormant_revive: float = Field(
        0.5,
        ge=0,
        le=1,
        json_schema_extra={"basis": "guess", "note": "chance a dormant bias revives"},
    )
    dormant_weeks: tuple[int, int] = Field(
        (32, 40), json_schema_extra={"basis": "design", "note": "weeks a bias may go dormant"}
    )
    revive_weeks: tuple[int, int] = Field(
        (42, 48), json_schema_extra={"basis": "design", "note": "weeks a dormant bias may revive"}
    )

    @field_validator(
        "bias_update_weeks", "preference_update_weeks", "dormant_weeks", "revive_weeks"
    )
    @classmethod
    def _check_week_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        first, last = value
        if first > last:
            raise ValueError("week range must have first <= last")
        return value

    @model_validator(mode="after")
    def _check_revive_follows_dormant(self) -> "DriftConfig":
        # A revive drawn from an overlapping range could predate its dormant event.
        if self.revive_weeks[0] <= self.dormant_weeks[1]:
            raise ValueError("revive_weeks must start after dormant_weeks ends")
        return self

    @field_validator("bias_update_remaining")
    @classmethod
    def _check_remaining_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        first, last = value
        if not (0.0 <= first <= last <= 1.0):
            raise ValueError("bias_update_remaining must be within [0, 1] with first <= last")
        return value


class CalendarConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: date = Field(
        date(2026, 1, 5), json_schema_extra={"basis": "design", "note": "simulation start Monday"}
    )
    n_weeks: int = Field(
        52, ge=1, json_schema_extra={"basis": "design", "note": "simulation horizon in weeks"}
    )

    @model_validator(mode="after")
    def _check_timeline(self) -> "CalendarConfig":
        Timeline(self.start, self.n_weeks)
        return self


class EngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_days: int = Field(
        20,
        ge=5,
        json_schema_extra={
            "basis": "guess",
            "note": "forward window of the own signal, trailing window for vol and extrapolation",
        },
    )
    skill: float = Field(
        0.15,
        ge=0,
        lt=1,
        json_schema_extra={
            "basis": "guess",
            "note": "correlation of the own signal with the realised move; a small positive edge",
        },
    )
    arrival_rate: float = Field(
        0.6,
        gt=0,
        json_schema_extra={
            "basis": "guess",
            "note": "candidate attempts per day; sets ideas per PM-year (about 48 at default)",
        },
    )
    rr_range: tuple[float, float] = Field(
        (1.5, 3.0),
        json_schema_extra={
            "basis": "guess",
            "note": "target distance as a multiple of stop distance",
        },
    )
    signpost_k: float = Field(
        1.0,
        gt=0,
        json_schema_extra={
            "basis": "guess",
            "note": "level and relative signposts sit at k times the horizon vol from entry",
        },
    )
    preferred_form_weight: float = Field(
        0.7,
        gt=0,
        le=1,
        json_schema_extra={
            "basis": "guess",
            "note": "share of ideas in the preferred expression when a mapped preference is held",
        },
    )
    base_hazard: float = Field(
        0.03,
        gt=0,
        le=1,
        json_schema_extra={
            "basis": "guess",
            "note": "daily discretionary sell hazard at zero progress, about one in 33 days",
        },
    )

    @model_validator(mode="after")
    def _check_rr_range(self) -> "EngineConfig":
        lo, hi = self.rr_range
        if lo > hi:
            raise ValueError("rr_range must be ascending")
        if lo < 1:
            raise ValueError("rr_range[0] must be at least 1")
        return self


class Gate1Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    floor_se: float = Field(
        2.0,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": "neutral standard deviations between the neutral mean and the active floor",
        },
    )
    gap_fraction: float = Field(
        0.5,
        gt=0,
        le=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "neutral sd at most this share of the active-neutral gap puts the active "
                "mean past the floor"
            ),
        },
    )
    min_rank_corr: float = Field(
        0.4,
        ge=0,
        lt=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "about two null standard errors of a Spearman correlation over about 30 active PMs"
            ),
        },
    )
    min_pms: int = Field(
        5,
        ge=3,
        json_schema_extra={
            "basis": "design",
            "note": "smallest neutral or active set with a usable standard deviation",
        },
    )
    min_pop_z: float = Field(
        3.0,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": "about a 0.1% one-sided false pass per test",
        },
    )
    population_params: tuple[str, ...] = Field(
        (
            "herding_weight",
            "conviction_size_miscalibration",
            "disposition_ratio",
            "anchoring_rho",
            "loss_aversion_lambda",
        ),
        json_schema_extra={
            "basis": "design",
            "note": (
                "parameters limited by how many decisions one PM makes a year. "
                "loss_aversion_lambda's active rank correlation has median 0.53 over 9 "
                "roots but falls below 0.4 on 2, since a no-add rule leaves most PMs few "
                "add-allowed days; its pooled z is at least 4.9 on all 9"
            ),
        },
    )
    report_only_params: tuple[str, ...] = Field(
        ("herding_weight", "disposition_ratio", "anchoring_rho"),
        json_schema_extra={
            "basis": "design",
            "note": (
                "disposition_ratio's pooled population z is about 1 (median 1.0 over 12 "
                "roots) at the sourced 1.2 centre, and anchoring_rho's pooled z has median "
                "3.9 but falls below 3 on 2 of 12 roots. herding_weight passes pooled on "
                "all 12 roots but stays report-only because the trend-built street view "
                "couples it to extrapolation, so a pass does not isolate it"
            ),
        },
    )

    @field_validator("population_params", "report_only_params")
    @classmethod
    def _check_param_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(value) - set(BIAS_PARAMS)
        if unknown:
            raise ValueError(f"unknown bias parameter(s): {sorted(unknown)}")
        if len(set(value)) != len(value):
            raise ValueError(f"must not repeat an entry: {list(value)}")
        return value


class PlanConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bias_signals_min: int = Field(
        8,
        ge=1,
        json_schema_extra={
            "basis": "guess",
            "note": (
                "fewest planted carriers per active bias that a full-context "
                "reader can still pick up"
            ),
        },
    )
    bias_signals_max: int = Field(
        10,
        ge=1,
        json_schema_extra={"basis": "guess", "note": "upper end of the per-bias carrier range"},
    )
    pref_signals: int = Field(
        3,
        ge=1,
        json_schema_extra={
            "basis": "guess",
            "note": "a preference is stated or shown about three times a year",
        },
    )
    bias_revealed_weight: float = Field(
        0.65,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": "most bias evidence is visible only in decisions",
        },
    )
    bias_stated_weight: float = Field(
        0.175,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": "a minority of bias evidence is the PM describing it",
        },
    )
    bias_contradiction_weight: float = Field(
        0.10,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": "about one in ten bias signals sets a stated view against a later act",
        },
    )
    pref_stated_weight: float = Field(
        0.65,
        gt=0,
        json_schema_extra={"basis": "guess", "note": "preferences are mostly said outright"},
    )
    pref_revealed_weight: float = Field(
        0.35,
        gt=0,
        json_schema_extra={
            "basis": "guess",
            "note": "the rest show as a reaction or an instrument choice",
        },
    )
    retracted_share: float = Field(
        0.075,
        ge=0,
        lt=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "a few statements are taken back in the same session and must not count as evidence"
            ),
        },
    )
    third_party_share: float = Field(
        0.10,
        ge=0,
        lt=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "about one signal in ten belongs to a colleague or client, as an "
                "ownership distractor"
            ),
        },
    )
    claim_lead_days: tuple[int, int] = Field(
        (10, 40),
        json_schema_extra={
            "basis": "guess",
            "note": (
                "the earlier claim is its own session yet inside a quarter of "
                "the act it contradicts"
            ),
        },
    )
    ledger_session_percentile: float = Field(
        75.0,
        gt=0,
        le=100,
        json_schema_extra={
            "basis": "guess",
            "note": (
                "orders at or above the PM's own upper quartile of risk are "
                "large enough that the PM mentions them"
            ),
        },
    )
    signal_session_cap: float = Field(
        0.40,
        gt=0,
        le=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                'most sessions carry no planted signal, so "links nothing" '
                "stays a common correct outcome"
            ),
        },
    )
    filler_silence_share: float = Field(
        0.5,
        ge=0,
        le=1,
        json_schema_extra={"basis": "guess", "note": "half of filler is a pure market question"},
    )
    drift_min_per_side: int = Field(
        6,
        ge=0,
        json_schema_extra={
            "basis": "design",
            "note": (
                "enough evidence on each side of a drift event to tell the old value from the new"
            ),
        },
    )
    max_signals_per_session: int = Field(
        2,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "two stances per session keeps planted signals from crowding one short "
                "exchange while holding signal-carrying sessions near the share cap"
            ),
        },
    )

    @field_validator("claim_lead_days")
    @classmethod
    def _check_claim_lead_days(cls, value: tuple[int, int]) -> tuple[int, int]:
        first, last = value
        if first < 1:
            raise ValueError("claim_lead_days[0] must be at least 1")
        if first > last:
            raise ValueError("claim_lead_days must have first <= second")
        return value

    @model_validator(mode="after")
    def _check_bias_signals_range(self) -> "PlanConfig":
        if self.bias_signals_min > self.bias_signals_max:
            raise ValueError("bias_signals_min must not be greater than bias_signals_max")
        return self


class OutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["jsonl", "parquet"] = Field(
        "jsonl", json_schema_extra={"basis": "design", "note": "default output format"}
    )
    tables: dict[str, Literal["jsonl", "parquet"]] = Field(
        default_factory=dict,
        json_schema_extra={"basis": "design", "note": "per-table format overrides"},
    )


@dataclass(frozen=True)
class RegimeParams:
    driver_mean: float
    vol_multiplier: float
    mean_reversion_kappa: float


class MarketUniverseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_equities: int = Field(
        80,
        gt=0,
        json_schema_extra={"basis": "design", "note": "size of the simulated equity universe"},
    )
    n_sectors: int = Field(
        10,
        ge=1,
        le=99,
        json_schema_extra={
            "basis": "design",
            "note": "number of equity sectors to spread names across",
        },
    )
    n_credit_issuers: int = Field(
        48,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": "size of the simulated credit issuer universe",
        },
    )
    credit_band_shares: dict[RatingBand, float] = Field(
        default_factory=lambda: {
            RatingBand.AA: 0.15,
            RatingBand.A: 0.25,
            RatingBand.BBB: 0.30,
            RatingBand.BB: 0.20,
            RatingBand.B: 0.10,
        },
        json_schema_extra={
            "basis": "guess",
            "note": (
                "an investment-grade-heavy mix in the spirit of the ICE BofA US Corporate "
                "and High Yield indices, with the high-yield share raised; the index fact "
                "sheets would verify it."
            ),
        },
    )
    curves: tuple[str, ...] = Field(
        ("USD", "EUR", "GBP", "JPY"),
        json_schema_extra={"basis": "design", "note": "sovereign curves in the simulated universe"},
    )
    commodities: dict[CommodityGroup, int] = Field(
        default_factory=lambda: {
            CommodityGroup.ENERGY: 6,
            CommodityGroup.INDUSTRIAL_METALS: 4,
            CommodityGroup.PRECIOUS: 3,
            CommodityGroup.AGRICULTURE: 7,
        },
        json_schema_extra={
            "basis": "design",
            "note": "how many commodities of each group's code table are simulated",
        },
    )
    fx_pairs: tuple[str, ...] = Field(
        (
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCHF",
            "USDCAD",
            "EURGBP",
            "EURJPY",
            "AUDJPY",
        ),
        json_schema_extra={"basis": "design", "note": "FX pairs in the simulated universe"},
    )
    equity_beta_range: tuple[float, float] = Field(
        (0.6, 1.4),
        json_schema_extra={
            "basis": "guess",
            "note": "about the 10th to 90th percentile of large-cap equity betas",
        },
    )
    credit_duration_range: tuple[float, float] = Field(
        (3.0, 8.0),
        json_schema_extra={
            "basis": "guess",
            "note": (
                "high-yield index duration is about 4 and investment-grade about 7; the "
                "index fact sheets would verify it."
            ),
        },
    )
    expiry_rule: ExpiryRule = Field(
        ExpiryRule.MONTHLY_THIRD_FRIDAY,
        json_schema_extra={"basis": "design", "note": "futures and options expiry convention"},
    )

    @field_validator("credit_band_shares")
    @classmethod
    def _check_credit_band_shares(cls, value: dict[RatingBand, float]) -> dict[RatingBand, float]:
        if set(value) != set(RatingBand):
            raise ValueError(f"credit_band_shares keys must be exactly {set(RatingBand)}")
        if any(share < 0 for share in value.values()):
            raise ValueError("credit_band_shares must be non-negative")
        if abs(sum(value.values()) - 1.0) > 1e-9:
            raise ValueError("credit_band_shares must sum to 1")
        ig_bands = set(RatingBand) - HY_BANDS
        if not any(value[band] > 0 for band in ig_bands):
            raise ValueError("credit_band_shares must have a positive investment-grade share")
        return value

    @field_validator("curves")
    @classmethod
    def _check_curves(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("curves must not be empty")
        if len(set(value)) != len(value):
            raise ValueError(f"curves must not repeat an entry: {list(value)}")
        return value

    @field_validator("commodities")
    @classmethod
    def _check_commodities(cls, value: dict[CommodityGroup, int]) -> dict[CommodityGroup, int]:
        if set(value) != set(CommodityGroup):
            raise ValueError(f"commodities keys must be exactly {set(CommodityGroup)}")
        for group, n in value.items():
            max_n = len(COMMODITIES[group])
            if not (0 <= n <= max_n):
                raise ValueError(f"commodities[{group}] must be between 0 and {max_n}")
        if sum(value.values()) == 0:
            raise ValueError("commodities must include at least one commodity")
        return value

    @field_validator("fx_pairs")
    @classmethod
    def _check_fx_pairs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("fx_pairs must not be empty")
        unknown = set(value) - set(FX_PAIRS)
        if unknown:
            raise ValueError(f"fx_pairs has unknown pair(s): {sorted(unknown)}")
        if len(set(value)) != len(value):
            raise ValueError(f"fx_pairs must not repeat an entry: {list(value)}")
        if not set(value) & set(USD_PAIR.values()):
            raise ValueError("fx_pairs must include at least one USD pair")
        return value

    @field_validator("equity_beta_range", "credit_duration_range")
    @classmethod
    def _check_positive_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = value
        if not (0 < lo <= hi):
            raise ValueError("range must have 0 < lo <= hi")
        return value

    @model_validator(mode="after")
    def _check_credit_band_allocation(self) -> "MarketUniverseConfig":
        counts = largest_remainder(
            self.n_credit_issuers, self.credit_band_shares, CREDIT_BAND_ORDER
        )
        ig_bands = set(RatingBand) - HY_BANDS
        if sum(counts[band] for band in ig_bands) == 0:
            raise ValueError(
                "n_credit_issuers and credit_band_shares leave zero investment-grade issuers"
            )
        return self


class MarketRegimesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    driver_mean: dict[Regime, float] = Field(
        default_factory=lambda: {
            Regime.RANGE: 0.00,
            Regime.RISK_OFF: -0.09,
            Regime.RISK_ON: 0.06,
        },
        json_schema_extra={"basis": "design", "note": "mean daily driver level by regime"},
    )
    vol_multiplier: dict[Regime, float] = Field(
        default_factory=lambda: {
            Regime.RANGE: 1.0,
            Regime.RISK_OFF: 1.6,
            Regime.RISK_ON: 0.95,
        },
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "VIX medians of 17.0, 27.6 and 16.3 for range, risk-off and risk-on, "
                "conditioned on the trailing 60-day equity return, FRED VIXCLS and "
                "NASDAQCOM."
            ),
        },
    )
    mean_reversion_kappa: dict[Regime, float] = Field(
        default_factory=lambda: {
            Regime.RANGE: 0.05,
            Regime.RISK_OFF: 0.0,
            Regime.RISK_ON: 0.0,
        },
        json_schema_extra={
            "basis": "design",
            "note": "range regime pulls toward its mean with a half-life of about 14 days",
        },
    )

    @field_validator("driver_mean", "vol_multiplier", "mean_reversion_kappa")
    @classmethod
    def _check_regime_keys(cls, value: dict[Regime, float]) -> dict[Regime, float]:
        if set(value) != set(Regime):
            raise ValueError(f"must have exactly the three regimes: {set(Regime)}")
        return value

    @field_validator("vol_multiplier")
    @classmethod
    def _check_vol_multiplier_positive(cls, value: dict[Regime, float]) -> dict[Regime, float]:
        if any(v <= 0 for v in value.values()):
            raise ValueError("vol_multiplier must be positive")
        return value

    @field_validator("mean_reversion_kappa")
    @classmethod
    def _check_kappa_range(cls, value: dict[Regime, float]) -> dict[Regime, float]:
        if any(not (0 <= v < 1) for v in value.values()):
            raise ValueError("mean_reversion_kappa must be in [0, 1)")
        return value

    def params(self, regime: Regime) -> RegimeParams:
        return RegimeParams(
            driver_mean=self.driver_mean[regime],
            vol_multiplier=self.vol_multiplier[regime],
            mean_reversion_kappa=self.mean_reversion_kappa[regime],
        )


class EquityFamilyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    market_vol: float = Field(
        0.16,
        gt=0,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "S&P 500 realised volatility 2016-2026 is 18.1 percent, and 15.2 percent "
                "excluding 2020; the VIX median 1990-2026 is 17.6, FRED SP500."
            ),
        },
    )
    idio_vol: float = Field(
        0.25,
        gt=0,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "Campbell, Lettau, Malkiel and Xu 2023 estimate of average idiosyncratic "
                "equity volatility."
            ),
        },
    )


class RatesFamilyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level_vol_bp: dict[str, float] = Field(
        default_factory=lambda: {"USD": 90.0, "GBP": 90.0, "EUR": 70.0, "JPY": 30.0},
        json_schema_extra={
            "basis": "guess",
            "note": (
                "USD 10-year realised vol of daily changes 1990-2026 is 92bp a year, FRED "
                "DGS10; GBP, EUR and JPY are guesses scaled to their lower yield levels; "
                "OECD long-term rate series on FRED (IRLTLT01GBM156N, IRLTLT01DEM156N, "
                "IRLTLT01JPM156N) would verify them."
            ),
        },
    )
    slope_vol_bp: float = Field(
        60.0,
        gt=0,
        json_schema_extra={
            "basis": "sourced",
            "note": "the 2s10s spread's realised volatility 2000-2026 is 62bp a year, FRED T10Y2Y.",
        },
    )
    driver_corr: float = Field(
        0.3,
        ge=-1,
        le=1,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "the correlation between the 10-year yield change and the equity return "
                "is 0.34 for 2000-2020 and 0.26 for 2000-2026, the post-2000 regime of "
                "Campbell, Pflueger and Viceira 2020; it turned negative in 2022. FRED "
                "DGS10, NASDAQCOM."
            ),
        },
    )

    @field_validator("level_vol_bp")
    @classmethod
    def _check_level_vol_bp_positive(cls, value: dict[str, float]) -> dict[str, float]:
        if any(v <= 0 for v in value.values()):
            raise ValueError("level_vol_bp must be positive")
        return value


class CreditFamilyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    factor_vol: float = Field(
        0.25,
        gt=0,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "daily log OAS change volatility 2023-2026 is 0.23 for investment grade "
                "and 0.33 for high yield, FRED BAMLC0A0CM and BAMLH0A0HYM2."
            ),
        },
    )
    hy_vol_multiplier: float = Field(
        1.4,
        gt=0,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "ratio of the high-yield to investment-grade daily log OAS change "
                "volatility, 0.33 over 0.23, FRED BAMLH0A0HYM2 and BAMLC0A0CM."
            ),
        },
    )
    driver_corr: float = Field(
        -0.5,
        ge=-1,
        le=1,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "investment-grade and high-yield spreads correlate with the S&P 500 at "
                "-0.43 and -0.62 respectively, 2023-2026, FRED."
            ),
        },
    )
    asymmetry: float = Field(
        1.1,
        ge=1,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "the mean widening step over the mean tightening step of the daily log "
                "OAS 2023-2026 is 1.04 to 1.11, with skew 0.5 to 0.9, FRED."
            ),
        },
    )
    issuer_vol: float = Field(
        0.15,
        gt=0,
        json_schema_extra={
            "basis": "guess",
            "note": "issuer-level OAS is not public; set to about half the index volatility.",
        },
    )


class CommodityFamilyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    group_vol: dict[CommodityGroup, float] = Field(
        default_factory=lambda: {
            CommodityGroup.ENERGY: 0.40,
            CommodityGroup.INDUSTRIAL_METALS: 0.21,
            CommodityGroup.PRECIOUS: 0.15,
            CommodityGroup.AGRICULTURE: 0.23,
        },
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "WTI daily realised volatility 2000-2026 is 47 percent, and 37 percent "
                "excluding 2020, FRED DCOILWTICO; copper monthly volatility is 21 "
                "percent, FRED PCOPPUSDM; wheat and maize monthly volatility are 25 and "
                "21 percent, FRED PWHEAMTUSDM and PMAIZMTUSDM; precious is a guess at "
                "gold's usual 15 percent."
            ),
        },
    )
    driver_corr: dict[CommodityGroup, float] = Field(
        default_factory=lambda: {
            CommodityGroup.ENERGY: 0.15,
            CommodityGroup.INDUSTRIAL_METALS: 0.3,
            CommodityGroup.PRECIOUS: -0.1,
            CommodityGroup.AGRICULTURE: 0.0,
        },
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "WTI's correlation with the S&P 500 is 0.15 daily 2016-2026; copper's "
                "correlation with the NASDAQ is 0.30 monthly; wheat is -0.01 and maize is "
                "0.06, FRED; precious is a guess."
            ),
        },
    )
    curve_slope: dict[CommodityGroup, float] = Field(
        default_factory=lambda: {
            CommodityGroup.ENERGY: -0.05,
            CommodityGroup.INDUSTRIAL_METALS: 0.01,
            CommodityGroup.PRECIOUS: 0.03,
            CommodityGroup.AGRICULTURE: 0.04,
        },
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "average annual roll yield by commodity group, Erb and Harvey 2006 and "
                "Gorton and Rouwenhorst 2006; negative is backwardation."
            ),
        },
    )
    group_share: float = Field(
        0.6,
        gt=0,
        lt=1,
        json_schema_extra={
            "basis": "guess",
            "note": (
                "share of group variance versus idiosyncratic variance per commodity, "
                "about 0.5-0.7."
            ),
        },
    )

    @field_validator("group_vol", "driver_corr", "curve_slope")
    @classmethod
    def _check_commodity_group_keys(
        cls, value: dict[CommodityGroup, float]
    ) -> dict[CommodityGroup, float]:
        if set(value) != set(CommodityGroup):
            raise ValueError(f"must cover every commodity group: {set(CommodityGroup)}")
        return value

    @field_validator("group_vol")
    @classmethod
    def _check_group_vol_positive(
        cls, value: dict[CommodityGroup, float]
    ) -> dict[CommodityGroup, float]:
        if any(v <= 0 for v in value.values()):
            raise ValueError("group_vol must be positive")
        return value

    @field_validator("driver_corr")
    @classmethod
    def _check_commodity_driver_corr_range(
        cls, value: dict[CommodityGroup, float]
    ) -> dict[CommodityGroup, float]:
        if any(not (-1 <= v <= 1) for v in value.values()):
            raise ValueError("driver_corr must be within [-1, 1]")
        return value


class FxFamilyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    currency_vol: dict[str, float] = Field(
        default_factory=lambda: {
            "EUR": 0.09,
            "GBP": 0.09,
            "JPY": 0.10,
            "AUD": 0.12,
            "CHF": 0.10,
            "CAD": 0.08,
        },
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "realised volatility of EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF and "
                "USDCAD 2000-2026, FRED DEXUSEU, DEXUSUK, DEXJPUS, DEXUSAL, DEXSZUS, "
                "DEXCAUS."
            ),
        },
    )
    driver_corr: dict[str, float] = Field(
        default_factory=lambda: {
            "EUR": 0.1,
            "GBP": 0.2,
            "JPY": -0.1,
            "AUD": 0.3,
            "CHF": 0.0,
            "CAD": 0.25,
        },
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "correlation with the S&P 500 2016-2026 is 0.11, 0.21, -0.09, 0.30, 0.02 "
                "and 0.24 for EUR, GBP, JPY, AUD, CHF and CAD, FRED."
            ),
        },
    )

    @field_validator("currency_vol", "driver_corr")
    @classmethod
    def _check_fx_currency_keys(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != set(USD_PAIR):
            raise ValueError(f"must cover every non-USD currency: {set(USD_PAIR)}")
        return value

    @field_validator("currency_vol")
    @classmethod
    def _check_currency_vol_positive(cls, value: dict[str, float]) -> dict[str, float]:
        if any(v <= 0 for v in value.values()):
            raise ValueError("currency_vol must be positive")
        return value

    @field_validator("driver_corr")
    @classmethod
    def _check_fx_driver_corr_range(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not (-1 <= v <= 1) for v in value.values()):
            raise ValueError("driver_corr must be within [-1, 1]")
        return value


class MarketFamiliesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    equity: EquityFamilyConfig = Field(default_factory=EquityFamilyConfig)
    rates: RatesFamilyConfig = Field(default_factory=RatesFamilyConfig)
    credit: CreditFamilyConfig = Field(default_factory=CreditFamilyConfig)
    commodity: CommodityFamilyConfig = Field(default_factory=CommodityFamilyConfig)
    fx: FxFamilyConfig = Field(default_factory=FxFamilyConfig)
    student_t_df: int = Field(
        4,
        ge=3,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "S&P 500 daily kurtosis 2016-2026 is about 20, FRED SP500; a t "
                "distribution with 4 degrees of freedom already has infinite kurtosis."
            ),
        },
    )


class MarketLevelsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    equity_price_range: tuple[float, float] = Field(
        (10.0, 400.0),
        json_schema_extra={
            "basis": "design",
            "note": "starting price range for simulated equities",
        },
    )
    curve_start: dict[str, tuple[float, float, float, float]] = Field(
        default_factory=lambda: {
            "USD": (4.0, 3.9, 4.1, 4.4),
            "EUR": (2.4, 2.3, 2.5, 2.8),
            "GBP": (4.2, 4.0, 4.2, 4.6),
            "JPY": (0.4, 0.5, 1.0, 2.0),
        },
        json_schema_extra={
            "basis": "design",
            "note": (
                "approximate 2026 sovereign yield curve shape by currency, in percent "
                "for 2Y/5Y/10Y/30Y"
            ),
        },
    )
    credit_base_spread_bp: dict[RatingBand, float] = Field(
        default_factory=lambda: {
            RatingBand.AA: 50,
            RatingBand.A: 70,
            RatingBand.BBB: 105,
            RatingBand.BB: 180,
            RatingBand.B: 305,
        },
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "ICE BofA OAS medians 2023-09 to 2026-09 by rating band, FRED "
                "BAMLC0A2CAA, BAMLC0A3CA, BAMLC0A4CBBB, BAMLH0A1HYBB, BAMLH0A2HYB."
            ),
        },
    )
    fx_start: dict[str, float] = Field(
        default_factory=lambda: {
            "EURUSD": 1.10,
            "GBPUSD": 1.28,
            "USDJPY": 150.0,
            "AUDUSD": 0.66,
            "USDCHF": 0.88,
            "USDCAD": 1.36,
        },
        json_schema_extra={"basis": "design", "note": "starting level for each simulated FX pair"},
    )
    yield_floor_pct: float = Field(
        0.0,
        json_schema_extra={
            "basis": "design",
            "note": "floor below which simulated sovereign yields cannot fall",
        },
    )

    @field_validator("equity_price_range")
    @classmethod
    def _check_equity_price_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = value
        if not (0 < lo < hi):
            raise ValueError("equity_price_range must have 0 < lo < hi")
        return value

    @field_validator("credit_base_spread_bp")
    @classmethod
    def _check_credit_base_spread_bp(
        cls, value: dict[RatingBand, float]
    ) -> dict[RatingBand, float]:
        if set(value) != set(RatingBand):
            raise ValueError(f"credit_base_spread_bp keys must be exactly {set(RatingBand)}")
        if any(v <= 0 for v in value.values()):
            raise ValueError("credit_base_spread_bp must be positive")
        return value

    @field_validator("fx_start")
    @classmethod
    def _check_fx_start(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = set(value) - set(FX_PAIRS)
        if unknown:
            raise ValueError(f"fx_start has unknown pair(s): {sorted(unknown)}")
        if any(v <= 0 for v in value.values()):
            raise ValueError("fx_start must be positive")
        return value


class EventSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    per_year: float = Field(ge=0)
    jump_size: float = Field(ge=0)
    placement: Literal["grid", "poisson"]
    jitter_days: int = Field(0, ge=0)
    basis: Basis
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_grid_jitter(self) -> "EventSpec":
        if self.placement == "grid":
            if self.per_year <= 0:
                raise ValueError("grid placement requires per_year > 0")
            if HORIZON_DAYS_PER_YEAR / self.per_year <= 2 * self.jitter_days:
                raise ValueError("jitter_days is too large to keep jittered grid dates unique")
        return self


_MARKET_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.EARNINGS,
    EventType.RATING_DOWNGRADE,
    EventType.RATING_UPGRADE,
    EventType.CB_MEETING,
    EventType.INVENTORY_REPORT,
    EventType.CROP_REPORT,
    EventType.MACRO_PRINT,
)


def _default_market_events() -> dict[EventType, EventSpec]:
    return {
        EventType.EARNINGS: EventSpec(
            per_year=4,
            jump_size=0.05,
            placement="grid",
            jitter_days=5,
            basis="sourced",
            note=(
                "Dubinsky, Johannes, Kaeck and Seeger 2019 estimate of typical "
                "earnings-day option-implied jump size."
            ),
        ),
        EventType.RATING_DOWNGRADE: EventSpec(
            per_year=0.2,
            jump_size=0.20,
            placement="poisson",
            basis="guess",
            note=(
                "Frequency is a guess; the direction of the spread reaction follows "
                "Hand, Holthausen and Leftwich 1992."
            ),
        ),
        EventType.RATING_UPGRADE: EventSpec(
            per_year=0.1,
            jump_size=0.20,
            placement="poisson",
            basis="guess",
            note="Frequency and magnitude are a guess.",
        ),
        EventType.CB_MEETING: EventSpec(
            per_year=8,
            jump_size=8.0,
            placement="grid",
            basis="sourced",
            note=(
                "Eight scheduled FOMC meetings a year; jump size is in basis points on "
                "the curve level, following Gurkaynak, Sack and Swanson 2005 on "
                "policy-surprise magnitude."
            ),
        ),
        EventType.INVENTORY_REPORT: EventSpec(
            per_year=52,
            jump_size=0.012,
            placement="grid",
            basis="sourced",
            note=(
                "WTI's Wednesday realised volatility is 3.15 percent versus 2.93 "
                "percent on other days, FRED DCOILWTICO."
            ),
        ),
        EventType.CROP_REPORT: EventSpec(
            per_year=12,
            jump_size=0.03,
            placement="grid",
            basis="sourced",
            note=(
                "USDA WASDE monthly report; jump size follows Adjemian 2012 on price "
                "reaction magnitude."
            ),
        ),
        EventType.MACRO_PRINT: EventSpec(
            per_year=12,
            jump_size=1.0,
            placement="grid",
            basis="design",
            note=(
                "monthly macro data releases; jump size is a driver standard "
                "deviation, not a price percentage."
            ),
        ),
    }


class MarketConsensusConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    street_window_days: int = Field(
        20,
        gt=0,
        json_schema_extra={
            "basis": "guess",
            "note": (
                "trailing window for the street consensus view; also its EMA "
                "half-life in trading days"
            ),
        },
    )
    positioning_window_multiple: int = Field(
        3,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": (
                "positioning window as a multiple of the street window; that "
                "window is also its EMA half-life in trading days"
            ),
        },
    )
    revision_weekday: int = Field(
        2,
        ge=0,
        le=4,
        json_schema_extra={
            "basis": "design",
            "note": "0 is Monday; consensus revises midweek, Wednesday.",
        },
    )
    view_threshold: float = Field(
        0.25,
        gt=0,
        lt=0.5,
        json_schema_extra={
            "basis": "design",
            "note": "keeps twice the threshold inside [-1, 1] for the flip test",
        },
    )
    flips_per_instrument_year: float = Field(
        2.0,
        ge=0,
        json_schema_extra={
            "basis": "guess",
            "note": "rate of street view flips per instrument per year",
        },
    )
    positioning_thresholds: tuple[float, float] = Field(
        (20.0, 80.0),
        json_schema_extra={
            "basis": "design",
            "note": "positioning percentile bands for crowded short and crowded long",
        },
    )
    report_weekday: int = Field(
        4,
        ge=0,
        le=4,
        json_schema_extra={
            "basis": "sourced",
            "note": "0 is Monday; CFTC Commitment of Traders reports release on Fridays.",
        },
    )

    @field_validator("positioning_thresholds")
    @classmethod
    def _check_positioning_thresholds(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = value
        if not (0 <= lo < hi <= 100):
            raise ValueError("positioning_thresholds must satisfy 0 <= lo < hi <= 100")
        return value


class MarketCheckConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vol_tolerance: float = Field(
        0.30,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": "allowed relative slack on model-implied vol targets",
        },
    )
    corr_tolerance_se: float = Field(
        4.0,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": (
                "about 1 in 16,000 false alarms per test in Fisher-z space, which "
                "stabilises the variance of a correlation estimate near +-1"
            ),
        },
    )
    min_round_level_tests: float = Field(
        3.0,
        ge=0,
        json_schema_extra={
            "basis": "design",
            "note": "minimum mean round-level crossings per instrument to check",
        },
    )
    round_level_band: float = Field(
        0.002,
        gt=0,
        lt=0.05,
        json_schema_extra={
            "basis": "design",
            "note": "width of the band around a round level counted as a test",
        },
    )


def _default_market_seeds() -> dict[str, tuple[Regime, Regime, Regime]]:
    return {
        "A": (Regime.RANGE, Regime.RISK_OFF, Regime.RISK_ON),
        "B": (Regime.RISK_ON, Regime.RANGE, Regime.RISK_OFF),
        "C": (Regime.RISK_OFF, Regime.RISK_ON, Regime.RANGE),
    }


class RealSeedSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window_start: date
    regime_starts: tuple[tuple[Regime, date], tuple[Regime, date], tuple[Regime, date]]
    basis: Basis
    note: str = Field(min_length=1)

    @field_validator("window_start")
    @classmethod
    def _check_window_start_monday(cls, value: date) -> date:
        if value.weekday() != 0:
            raise ValueError("window_start must be a Monday")
        return value

    @model_validator(mode="after")
    def _check_regime_starts(self) -> "RealSeedSpec":
        regimes = [regime for regime, _ in self.regime_starts]
        if set(regimes) != set(Regime):
            raise ValueError(f"regime_starts must list each regime exactly once: {regimes}")
        dates = [start for _, start in self.regime_starts]
        if dates[0] != self.window_start:
            raise ValueError("regime_starts must begin at window_start")
        for earlier, later in zip(dates, dates[1:], strict=False):
            if earlier >= later:
                raise ValueError("regime_starts dates must be strictly ascending")
        for start in dates:
            if start.weekday() >= 5:
                raise ValueError(f"regime_starts date {start} must fall on a weekday")
        return self


def real_window_end(spec: RealSeedSpec, n_weeks: int) -> date:
    """The exclusive end of a real seed's simulation window."""
    return spec.window_start + timedelta(weeks=n_weeks)


def _last_weekday_before(day: date) -> date:
    """The last weekday strictly before `day`.

    `window_start` is always a Monday and `real_window_end` lands on a Monday
    too, so a seed's last simulated day is always the Friday 3 days earlier.
    """
    result = day - timedelta(days=1)
    while result.weekday() >= 5:
        result -= timedelta(days=1)
    return result


# A longer gap than a holiday week means the underlying series is broken, not thin.
REAL_FILL_LIMIT = 5
# So the axis's first day can still be filled forward from a value before the window.
REAL_FETCH_BUFFER_DAYS = 10
# Spaces requests to stay under Yahoo Finance's unofficial rate limit.
REAL_REQUEST_INTERVAL_S = 1.0
# Enough attempts to ride out a transient network failure.
REAL_RETRIES = 5


def _default_real_seeds() -> dict[str, RealSeedSpec]:
    return {
        "R1": RealSeedSpec(
            window_start=date(2018, 6, 4),
            regime_starts=(
                (Regime.RANGE, date(2018, 6, 4)),
                (Regime.RISK_OFF, date(2018, 10, 1)),
                (Regime.RISK_ON, date(2018, 12, 26)),
            ),
            basis="design",
            note=(
                "a calm range (+7%), the Q4 2018 selloff (-19%) and the 2019 rebound "
                "(+12%) give the pilot seed the range, risk_off, risk_on order; the "
                "year is less memorable than 2020 or 2022"
            ),
        )
    }


class MarketRealConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seeds: dict[str, RealSeedSpec] = Field(default_factory=_default_real_seeds)
    sec_user_agent: str = Field(
        "pm-traitbench admin@example.com",
        json_schema_extra={
            "basis": "design",
            "note": (
                "SEC's fair-access policy asks automated clients to declare a name "
                "and a contact email"
            ),
        },
    )

    @field_validator("sec_user_agent")
    @classmethod
    def _check_sec_user_agent(cls, value: str) -> str:
        parts = value.split()
        if len(parts) < 2 or "@" not in parts[-1]:
            raise ValueError("sec_user_agent must be a space-separated name and an email address")
        return value


class MarketConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seeds: dict[str, tuple[Regime, Regime, Regime]] = Field(
        default_factory=_default_market_seeds,
        json_schema_extra={
            "basis": "design",
            "note": "each market seed cycles the three regimes in a distinct order",
        },
    )
    boundary_weeks: tuple[int, int] = Field(
        (16, 30),
        json_schema_extra={
            "basis": "design",
            "note": "weeks at which a market seed's regime changes",
        },
    )
    burn_in_days: int = Field(
        60,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": "days of warm-up simulated before the first published day",
        },
    )
    universe: MarketUniverseConfig = Field(default_factory=MarketUniverseConfig)
    regimes: MarketRegimesConfig = Field(default_factory=MarketRegimesConfig)
    families: MarketFamiliesConfig = Field(default_factory=MarketFamiliesConfig)
    levels: MarketLevelsConfig = Field(default_factory=MarketLevelsConfig)
    events: dict[EventType, EventSpec] = Field(default_factory=_default_market_events)
    consensus: MarketConsensusConfig = Field(default_factory=MarketConsensusConfig)
    check: MarketCheckConfig = Field(default_factory=MarketCheckConfig)
    real: MarketRealConfig = Field(default_factory=MarketRealConfig)

    @field_validator("seeds")
    @classmethod
    def _check_seeds(
        cls, value: dict[str, tuple[Regime, Regime, Regime]]
    ) -> dict[str, tuple[Regime, Regime, Regime]]:
        for name, regimes in value.items():
            if set(regimes) != set(Regime):
                raise ValueError(
                    f"seed '{name}' must list each regime exactly once: {list(regimes)}"
                )
        return value

    @field_validator("boundary_weeks")
    @classmethod
    def _check_boundary_weeks(cls, value: tuple[int, int]) -> tuple[int, int]:
        first, last = value
        if first >= last:
            raise ValueError("boundary_weeks must be strictly increasing")
        return value

    @field_validator("events")
    @classmethod
    def _check_events_keys(cls, value: dict[EventType, EventSpec]) -> dict[EventType, EventSpec]:
        if set(value) != set(_MARKET_EVENT_TYPES):
            raise ValueError(f"events keys must be exactly {_MARKET_EVENT_TYPES}")
        return value

    @model_validator(mode="after")
    def _check_seed_namespaces(self) -> "MarketConfig":
        overlap = set(self.seeds) & set(self.real.seeds)
        if overlap:
            raise ValueError(
                f"seed name(s) in both market.seeds and market.real.seeds: {sorted(overlap)}"
            )
        if "synthetic" in self.seeds or "synthetic" in self.real.seeds:
            raise ValueError(
                "seed name 'synthetic' is reserved for gate 1's pooled synthetic group"
            )
        return self

    @model_validator(mode="after")
    def _check_cross_references(self) -> "MarketConfig":
        missing_curves = set(self.universe.curves) - set(self.levels.curve_start)
        if missing_curves:
            raise ValueError(f"levels.curve_start missing curve(s): {sorted(missing_curves)}")
        missing_level_vol = set(self.universe.curves) - set(self.families.rates.level_vol_bp)
        if missing_level_vol:
            raise ValueError(
                f"families.rates.level_vol_bp missing curve(s): {sorted(missing_level_vol)}"
            )
        used_currencies = {
            currency
            for pair in self.universe.fx_pairs
            for currency in FX_PAIRS[pair]
            if currency != "USD"
        }
        missing_vol = used_currencies - set(self.families.fx.currency_vol)
        if missing_vol:
            raise ValueError(
                f"families.fx.currency_vol missing currency(ies): {sorted(missing_vol)}"
            )
        missing_start = {USD_PAIR[currency] for currency in used_currencies} - set(
            self.levels.fx_start
        )
        if missing_start:
            raise ValueError(f"levels.fx_start missing pair(s): {sorted(missing_start)}")
        return self


_TURN_RANGE_FIELD_BY_KIND: dict[SessionKind, str] = {
    SessionKind.SILENCE: "silence",
    SessionKind.CHECK_IN: "check_in",
    SessionKind.DECISION: "decision",
}


class TurnRanges(BaseModel):
    """Candidate turn counts a dialogue session may run, keyed by session kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    silence: tuple[int, ...] = Field(
        (2, 4),
        json_schema_extra={
            "basis": "guess",
            "note": "a factual question and its answer, at most one follow-up",
        },
    )
    check_in: tuple[int, ...] = Field(
        (2, 4, 6),
        json_schema_extra={
            "basis": "guess",
            "note": "a status update takes a little longer than a silent session",
        },
    )
    decision: tuple[int, ...] = Field(
        (4, 6, 8),
        json_schema_extra={
            "basis": "guess",
            "note": "a decision gets discussed before it is closed",
        },
    )

    @field_validator("silence", "check_in", "decision")
    @classmethod
    def _check_turn_counts(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("turn counts must not be empty")
        if list(value) != sorted(set(value)):
            raise ValueError(f"turn counts must be strictly increasing: {list(value)}")
        if any(n % 2 != 0 or not (2 <= n <= 8) for n in value):
            raise ValueError(f"turn counts must be even and within [2, 8]: {list(value)}")
        return value

    def for_kind(self, kind: SessionKind) -> tuple[int, ...]:
        return getattr(self, _TURN_RANGE_FIELD_BY_KIND[kind])


class PmFilter(BaseModel):
    """Optional criteria to restrict which PMs the dialogue stage generates sessions for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Split | None = Field(
        None,
        json_schema_extra={
            "basis": "design",
            "note": "restrict to one split so a first run can measure cost before scaling up",
        },
    )
    typicality: Typicality | None = Field(
        None,
        json_schema_extra={
            "basis": "design",
            "note": "restrict to one typicality so a first run can measure cost before scaling up",
        },
    )
    drift: DriftStatus | None = Field(
        None,
        json_schema_extra={
            "basis": "design",
            "note": (
                "restrict to static or drift PMs, so narration can go static pilot PMs "
                "first, then drift PMs once dialogue recovery is checked"
            ),
        },
    )
    pm_ids: tuple[str, ...] = Field(
        (),
        json_schema_extra={
            "basis": "design",
            "note": "restrict to a few named PMs so a first run can measure cost before scaling up",
        },
    )


class DialogueConfig(BaseModel):
    """Settings for the dialogue stage: narrator and advisor models, turn budgets and limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    narrator_model: str = Field(
        DEFAULT_MODEL,
        json_schema_extra={
            "basis": "design",
            "note": "one narrator model so narration never confounds trait recovery",
        },
    )
    advisor_model: str = Field(
        DEFAULT_MODEL,
        json_schema_extra={
            "basis": "design",
            "note": (
                "same model as the narrator, so one API behaviour to handle; the two "
                "sides differ by prompt and inputs"
            ),
        },
    )
    effort: Effort = Field(
        Effort.LOW,
        json_schema_extra={
            "basis": "design",
            "note": "a narrated turn is short and constrained by its directive",
        },
    )
    advisor_prompt_path: Path | None = Field(
        None,
        json_schema_extra={"basis": "design", "note": "None means the packaged advisor prompt"},
    )
    turns_by_kind: TurnRanges = Field(default_factory=TurnRanges)
    max_tool_rounds: int = Field(
        3,
        ge=1,
        json_schema_extra={
            "basis": "guess",
            "note": "parallel tool calls answer most turns in one round",
        },
    )
    max_retries: int = Field(
        3,
        ge=0,
        json_schema_extra={
            "basis": "guess",
            "note": (
                "re-sends of a reply the pipeline rejected (refusal, truncation or invalid "
                "output); transport errors are retried separately, by the SDK"
            ),
        },
    )
    api_max_retries: int = Field(
        4,
        ge=0,
        json_schema_extra={
            "basis": "guess",
            "note": "SDK retries for rate limits, overloads and dropped connections, with backoff",
        },
    )
    max_concurrency: int = Field(
        8,
        ge=1,
        json_schema_extra={"basis": "guess", "note": "well inside default API rate limits"},
    )
    max_output_tokens: int = Field(
        4000,
        ge=256,
        json_schema_extra={
            "basis": "design",
            "note": "room for low-effort thinking plus a short turn",
        },
    )
    token_budget: int | None = Field(
        None,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": "fresh input plus output tokens per run",
        },
    )
    pm_filter: PmFilter = Field(default_factory=PmFilter)


class ValidateConfig(BaseModel):
    """Settings for the validate stage: the judge model, tolerances and attempt budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_model: str = Field(
        DEFAULT_MODEL,
        json_schema_extra={
            "basis": "design",
            "note": "strongest current model, one judge",
        },
    )
    refusal_fallback_model: str | None = Field(
        "claude-sonnet-5",
        min_length=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "re-judges a request the judge model refused, such as an API "
                "reasoning-extraction refusal; None disables the fallback"
            ),
        },
    )
    effort: Effort = Field(
        Effort.LOW,
        json_schema_extra={"basis": "design", "note": "a yes/no reading task"},
    )
    max_output_tokens: int = Field(
        1000,
        gt=0,
        json_schema_extra={"basis": "design", "note": "two short JSON objects"},
    )
    size_tolerance: float = Field(
        0.05,
        ge=0,
        lt=1,
        json_schema_extra={
            "basis": "guess",
            "note": "the narrator sees raw floats and may round to a desk-sized figure",
        },
    )
    level_tolerance: float = Field(
        0.01,
        ge=0,
        lt=1,
        json_schema_extra={
            "basis": "guess",
            "note": "quoted levels are rounded to display precision",
        },
    )
    max_attempts: int = Field(
        3,
        ge=1,
        json_schema_extra={
            "basis": "guess",
            "note": "one original plus two regenerations, a third failure is a prompt problem",
        },
    )
    max_concurrency: int = Field(
        8,
        ge=1,
        json_schema_extra={"basis": "design", "note": "same as dialogue"},
    )
    token_budget: int | None = Field(
        None,
        gt=0,
        json_schema_extra={"basis": "design", "note": "same semantics as dialogue"},
    )


class HarnessConfig(BaseModel):
    """Settings for the evaluation harness: the baseline model and answer limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        DEFAULT_MODEL,
        json_schema_extra={
            "basis": "design",
            "note": "the Gate 2 model, so the full-context baseline is the Gate 2 ceiling",
        },
    )
    effort: Effort = Field(
        Effort.HIGH,
        json_schema_extra={
            "basis": "design",
            "note": "Gate 2's recovery effort, so the baseline matches the Gate 2 ceiling",
        },
    )
    max_answer_tokens: int = Field(
        8000,
        ge=256,
        json_schema_extra={
            "basis": "design",
            "note": (
                "high-effort thinking plus a reply up to a short page; a reply still "
                "unparsable after retries is recorded empty and scores wrong"
            ),
        },
    )
    short_page_words: int = Field(
        400,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": 'about half a printed page, the ceiling for "up to a short page"',
        },
    )
    pm_token_budget: int | None = Field(
        None,
        ge=1,
        json_schema_extra={
            "basis": "guess",
            "note": (
                "cap on fresh tokens per PM for a baseline, since each PM's adapter owns "
                "its client; spending it fails that PM"
            ),
        },
    )


class JudgeConfig(BaseModel):
    """Settings for the LLM judges that grade open replies against a rubric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        DEFAULT_MODEL,
        json_schema_extra={
            "basis": "design",
            "note": "one judge model; the human-rated sample bounds its noise",
        },
    )
    effort: Effort = Field(
        Effort.HIGH,
        json_schema_extra={
            "basis": "design",
            "note": "grading against a rubric benefits from thinking, at Gate 2's effort",
        },
    )
    max_tokens: int = Field(
        4000,
        ge=256,
        json_schema_extra={
            "basis": "design",
            "note": "thinking plus a short JSON verdict, the dialogue stage's output cap",
        },
    )
    max_concurrency: int = Field(
        8,
        ge=1,
        json_schema_extra={
            "basis": "guess",
            "note": "the dialogue stage's value, well inside default API rate limits",
        },
    )
    sample_size: int = Field(
        100,
        ge=5,
        json_schema_extra={
            "basis": "guess",
            "note": "20 items per judge, enough to notice an agreement rate below 0.8",
        },
    )
    token_budget: int | None = Field(
        None,
        ge=1,
        json_schema_extra={
            "basis": "guess",
            "note": (
                "cap on fresh tokens for one judge pass; spending it stops the pass "
                "and finished items stay cached"
            ),
        },
    )


class Gate2Config(BaseModel):
    """Settings for Gate 2: the recovery model, exact-test level, overlap measure and limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        DEFAULT_MODEL,
        json_schema_extra={"basis": "design", "note": "strongest current model, one model"},
    )
    effort: Effort = Field(
        Effort.HIGH,
        json_schema_extra={
            "basis": "design",
            "note": (
                "the recovery ceiling wants the strong model at strength, unlike the "
                "narrator and judges at low"
            ),
        },
    )
    recovery_max_output_tokens: int = Field(
        32000,
        ge=256,
        json_schema_extra={
            "basis": "design",
            "note": (
                "high-effort thinking over about 100k transcript tokens plus one JSON "
                "object; a max_tokens stop is an unparsable reply that identical retries repeat"
            ),
        },
    )
    classify_max_output_tokens: int = Field(
        8000,
        ge=256,
        json_schema_extra={
            "basis": "design",
            "note": (
                "high-effort thinking plus one or two quotes; a max_tokens stop is an "
                "unparsable reply that identical retries repeat"
            ),
        },
    )
    alpha: float = Field(
        0.05,
        gt=0,
        lt=1,
        json_schema_extra={
            "basis": "guess",
            "note": (
                "one-sided exact test level per blocking row; nine rows give a "
                "family-wise false-pass bound of about 0.37 under no recovery, and power "
                "at pilot size is low, so a pass needs near-perfect recovery on 12 PMs"
            ),
        },
    )
    min_class: int = Field(
        2,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": "the smallest class an exact test can say anything about",
        },
    )
    ngram_n: int = Field(
        5,
        ge=2,
        json_schema_extra={
            "basis": "guess",
            "note": "long enough to be a phrase, short enough to recur",
        },
    )
    overlap_warning: float = Field(
        0.15,
        ge=0,
        le=1,
        json_schema_extra={"basis": "guess", "note": "to be re-centred on pilot output"},
    )
    max_concurrency: int = Field(
        4,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": "long-context calls, half the dialogue stage's",
        },
    )
    token_budget: int | None = Field(
        None,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": "fresh input plus output tokens per run, as dialogue",
        },
    )


class ProbesConfig(BaseModel):
    """Settings for the probes stage: checkpoint offsets, probe counts and the MCQ situations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    post_drift_weeks: int = Field(
        4,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "the post-drift checkpoint sits four weeks after the event, time for the "
                "change to show in several sessions"
            ),
        },
    )
    presence_never_held: int = Field(
        3,
        ge=0,
        json_schema_extra={
            "basis": "guess",
            "note": ("never-held preference values asked per checkpoint as presence negatives"),
        },
    )
    routine_per_checkpoint: int = Field(
        2,
        ge=0,
        json_schema_extra={
            "basis": "guess",
            "note": ("routine questions per checkpoint"),
        },
    )
    disposition_progress: float = Field(
        0.5,
        gt=0,
        lt=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "a gain about halfway to target, where a sale is early by the PM's own target"
            ),
        },
    )
    loss_depth: float = Field(
        0.5,
        gt=0,
        lt=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "the loss situation sits halfway to the stop, a loser the stop has not yet cut"
            ),
        },
    )
    anchor_approach: float = Field(
        0.9,
        gt=0,
        lt=1,
        json_schema_extra={
            "basis": "design",
            "note": ("the price is most of the way from entry to the round level but short of it"),
        },
    )
    extrapolation_thesis_sd: float = Field(
        -1.0,
        json_schema_extra={
            "basis": "design",
            "note": ("the own thesis at one normal move against the run"),
        },
    )
    extrapolation_trailing_sd: float = Field(
        3.0,
        json_schema_extra={
            "basis": "design",
            "note": (
                "a run of three normal moves; with the thesis at -1 the forecast reaches "
                "the entry threshold 1.0 at theta 0.5, between the neutral and active "
                "means"
            ),
        },
    )
    overconfidence_size_edges: tuple[float, float] = Field(
        (1.25, 2.0),
        json_schema_extra={
            "basis": "design",
            "note": (
                "size-factor bucket edges: calibrated coverage 0.8 gives factor 1.0, "
                "standard size; the active centre 0.4 gives about 2.45, the top bucket"
            ),
        },
    )
    conviction_rating: int = Field(
        2,
        ge=1,
        le=4,
        json_schema_extra={
            "basis": "design",
            "note": (
                "a low stated rating, so sizing to the rating and full size are different answers"
            ),
        },
    )
    decline_excess_pct: float = Field(
        5.0,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": ("a decline request asks for the mandate cap plus this many points of book"),
        },
    )
    max_horizon: int = Field(
        60,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": (
                "the longest horizon, in sessions, searched for a hazard bias's MCQ, "
                "about a quarter of a year"
            ),
        },
    )
    situation_attempts: int = Field(
        10,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": ("instruments tried before a situation is skipped"),
        },
    )

    @model_validator(mode="after")
    def _check_extrapolation_signs(self) -> "ProbesConfig":
        if not self.extrapolation_thesis_sd < 0 < self.extrapolation_trailing_sd:
            raise ValueError("extrapolation_thesis_sd must be negative and trailing_sd positive")
        return self

    @model_validator(mode="after")
    def _check_size_edges(self) -> "ProbesConfig":
        low, high = self.overconfidence_size_edges
        if not 1.0 < low < high:
            raise ValueError("overconfidence_size_edges must satisfy 1.0 < first < second")
        return self


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: SeedConfig = Field(default_factory=SeedConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    biases: BiasesConfig = Field(default_factory=BiasesConfig)
    mandate: MandateConfig = Field(default_factory=MandateConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    preferences: PreferencesConfig = Field(default_factory=PreferencesConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    gate1: Gate1Config = Field(default_factory=Gate1Config)
    plan: PlanConfig = Field(default_factory=PlanConfig)
    dialogue: DialogueConfig = Field(default_factory=DialogueConfig)
    validation: ValidateConfig = Field(default_factory=ValidateConfig)
    gate2: Gate2Config = Field(default_factory=Gate2Config)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    probes: ProbesConfig = Field(default_factory=ProbesConfig)

    @model_validator(mode="after")
    def _check_week_ranges_within_calendar(self) -> "Config":
        n_weeks = self.calendar.n_weeks
        ranges = {
            "drift.bias_update_weeks": self.drift.bias_update_weeks,
            "drift.preference_update_weeks": self.drift.preference_update_weeks,
            "drift.dormant_weeks": self.drift.dormant_weeks,
            "drift.revive_weeks": self.drift.revive_weeks,
        }
        for path, (first, last) in ranges.items():
            if not (1 <= first <= last <= n_weeks):
                raise ValueError(f"{path} must fall within 1..{n_weeks}")
        return self

    @model_validator(mode="after")
    def _check_market_consistency(self) -> "Config":
        known_seeds = set(self.market.seeds) | set(self.market.real.seeds)
        for name, seeds in (
            ("population.pilot_market_seeds", self.population.pilot_market_seeds),
            ("population.market_seeds", self.population.market_seeds),
        ):
            missing = set(seeds) - known_seeds
            if missing:
                raise ValueError(
                    f"{name} references seed(s) not in market.seeds or "
                    f"market.real.seeds: {sorted(missing)}"
                )
        n_weeks = self.calendar.n_weeks
        for seed_name, spec in self.market.real.seeds.items():
            window_end = real_window_end(spec, n_weeks)
            for regime, start in spec.regime_starts:
                if not (spec.window_start <= start < window_end):
                    raise ValueError(
                        f"market.real.seeds['{seed_name}'] regime start {start} for "
                        f"{regime} must fall within [{spec.window_start}, {window_end})"
                    )

        used_seeds = set(self.population.pilot_market_seeds) | set(self.population.market_seeds)
        cover_start, cover_end = EVENT_DATES_COVER
        for seed_name, spec in self.market.real.seeds.items():
            if seed_name not in used_seeds:
                continue
            window_end = real_window_end(spec, n_weeks)
            last_day = _last_weekday_before(window_end)
            if not (cover_start <= spec.window_start and last_day <= cover_end):
                raise ValueError(
                    f"market.real.seeds['{seed_name}'] window {spec.window_start} to "
                    f"{last_day} falls outside the FOMC/WASDE/NFP date coverage "
                    f"{cover_start} to {cover_end}"
                )
        first, last = self.market.boundary_weeks
        if not (1 <= first < last <= n_weeks - 1):
            raise ValueError(f"market.boundary_weeks must fall within 1..{n_weeks - 1}")
        return self

    @model_validator(mode="after")
    def _check_engine_horizon_within_calendar(self) -> "Config":
        if self.engine.horizon_days * 4 > self.calendar.n_weeks * 5:
            raise ValueError(
                "engine.horizon_days must be at most a quarter of the horizon in trading days"
            )
        return self

    def timeline(self) -> Timeline:
        return Timeline(self.calendar.start, self.calendar.n_weeks)

    def dump_with_basis(self) -> list["BasisRow"]:
        rows: list[BasisRow] = []
        _walk(self, "", rows)
        return rows


def referenced_seeds(config: Config) -> list[str]:
    """Every market seed the population grid uses, pilot seeds first, without repeats."""
    seen: dict[str, None] = {}
    for seed in (*config.population.pilot_market_seeds, *config.population.market_seeds):
        seen.setdefault(seed, None)
    return list(seen)


@dataclass(frozen=True)
class BasisRow:
    path: str
    value: Any
    basis: Basis
    note: str


def _has_basis_field(value: Any) -> bool:
    return isinstance(value, BaseModel) and "basis" in type(value).model_fields


def _dumped(value: Any) -> Any:
    return value.model_dump(mode="python") if isinstance(value, BaseModel) else value


def _walk(model: BaseModel, prefix: str, rows: list[BasisRow]) -> None:
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        path = f"{prefix}.{name}" if prefix else name
        extra = field.json_schema_extra
        if isinstance(extra, dict) and "basis" in extra:
            rows.append(BasisRow(path, _dumped(value), extra["basis"], extra.get("note", "")))
        elif _has_basis_field(value):
            rows.append(BasisRow(path, _dumped(value), value.basis, value.note))
        elif isinstance(value, dict) and value and all(_has_basis_field(v) for v in value.values()):
            for key, entry in value.items():
                rows.append(BasisRow(f"{path}.{key}", _dumped(entry), entry.basis, entry.note))
        elif isinstance(value, dict) and not value:
            pass  # an empty dict of per-entry-basis models has no leaves to record
        elif isinstance(value, BaseModel):
            _walk(value, path, rows)
        else:
            raise ConfigError(f"leaf at '{path}' has no basis")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None) -> Config:
    """Load a config from YAML, deep-merged onto the binding defaults."""
    if path is None:
        return Config()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except OSError as e:
        raise ConfigError(f"failed to read config file '{path}': {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in config file '{path}': {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"config file '{path}' must contain a mapping")
    merged = _deep_merge(Config().model_dump(mode="python"), raw)
    try:
        return Config.model_validate(merged)
    except ValidationError as e:
        raise ConfigError(f"invalid configuration in '{path}': {e}") from e
