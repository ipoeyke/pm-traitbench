"""Pipeline configuration: sections, binding defaults and YAML loading."""

from dataclasses import dataclass
from datetime import date
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
    EventType,
    ExpiryRule,
    RatingBand,
    Regime,
)
from pm_traitbench.errors import ConfigError
from pm_traitbench.market.constants import COMMODITIES, FX_PAIRS, HORIZON_DAYS_PER_YEAR, USD_PAIR
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
    pilot_market_seed_count: int = Field(
        1,
        ge=1,
        json_schema_extra={
            "basis": "design",
            "note": "the pilot uses only this many of the leading market seeds",
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
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not repeat an entry: {list(values)}")
        if any(not seed.strip() for seed in self.market_seeds):
            raise ValueError("market_seeds must not contain a blank name")
        if self.pilot_market_seed_count > len(self.market_seeds):
            raise ValueError(
                f"pilot_market_seed_count is {self.pilot_market_seed_count} but only "
                f"{len(self.market_seeds)} market_seeds are configured"
            )
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
        ge=0,
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
        ge=0,
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

    @field_validator("commodities")
    @classmethod
    def _check_commodities(cls, value: dict[CommodityGroup, int]) -> dict[CommodityGroup, int]:
        if set(value) != set(CommodityGroup):
            raise ValueError(f"commodities keys must be exactly {set(CommodityGroup)}")
        for group, n in value.items():
            max_n = len(COMMODITIES[group])
            if not (0 <= n <= max_n):
                raise ValueError(f"commodities[{group}] must be between 0 and {max_n}")
        return value

    @field_validator("fx_pairs")
    @classmethod
    def _check_fx_pairs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(value) - set(FX_PAIRS)
        if unknown:
            raise ValueError(f"fx_pairs has unknown pair(s): {sorted(unknown)}")
        if len(set(value)) != len(value):
            raise ValueError(f"fx_pairs must not repeat an entry: {list(value)}")
        return value

    @field_validator("equity_beta_range", "credit_duration_range")
    @classmethod
    def _check_positive_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = value
        if not (0 < lo <= hi):
            raise ValueError("range must have 0 < lo <= hi")
        return value


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

    level_vol_bp: float = Field(
        90.0,
        gt=0,
        json_schema_extra={
            "basis": "sourced",
            "note": (
                "realised volatility of daily 10-year yield changes 1990-2026 is 92bp a "
                "year, FRED DGS10."
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
                "Eight scheduled FOMC meetings a year; jump size follows Gurkaynak, "
                "Sack and Swanson 2005 on policy-surprise magnitude."
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
            "note": "trailing window for the street consensus view",
        },
    )
    positioning_window_multiple: int = Field(
        3,
        gt=0,
        json_schema_extra={
            "basis": "design",
            "note": "positioning window as a multiple of the street window",
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
                "about 1 in 16,000 false alarms per test for a normal-distributed "
                "correlation estimate"
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
        ge=0,
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
    def _check_cross_references(self) -> "MarketConfig":
        missing_curves = set(self.universe.curves) - set(self.levels.curve_start)
        if missing_curves:
            raise ValueError(f"levels.curve_start missing curve(s): {sorted(missing_curves)}")
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
        missing_seeds = set(self.population.market_seeds) - set(self.market.seeds)
        if missing_seeds:
            raise ValueError(
                "market.seeds missing seed(s) used by population.market_seeds: "
                f"{sorted(missing_seeds)}"
            )
        first, last = self.market.boundary_weeks
        n_weeks = self.calendar.n_weeks
        if not (1 <= first < last <= n_weeks - 1):
            raise ValueError(f"market.boundary_weeks must fall within 1..{n_weeks - 1}")
        return self

    def timeline(self) -> Timeline:
        return Timeline(self.calendar.start, self.calendar.n_weeks)

    def dump_with_basis(self) -> list["BasisRow"]:
        rows: list[BasisRow] = []
        _walk(self, "", rows)
        return rows


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
