"""Pipeline configuration: sections, binding defaults and YAML loading."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pm_traitbench.distributions import BetaSpec, Distribution, LogNormalSpec
from pm_traitbench.enums import AssetClass, Regime
from pm_traitbench.errors import ConfigError
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
    pilot_per_cell: int = Field(
        1, ge=0, json_schema_extra={"basis": "design", "note": "small pilot batch per cell"}
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
