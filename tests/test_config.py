from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from pm_traitbench.config import (
    BIAS_PARAMS,
    BiasesConfig,
    BiasSpec,
    CalendarConfig,
    Config,
    DriftConfig,
    EventSpec,
    MarketConfig,
    RegimeParams,
    load_config,
)
from pm_traitbench.distributions import BetaSpec, LogNormalSpec
from pm_traitbench.enums import EventType, Regime
from pm_traitbench.errors import ConfigError


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_config_builds_with_defaults() -> None:
    config = Config()
    assert config.seed.root == 20260105
    assert config.population.pilot_per_cell == 2
    assert config.population.pilot_market_seed_count == 1
    assert config.population.full_per_cell == 3
    assert config.mandate.book_size_min == 50e6
    assert config.mandate.book_size_max == 2e9
    assert config.drift.bias_update_weeks == (18, 30)
    assert config.calendar.n_weeks == 52
    assert config.output.format == "jsonl"


def test_biases_params_keys_equal_bias_params_in_order() -> None:
    config = Config()
    assert tuple(config.biases.params.keys()) == BIAS_PARAMS


def test_config_timeline_matches_calendar() -> None:
    config = Config()
    timeline = config.timeline()
    assert timeline.start == date(2026, 1, 5)
    assert timeline.n_weeks == 52


def test_yaml_override_of_nested_leaf_keeps_siblings(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"biases": {"p_active": 0.5}})
    config = load_config(path)
    assert config.biases.p_active == 0.5
    assert config.biases.min_active == Config().biases.min_active
    assert config.biases.p_regime_cluster == Config().biases.p_regime_cluster


def test_yaml_override_of_one_bias_entry_field_keeps_others(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path, {"biases": {"params": {"herding_weight": {"cluster_regime": None}}}}
    )
    config = load_config(path)
    assert config.biases.params["herding_weight"].cluster_regime is None
    default = Config().biases.params
    for name in BIAS_PARAMS:
        if name == "herding_weight":
            continue
        assert config.biases.params[name] == default[name]


@pytest.mark.parametrize(
    ("population", "message"),
    [
        ({"asset_classes": ["equities", "equities"]}, "asset_classes"),
        ({"asset_classes": []}, "asset_classes"),
        ({"market_seeds": ["A", "A"]}, "market_seeds"),
        ({"market_seeds": []}, "market_seeds"),
        ({"market_seeds": ["A", " "]}, "market_seeds"),
        ({"pilot_market_seed_count": 4}, "pilot_market_seed_count"),
        ({"pilot_market_seed_count": 0}, "pilot_market_seed_count"),
        ({"pilot_per_cell": 0, "full_per_cell": 0}, "per_cell"),
    ],
)
def test_population_that_would_skew_or_empty_the_grid_is_rejected(
    population: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Config.model_validate({"population": population})


def test_population_problem_in_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"population": {"market_seeds": ["A", "A"]}}))
    with pytest.raises(ConfigError, match="market_seeds"):
        load_config(path)


@pytest.mark.parametrize("output", [{"format": "csv"}, {"tables": {"traits": "csv"}}])
def test_unsupported_output_format_raises_config_error(tmp_path: Path, output: dict) -> None:
    path = _write_yaml(tmp_path, {"output": output})
    with pytest.raises(ConfigError, match="output"):
        load_config(path)


def test_unknown_top_level_key_raises_config_error_naming_key(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"bogus_top_level": 1})
    with pytest.raises(ConfigError, match="bogus_top_level"):
        load_config(path)


def test_unknown_nested_key_raises_config_error_naming_key(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"biases": {"bogus_nested": 1}})
    with pytest.raises(ConfigError, match="bogus_nested"):
        load_config(path)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_list_document_raises_config_error_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_scalar_document_raises_config_error_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("just a scalar\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_load_config_none_equals_default() -> None:
    assert load_config(None) == Config()


def test_probability_outside_unit_interval_raises() -> None:
    with pytest.raises(ValidationError):
        BiasesConfig(p_active=1.5)


def test_week_range_first_greater_than_last_raises() -> None:
    with pytest.raises(ValidationError):
        DriftConfig(bias_update_weeks=(30, 18))


def test_week_range_outside_calendar_horizon_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"drift": {"bias_update_weeks": (1, 60)}, "calendar": {"n_weeks": 52}}
        )


def test_non_monday_calendar_start_raises() -> None:
    with pytest.raises(ValidationError):
        CalendarConfig(start=date(2026, 1, 6), n_weeks=52)


def test_non_monday_calendar_start_from_yaml_raises_config_error(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"calendar": {"start": date(2026, 1, 6)}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_degenerate_truncation_bound_from_yaml_raises_config_error(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"biases": {"regime_multiplier": {"lo": 20.0, "hi": None}}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_params_out_of_order_input_normalizes_to_bias_params_order() -> None:
    default = Config().biases.params
    shuffled = {name: default[name] for name in reversed(BIAS_PARAMS)}
    biases = BiasesConfig(params=shuffled)
    assert tuple(biases.params.keys()) == BIAS_PARAMS


def _default_correlation_matrix() -> list[list[float]]:
    n = len(BIAS_PARAMS)
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def test_asymmetric_correlation_raises() -> None:
    matrix = _default_correlation_matrix()
    matrix[0][1] = 0.5
    matrix[1][0] = 0.4
    with pytest.raises(ValidationError):
        BiasesConfig(correlation=tuple(tuple(row) for row in matrix))


def test_non_unit_diagonal_correlation_raises() -> None:
    matrix = _default_correlation_matrix()
    matrix[0][0] = 0.9
    with pytest.raises(ValidationError):
        BiasesConfig(correlation=tuple(tuple(row) for row in matrix))


def test_wrong_shape_correlation_raises() -> None:
    n = len(BIAS_PARAMS)
    matrix = [[1.0 if i == j else 0.0 for j in range(n - 1)] for i in range(n - 1)]
    with pytest.raises(ValidationError):
        BiasesConfig(correlation=tuple(tuple(row) for row in matrix))


def test_non_positive_definite_correlation_raises() -> None:
    matrix = _default_correlation_matrix()
    matrix[0][1] = matrix[1][0] = 1.5
    with pytest.raises(ValidationError):
        BiasesConfig(correlation=tuple(tuple(row) for row in matrix))


_EXPECTED_CORRELATION_PAIRS: dict[tuple[str, str], float] = {
    ("loss_aversion_lambda", "disposition_ratio"): 0.42,
    ("overconfidence_coverage", "herding_weight"): -0.31,
    ("loss_aversion_lambda", "herding_weight"): -0.08,
    ("extrapolation_theta", "herding_weight"): 0.05,
    ("exit_deficiency", "disposition_ratio"): 0.30,
    ("exit_deficiency", "loss_aversion_lambda"): 0.126,
}


def test_correlation_pairs_match_binding_defaults() -> None:
    matrix = Config().biases.correlation
    n = len(BIAS_PARAMS)
    expected = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for (name_a, name_b), value in _EXPECTED_CORRELATION_PAIRS.items():
        i, j = BIAS_PARAMS.index(name_a), BIAS_PARAMS.index(name_b)
        expected[i][j] = value
        expected[j][i] = value
    for i in range(n):
        for j in range(n):
            assert matrix[i][j] == pytest.approx(expected[i][j])


def test_exit_deficiency_is_independent_of_loss_aversion_given_disposition() -> None:
    matrix = Config().biases.correlation
    exit_d, loss, disp = (
        BIAS_PARAMS.index(name)
        for name in ("exit_deficiency", "loss_aversion_lambda", "disposition_ratio")
    )
    partial_numerator = matrix[exit_d][loss] - matrix[exit_d][disp] * matrix[disp][loss]
    assert partial_numerator == pytest.approx(0.0, abs=1e-12)


def test_conviction_size_miscalibration_is_uncorrelated_with_every_other_bias() -> None:
    matrix = Config().biases.correlation
    row = BIAS_PARAMS.index("conviction_size_miscalibration")
    assert all(matrix[row][j] == 0.0 for j in range(len(BIAS_PARAMS)) if j != row)


_EXPECTED_BIAS_PARAMS: dict[str, tuple] = {
    "loss_aversion_lambda": (
        LogNormalSpec(median=1.1, sigma=0.10),
        LogNormalSpec(median=2.0, sigma=0.25, lo=1.5),
        True,
        Regime.RISK_OFF,
    ),
    "disposition_ratio": (
        LogNormalSpec(median=1.0, sigma=0.08),
        LogNormalSpec(median=1.2, sigma=0.15, lo=1.2),
        True,
        Regime.RISK_OFF,
    ),
    "anchoring_rho": (
        BetaSpec(a=2, b=12),
        BetaSpec(a=9, b=12),
        True,
        Regime.RANGE,
    ),
    "extrapolation_theta": (
        BetaSpec(a=2, b=10),
        BetaSpec(a=12, b=8),
        True,
        Regime.RISK_ON,
    ),
    "herding_weight": (
        BetaSpec(a=2, b=10),
        BetaSpec(a=7, b=5),
        True,
        Regime.RISK_ON,
    ),
    "overconfidence_coverage": (
        BetaSpec(a=16, b=4),
        BetaSpec(a=4, b=6),
        False,
        None,
    ),
    "conviction_size_miscalibration": (
        BetaSpec(a=2, b=10),
        BetaSpec(a=5, b=5),
        True,
        None,
    ),
    "exit_deficiency": (
        BetaSpec(a=1, b=15),
        BetaSpec(a=4, b=5),
        True,
        None,
    ),
}


@pytest.mark.parametrize("name", BIAS_PARAMS)
def test_bias_param_matches_binding_defaults(name: str) -> None:
    spec = Config().biases.params[name]
    neutral, active, higher_is_stronger, cluster_regime = _EXPECTED_BIAS_PARAMS[name]
    assert spec.neutral == neutral
    assert spec.active == active
    assert spec.higher_is_stronger is higher_is_stronger
    assert spec.cluster_regime == cluster_regime


def test_bias_spec_empty_note_raises() -> None:
    with pytest.raises(ValidationError):
        BiasSpec(
            neutral=BetaSpec(a=2, b=10),
            active=BetaSpec(a=5, b=5),
            higher_is_stronger=True,
            cluster_regime=None,
            basis="guess",
            note="",
        )


def test_dump_with_basis_covers_every_leaf() -> None:
    config = Config()
    rows = config.dump_with_basis()
    paths = {row.path for row in rows}

    for row in rows:
        assert row.basis in ("sourced", "design", "guess")
        assert isinstance(row.note, str)
        assert row.note.strip()

    assert "biases.p_active" in paths
    assert "biases.params.herding_weight" in paths
    assert "market.families.credit.asymmetry" in paths
    assert "market.events.earnings" in paths

    expected_paths: set[str] = set()

    def _has_basis_field(value: object) -> bool:
        return isinstance(value, BaseModel) and "basis" in type(value).model_fields

    def walk(model: BaseModel, prefix: str) -> None:
        for name, field in type(model).model_fields.items():
            value = getattr(model, name)
            path = f"{prefix}.{name}" if prefix else name
            extra = field.json_schema_extra
            if isinstance(extra, dict) and "basis" in extra:
                expected_paths.add(path)
            elif _has_basis_field(value):
                expected_paths.add(path)
            elif (
                isinstance(value, dict)
                and value
                and all(_has_basis_field(v) for v in value.values())
            ):
                for key in value:
                    expected_paths.add(f"{path}.{key}")
            elif isinstance(value, BaseModel):
                walk(value, path)
            else:
                pytest.fail(f"leaf at '{path}' has no basis")

    walk(config, "")
    assert paths == expected_paths


@pytest.mark.parametrize("revive_first", [38, 40])
def test_revive_weeks_must_start_after_dormant_weeks_end(revive_first: int) -> None:
    overrides = {"drift": {"dormant_weeks": [32, 40], "revive_weeks": [revive_first, 48]}}
    with pytest.raises(ValidationError, match="revive_weeks must start after dormant_weeks"):
        Config.model_validate(overrides)


def test_revive_weeks_starting_the_week_after_dormant_weeks_is_accepted() -> None:
    config = Config.model_validate({"drift": {"dormant_weeks": [32, 40], "revive_weeks": [41, 48]}})
    assert config.drift.revive_weeks == (41, 48)


def test_market_defaults_spot_check() -> None:
    market = Config().market
    assert market.universe.n_equities == 80
    assert market.universe.n_sectors == 10
    assert market.universe.n_credit_issuers == 48
    assert market.families.credit.asymmetry == 1.1
    assert market.families.commodity.group_share == 0.6
    assert market.levels.yield_floor_pct == 0.0
    assert market.events[EventType.EARNINGS].jitter_days == 5
    assert market.consensus.report_weekday == 4
    assert market.check.corr_tolerance_se == 4.0
    assert market.boundary_weeks == (16, 30)
    assert market.burn_in_days == 60
    assert market.seeds["A"] == (Regime.RANGE, Regime.RISK_OFF, Regime.RISK_ON)
    assert market.regimes.params(Regime.RISK_OFF) == RegimeParams(
        driver_mean=-0.09, vol_multiplier=1.6, mean_reversion_kappa=0.0
    )


def test_market_seed_with_repeated_regime_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"market": {"seeds": {"A": ["range", "range", "risk_on"]}}})


def test_market_seed_with_two_regimes_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"market": {"seeds": {"A": ["range", "risk_on"]}}})


def test_population_market_seed_not_in_market_seeds_raises() -> None:
    with pytest.raises(ValidationError, match="D"):
        Config.model_validate({"population": {"market_seeds": ["A", "B", "D"]}})


@pytest.mark.parametrize(
    "boundary_weeks",
    [(30, 16), (0, 16), (16, 52)],
)
def test_market_boundary_weeks_out_of_range_raises(boundary_weeks: tuple[int, int]) -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"market": {"boundary_weeks": boundary_weeks}, "calendar": {"n_weeks": 52}}
        )


def test_credit_band_shares_not_summing_to_one_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "market": {
                    "universe": {
                        "credit_band_shares": {
                            "AA": 0.15,
                            "A": 0.25,
                            "BBB": 0.30,
                            "BB": 0.10,
                            "B": 0.10,
                        }
                    }
                }
            }
        )


def test_commodities_count_over_table_size_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"market": {"universe": {"commodities": {"energy": 7}}}})


def test_unknown_fx_pair_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"market": {"universe": {"fx_pairs": ["EURUSD", "NOKUSD"]}}})


def test_vol_tolerance_not_positive_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"market": {"check": {"vol_tolerance": 0}}})


def test_corr_tolerance_se_negative_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"market": {"check": {"corr_tolerance_se": -1}}})


def test_grid_event_with_excessive_jitter_raises() -> None:
    with pytest.raises(ValidationError):
        EventSpec(
            per_year=4,
            jump_size=0.05,
            placement="grid",
            jitter_days=50,
            basis="sourced",
            note="test",
        )


def test_events_missing_macro_print_raises() -> None:
    events = {k: v for k, v in Config().market.events.items() if k != EventType.MACRO_PRINT}
    with pytest.raises(ValidationError):
        MarketConfig(events=events)


def test_market_problem_in_yaml_raises_config_error(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"market": {"universe": {"commodities": {"energy": 7}}}})
    with pytest.raises(ConfigError, match="commodities"):
        load_config(path)


def test_market_yaml_override_keeps_sibling_defaults(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"market": {"universe": {"n_equities": 20}}})
    config = load_config(path)
    assert config.market.universe.n_equities == 20
    default = Config().market.universe
    assert config.market.universe.n_sectors == default.n_sectors
    assert config.market.universe.n_credit_issuers == default.n_credit_issuers
    assert config.market.families == Config().market.families
