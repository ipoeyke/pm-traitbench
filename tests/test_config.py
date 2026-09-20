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
    load_config,
)
from pm_traitbench.errors import ConfigError


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_config_builds_with_defaults() -> None:
    config = Config()
    assert config.seed.root == 20260105
    assert config.population.pilot_per_cell == 1
    assert config.population.full_per_cell == 3
    assert config.mandate.book_size_min == 50e6
    assert config.mandate.book_size_max == 2e9
    assert config.drift.bias_update_weeks == (18, 30)
    assert config.calendar.n_weeks == 52
    assert config.output.format == "default"


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


def test_dump_with_basis_covers_every_leaf() -> None:
    config = Config()
    rows = config.dump_with_basis()
    paths = {row.path for row in rows}

    for row in rows:
        assert row.basis in ("sourced", "design", "guess")
        assert isinstance(row.note, str)

    assert "biases.p_active" in paths
    assert "biases.params.herding_weight" in paths

    expected_paths: set[str] = set()

    def walk(model: BaseModel, prefix: str) -> None:
        for name, field in type(model).model_fields.items():
            value = getattr(model, name)
            path = f"{prefix}.{name}" if prefix else name
            extra = field.json_schema_extra
            if isinstance(extra, dict) and "basis" in extra:
                expected_paths.add(path)
            elif isinstance(value, BiasSpec):
                expected_paths.add(path)
            elif (
                isinstance(value, dict)
                and value
                and all(isinstance(v, BiasSpec) for v in value.values())
            ):
                for key in value:
                    expected_paths.add(f"{path}.{key}")
            elif isinstance(value, BaseModel):
                walk(value, path)
            else:
                pytest.fail(f"leaf at '{path}' has no basis")

    walk(config, "")
    assert paths == expected_paths
