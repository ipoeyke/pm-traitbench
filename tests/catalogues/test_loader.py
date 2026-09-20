"""Tests for catalogue loading and consistency checks."""

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from pm_traitbench.catalogues.loader import check_catalogue, load_catalogue, render_template
from pm_traitbench.catalogues.models import (
    Catalogue,
    RuleEntry,
    RuleVariant,
)
from pm_traitbench.enums import Action, AssetClass, Op
from pm_traitbench.errors import CatalogueError

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "catalogue"
_ASSET_CLASSES = list(AssetClass)
_N_PREFERENCES_MAX = 8


def _copy_fixture(tmp_path: Path) -> Path:
    for name in ("preferences.yaml", "rules.yaml", "mandates.yaml", "self_descriptions.yaml"):
        shutil.copy(_FIXTURE_DIR / name, tmp_path / name)
    return tmp_path


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _check(catalogue: Catalogue) -> None:
    check_catalogue(catalogue, _ASSET_CLASSES, n_preferences_max=_N_PREFERENCES_MAX)


# --- fixture catalogue basics ---


def test_fixture_catalogue_loads_and_passes_checks(fixture_catalogue: Catalogue) -> None:
    _check(fixture_catalogue)


def test_preferences_for_filters_by_asset_class_and_keeps_order(
    fixture_catalogue: Catalogue,
) -> None:
    all_params = [entry.param for entry in fixture_catalogue.preferences]

    equities = fixture_catalogue.preferences_for(AssetClass.EQUITIES)
    assert [entry.param for entry in equities] == [
        p for p in all_params if p != "curve_positioning_language"
    ]
    assert all(AssetClass.EQUITIES in entry.asset_classes for entry in equities)

    rates_credit = fixture_catalogue.preferences_for(AssetClass.RATES_CREDIT)
    assert [entry.param for entry in rates_credit] == all_params


def test_variant_for_prefers_specific_variant_and_falls_back_to_generic() -> None:
    specific = RuleVariant(
        asset_class=AssetClass.EQUITIES,
        sub_styles=("value",),
        field="x",
        op=Op.LE,
        unit=None,
        level_choices=(1,),
        action=Action.CAP,
        templates=("t",),
    )
    generic = RuleVariant(
        asset_class=AssetClass.EQUITIES,
        field="x",
        op=Op.LE,
        unit=None,
        level_choices=(1,),
        action=Action.CAP,
        templates=("t",),
    )
    entry = RuleEntry(param="p", share=0.5, variants=(specific, generic))
    assert entry.variant_for(AssetClass.EQUITIES, "value") is specific
    assert entry.variant_for(AssetClass.EQUITIES, "growth") is generic


def test_variant_for_raises_catalogue_error_when_no_variant_matches() -> None:
    entry = RuleEntry(
        param="p",
        share=0.5,
        variants=(
            RuleVariant(
                asset_class=AssetClass.EQUITIES,
                field="x",
                op=Op.LE,
                unit=None,
                level_choices=(1,),
                action=Action.CAP,
                templates=("t",),
            ),
        ),
    )
    with pytest.raises(CatalogueError, match="p"):
        entry.variant_for(AssetClass.COMMODITIES, "any")


def test_fixture_stop_loss_variant_for_prefers_matching_sub_style(
    fixture_catalogue: Catalogue,
) -> None:
    stop_loss = next(e for e in fixture_catalogue.rules.entries if e.param == "stop_loss")
    sovereign = stop_loss.variant_for(AssetClass.RATES_CREDIT, "sovereign_rates")
    credit = stop_loss.variant_for(AssetClass.RATES_CREDIT, "long_short_credit")
    assert sovereign.field == "adverse_yield_move_bp"
    assert credit.field == "adverse_spread_move_bp"


# --- render_template ---


def test_render_template_formats_float_level_and_unit() -> None:
    assert render_template("stop at {level}{unit}", -15.0, "%") == "stop at -15%"


def test_render_template_with_none_unit_has_no_literal_none() -> None:
    result = render_template("hold for {level} {unit}", 20.0, None)
    assert "None" not in result
    assert result == "hold for 20 "


def test_render_template_with_string_level() -> None:
    assert render_template("exclude {level}", "energy", None) == "exclude energy"


# --- load_catalogue error paths ---


def test_missing_file_raises_catalogue_error(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    (tmp_path / "mandates.yaml").unlink()
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_malformed_yaml_raises_catalogue_error(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    (tmp_path / "rules.yaml").write_text("mandate_cap: [this is not, valid: yaml\n")
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_unknown_top_level_key_raises_catalogue_error(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["bogus_top_level"] = 1
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_unknown_nested_key_raises_catalogue_error(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["preferences"][0]["bogus_field"] = 1
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_load_catalogue_none_raises_when_packaged_files_missing() -> None:
    with pytest.raises(CatalogueError):
        load_catalogue(None)


# --- level-shape invariant (mandate_cap vs. other rule entries) ---


def test_mandate_cap_variant_with_level_choices_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    data["mandate_cap"]["variants"][0]["level_choices"] = [5]
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_non_cap_rule_variant_without_range_or_choices_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    trim = next(e for e in data["entries"] if e["param"] == "trim_at_target")
    del trim["variants"][0]["level_choices"]
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_partial_level_range_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    stop_loss = next(e for e in data["entries"] if e["param"] == "stop_loss")
    variant = next(v for v in stop_loss["variants"] if v["asset_class"] == "equities")
    del variant["round_to"]
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


# --- check_catalogue: one test per numbered check, plus a few supplementary ones ---


def test_check_missing_preference_group_for_asset_class_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["preferences"] = [e for e in data["preferences"] if e["group"] != "communication"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="communication"):
        _check(catalogue)


def test_check_too_few_applicable_preferences_raises(fixture_catalogue: Catalogue) -> None:
    with pytest.raises(CatalogueError, match="preferences"):
        check_catalogue(fixture_catalogue, _ASSET_CLASSES, n_preferences_max=10)


def test_check_duplicate_preference_param_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    duplicate_param = data["preferences"][0]["param"]
    data["preferences"][1]["param"] = duplicate_param
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match=duplicate_param):
        _check(catalogue)


def test_check_preference_value_parsing_as_float_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["preferences"][0]["values"][0] = "12.5"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match=data["preferences"][0]["param"]):
        _check(catalogue)


def test_check_preference_empty_value_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["preferences"][0]["values"][0] = "   "
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match=data["preferences"][0]["param"]):
        _check(catalogue)


def test_check_duplicate_rule_param_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    duplicate_param = data["entries"][0]["param"]
    data["entries"][1]["param"] = duplicate_param
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match=duplicate_param):
        _check(catalogue)


def test_check_missing_rule_variant_for_asset_class_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    max_positions = next(e for e in data["entries"] if e["param"] == "max_positions")
    max_positions["variants"] = [
        v for v in max_positions["variants"] if v["asset_class"] != "multi_asset"
    ]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="max_positions"):
        _check(catalogue)


def test_check_no_mandatory_rule_entry_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    for entry in data["entries"]:
        entry["mandatory"] = False
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="mandatory"):
        _check(catalogue)


def test_check_no_discipline_rule_entry_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    for entry in data["entries"]:
        entry["discipline"] = False
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="discipline"):
        _check(catalogue)


def test_check_template_with_unknown_slot_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    data["mandate_cap"]["variants"][0]["templates"][0] = "cap at {level}% of {bogus}"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="bogus"):
        _check(catalogue)


def test_check_level_min_not_less_than_level_max_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    stop_loss = next(e for e in data["entries"] if e["param"] == "stop_loss")
    variant = next(v for v in stop_loss["variants"] if v["asset_class"] == "equities")
    variant["level_min"], variant["level_max"] = variant["level_max"], variant["level_min"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="stop_loss"):
        _check(catalogue)


def test_check_non_positive_round_to_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    stop_loss = next(e for e in data["entries"] if e["param"] == "stop_loss")
    variant = next(v for v in stop_loss["variants"] if v["asset_class"] == "equities")
    variant["round_to"] = 0
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="stop_loss"):
        _check(catalogue)


def test_check_asset_class_without_sub_style_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "mandates.yaml"
    data = _load_yaml(path)
    data["sub_styles"]["rates_credit"] = []
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="rates_credit"):
        _check(catalogue)


def test_check_duplicate_sub_style_name_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "mandates.yaml"
    data = _load_yaml(path)
    data["sub_styles"]["equities"][1]["name"] = data["sub_styles"]["equities"][0]["name"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="equities"):
        _check(catalogue)


def test_check_self_descriptions_missing_bias_param_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "self_descriptions.yaml"
    data = _load_yaml(path)
    del data["self_descriptions"]["exit_deficiency"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="self_descriptions"):
        _check(catalogue)


def test_check_self_descriptions_too_few_agree_raises(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    path = tmp_path / "self_descriptions.yaml"
    data = _load_yaml(path)
    data["self_descriptions"]["exit_deficiency"]["agree"] = ["only one phrasing"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="exit_deficiency"):
        _check(catalogue)
