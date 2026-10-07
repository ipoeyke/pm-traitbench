"""Tests for catalogue loading and consistency checks."""

from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import yaml

from pm_traitbench.catalogues.loader import (
    banned_words_in,
    check_catalogue,
    check_gate2_catalogue,
    load_catalogue,
    matched_params,
    render_signpost,
    render_template,
    render_thesis,
)
from pm_traitbench.catalogues.models import (
    Catalogue,
    PreferenceGroup,
    RuleEntry,
    RuleVariant,
)
from pm_traitbench.enums import Action, AssetClass, Op
from pm_traitbench.errors import CatalogueError

_CATALOGUE_FILES = (
    "preferences.yaml",
    "rules.yaml",
    "mandates.yaml",
    "self_descriptions.yaml",
    "signposts.yaml",
    "theses.yaml",
    "stances.yaml",
    "voices.yaml",
    "avoid.yaml",
    "bias_definitions.yaml",
    "probes.yaml",
)
_ASSET_CLASSES = list(AssetClass)
_N_PREFERENCES_MAX = 8


def _copy_shipped(tmp_path: Path) -> Path:
    base = resources.files("pm_traitbench.catalogues")
    for name in _CATALOGUE_FILES:
        (tmp_path / name).write_text(base.joinpath(name).read_text())
    return tmp_path


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _check(catalogue: Catalogue) -> None:
    check_catalogue(catalogue, _ASSET_CLASSES, n_preferences_max=_N_PREFERENCES_MAX)


# --- catalogue basics ---


def test_catalogue_loads_and_passes_checks(catalogue: Catalogue) -> None:
    _check(catalogue)


def test_preferences_for_filters_by_asset_class_and_keeps_order(
    catalogue: Catalogue,
) -> None:
    # Build a small catalogue where every preference applies everywhere except
    # one, which is narrowed to a single asset class.
    universal = [e for e in catalogue.preferences if len(e.asset_classes) == len(AssetClass)]
    base = universal[:3]
    restricted = base[-1].model_copy(update={"asset_classes": (AssetClass.RATES_CREDIT,)})
    small = catalogue.model_copy(update={"preferences": (*base[:-1], restricted)})

    all_params = [entry.param for entry in small.preferences]

    equities = small.preferences_for(AssetClass.EQUITIES)
    assert [entry.param for entry in equities] == [p for p in all_params if p != restricted.param]
    assert all(AssetClass.EQUITIES in entry.asset_classes for entry in equities)

    rates_credit = small.preferences_for(AssetClass.RATES_CREDIT)
    assert [entry.param for entry in rates_credit] == all_params


def test_entries_for_filters_by_asset_class_and_keeps_order(catalogue: Catalogue) -> None:
    all_params = [entry.param for entry in catalogue.rules.entries]

    commodities = catalogue.rules.entries_for(AssetClass.COMMODITIES)
    assert [entry.param for entry in commodities] == all_params
    assert all(AssetClass.COMMODITIES in entry.asset_classes for entry in commodities)

    equities = catalogue.rules.entries_for(AssetClass.EQUITIES)
    assert [entry.param for entry in equities] == [
        p for p in all_params if p != "roll_before_expiry"
    ]


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


def test_stop_loss_variant_for_prefers_matching_sub_style(catalogue: Catalogue) -> None:
    stop_loss = next(e for e in catalogue.rules.entries if e.param == "stop_loss")
    sub_styles = catalogue.sub_styles[AssetClass.RATES_CREDIT]
    fields = set()
    for sub_style in sub_styles:
        variant = stop_loss.variant_for(AssetClass.RATES_CREDIT, sub_style.name)
        assert not variant.sub_styles or sub_style.name in variant.sub_styles
        fields.add(variant.field)
    if len(fields) < 2:
        pytest.skip("catalogue has no two rates_credit sub-styles with different stop_loss fields")


# --- render_template ---


def test_render_template_formats_float_level_and_unit() -> None:
    assert render_template("stop at {level}{unit}", -15.0, "%") == "stop at -15%"


def test_render_template_with_none_unit_has_no_literal_none() -> None:
    result = render_template("hold for {level} {unit}", 20.0, None)
    assert "None" not in result
    assert result == "hold for 20 "


def test_render_template_with_string_level() -> None:
    assert render_template("exclude {level}", "energy", None) == "exclude energy"


def test_render_template_with_string_level_replaces_underscores_with_spaces() -> None:
    assert render_template("exclude {level}", "ccc_and_below", None) == "exclude ccc and below"
    assert (
        render_template("exclude {level}", "emerging_markets", None) == "exclude emerging markets"
    )


def test_render_template_with_float_level_ignores_underscore_handling() -> None:
    assert render_template("stop at {level}%", -15.0, None) == "stop at -15%"


def test_render_template_with_int_level() -> None:
    assert render_template("cap at {level}", 4, None) == "cap at 4"


# --- load_catalogue error paths ---


def test_missing_file_raises_catalogue_error(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    (tmp_path / "mandates.yaml").unlink()
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_malformed_yaml_raises_catalogue_error(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    (tmp_path / "rules.yaml").write_text("mandate_cap: [this is not, valid: yaml\n")
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_unknown_top_level_key_raises_catalogue_error(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["bogus_top_level"] = 1
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_unknown_nested_key_raises_catalogue_error(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["preferences"][0]["bogus_field"] = 1
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


# --- level-shape invariant (mandate_cap vs. other rule entries) ---


def test_mandate_cap_variant_with_level_choices_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    data["mandate_cap"]["variants"][0]["level_choices"] = [5]
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_non_cap_rule_variant_without_range_or_choices_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    trim = next(e for e in data["entries"] if e["param"] == "trim_at_target")
    del trim["variants"][0]["level_choices"]
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_partial_level_range_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
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
    _copy_shipped(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["preferences"] = [e for e in data["preferences"] if e["group"] != "communication"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="communication"):
        _check(catalogue)


def test_check_too_few_applicable_preferences_raises(catalogue: Catalogue) -> None:
    one_per_group: list = []
    seen_groups: set = set()
    for entry in catalogue.preferences_for(AssetClass.EQUITIES):
        if entry.group not in seen_groups:
            one_per_group.append(entry)
            seen_groups.add(entry.group)
    assert seen_groups == set(PreferenceGroup)
    small_catalogue = catalogue.model_copy(update={"preferences": tuple(one_per_group)})
    with pytest.raises(CatalogueError, match="preferences"):
        check_catalogue(
            small_catalogue, [AssetClass.EQUITIES], n_preferences_max=len(one_per_group) + 1
        )


def test_check_duplicate_preference_param_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    duplicate_param = data["preferences"][0]["param"]
    data["preferences"][1]["param"] = duplicate_param
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match=duplicate_param):
        _check(catalogue)


def test_check_preference_value_that_looks_like_a_number_is_allowed(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["preferences"][0]["values"][0] = "12.5"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    _check(catalogue)


def test_check_preference_empty_value_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "preferences.yaml"
    data = _load_yaml(path)
    data["preferences"][0]["values"][0] = "   "
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match=data["preferences"][0]["param"]):
        _check(catalogue)


def test_check_duplicate_rule_param_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    duplicate_param = data["entries"][0]["param"]
    data["entries"][1]["param"] = duplicate_param
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match=duplicate_param):
        _check(catalogue)


def test_check_missing_variant_for_non_mandatory_entry_does_not_raise(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    max_positions = next(e for e in data["entries"] if e["param"] == "max_positions")
    max_positions["variants"] = [
        v for v in max_positions["variants"] if v["asset_class"] != "multi_asset"
    ]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    _check(catalogue)
    multi_asset_params = {e.param for e in catalogue.rules.entries_for(AssetClass.MULTI_ASSET)}
    assert "max_positions" not in multi_asset_params


def test_check_mandatory_entry_missing_asset_class_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    stop_loss = next(e for e in data["entries"] if e["param"] == "stop_loss")
    stop_loss["variants"] = [v for v in stop_loss["variants"] if v["asset_class"] != "multi_asset"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="stop_loss.*multi_asset"):
        _check(catalogue)


def test_check_discipline_entry_missing_asset_class_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    min_holding = next(e for e in data["entries"] if e["param"] == "min_holding_period")
    min_holding["variants"] = [
        v for v in min_holding["variants"] if v["asset_class"] != "commodities"
    ]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="min_holding_period.*commodities"):
        _check(catalogue)


def test_check_rule_entry_with_no_variants_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    max_positions = next(e for e in data["entries"] if e["param"] == "max_positions")
    max_positions["variants"] = []
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="max_positions"):
        _check(catalogue)


def test_check_no_mandatory_rule_entry_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    for entry in data["entries"]:
        entry["mandatory"] = False
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="mandatory"):
        _check(catalogue)


def test_check_no_discipline_rule_entry_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    for entry in data["entries"]:
        entry["discipline"] = False
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="discipline"):
        _check(catalogue)


def test_check_template_with_unknown_slot_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "rules.yaml"
    data = _load_yaml(path)
    data["mandate_cap"]["variants"][0]["templates"][0] = "cap at {level}% of {bogus}"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="bogus"):
        _check(catalogue)


def test_check_level_min_not_less_than_level_max_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
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
    _copy_shipped(tmp_path)
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
    _copy_shipped(tmp_path)
    path = tmp_path / "mandates.yaml"
    data = _load_yaml(path)
    data["sub_styles"]["rates_credit"] = []
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="rates_credit"):
        _check(catalogue)


def test_check_duplicate_sub_style_name_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "mandates.yaml"
    data = _load_yaml(path)
    data["sub_styles"]["equities"][1]["name"] = data["sub_styles"]["equities"][0]["name"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="equities"):
        _check(catalogue)


def test_check_self_descriptions_missing_bias_param_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "self_descriptions.yaml"
    data = _load_yaml(path)
    del data["self_descriptions"]["exit_deficiency"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="missing.*exit_deficiency"):
        _check(catalogue)


def test_check_self_descriptions_extra_key_raises_naming_it(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "self_descriptions.yaml"
    data = _load_yaml(path)
    data["self_descriptions"]["not_a_bias_param"] = data["self_descriptions"]["exit_deficiency"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="extra.*not_a_bias_param"):
        _check(catalogue)


def test_check_self_descriptions_too_few_agree_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "self_descriptions.yaml"
    data = _load_yaml(path)
    data["self_descriptions"]["exit_deficiency"]["agree"] = ["only one phrasing"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="exit_deficiency"):
        _check(catalogue)


# --- signposts and theses ---


def test_signpost_cell_with_one_template_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "signposts.yaml"
    data = _load_yaml(path)
    data["signposts"]["equities"]["level"] = data["signposts"]["equities"]["level"][:1]
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError, match="signposts.equities.level"):
        load_catalogue(tmp_path)


def test_thesis_template_with_unknown_slot_raises_naming_the_cell(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "theses.yaml"
    data = _load_yaml(path)
    data["theses"]["equities"]["outright"][0] += " {foo}"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="equities.*outright.*foo"):
        _check(catalogue)


def test_thesis_template_without_side_slot_raises_naming_the_cell(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "theses.yaml"
    data = _load_yaml(path)
    data["theses"]["commodities"]["outright"][0] = "{name} at {entry}, target {target}"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="commodities.*outright.*side"):
        _check(catalogue)


def test_outcomes_key_draw_is_rejected(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "theses.yaml"
    data = _load_yaml(path)
    data["outcomes"]["draw"] = data["outcomes"].pop("open")
    _dump_yaml(path, data)
    with pytest.raises(CatalogueError, match="outcomes keys must be exactly"):
        load_catalogue(tmp_path)


def test_render_signpost_fills_every_slot_and_replaces_event_underscores() -> None:
    rendered = render_signpost(
        "if {event} passes and it holds under {level} for {window} sessions, "
        "versus {peer}, i'm out",
        level="4.55%",
        window=5,
        event="rating_downgrade",
        peer="the AA band",
    )
    assert "rating downgrade" in rendered
    assert "under 4.55% for 5 sessions" in rendered
    assert "{" not in rendered and "}" not in rendered


def test_render_thesis_formats_signed_move_as_one_decimal() -> None:
    assert render_thesis("{move}{unit}", move=3.0, unit="pct") == "+3.0%"
    assert render_thesis("{move}{unit}", move=-3.0, unit="pct") == "-3.0%"


def test_render_thesis_formats_signed_pnl_as_one_decimal() -> None:
    assert render_thesis("{pnl}{unit}", pnl=12.34, unit="bp") == "+12.3bp"


def test_render_thesis_unit_pct_renders_as_percent_sign() -> None:
    assert render_thesis("{target}{unit}", target=103.5, unit="pct") == "103.5%"


def test_render_thesis_unit_none_renders_as_no_suffix() -> None:
    assert render_thesis("{move}{unit}", move=5.2, unit=None) == "+5.2"


def test_render_thesis_fills_side_slot() -> None:
    assert (
        render_thesis("{side} {name}", side="steepener", name="2Y versus 10Y")
        == "steepener 2Y versus 10Y"
    )


def test_render_thesis_maps_closer_to_a_phrase() -> None:
    assert render_thesis("out on {closer}", closer="stop") == "out on the stop"
    assert render_thesis("out on {closer}", closer="horizon_end") == "out on the horizon end"


def test_render_thesis_unknown_closer_raises_catalogue_error() -> None:
    with pytest.raises(CatalogueError, match="bogus"):
        render_thesis("out on {closer}", closer="bogus")


# --- check_dialogue_catalogue: voice bank and forbidden-behaviour catalogue ---


def test_avoid_missing_a_bias_param_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "avoid.yaml"
    data = _load_yaml(path)
    del data["biases"]["exit_deficiency"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="missing.*exit_deficiency"):
        _check(catalogue)


def test_avoid_missing_a_preference_param_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "avoid.yaml"
    data = _load_yaml(path)
    del data["preferences"]["positioning_context"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="missing.*positioning_context"):
        _check(catalogue)


def test_avoid_extra_bias_key_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "avoid.yaml"
    data = _load_yaml(path)
    data["biases"]["not_a_bias_param"] = "do not do the thing"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="extra.*not_a_bias_param"):
        _check(catalogue)


def test_avoid_blank_line_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "avoid.yaml"
    data = _load_yaml(path)
    data["preferences"]["positioning_context"] = "   "
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="positioning_context"):
        _check(catalogue)


@pytest.mark.parametrize(
    ("overlaps", "match"),
    [
        ({"not_a_param": ["disposition_ratio"]}, "overlaps key 'not_a_param'"),
        ({"register": ["not_a_param"]}, "names unknown param 'not_a_param'"),
        ({"register": ["register"]}, "'register' overlaps itself"),
        ({"register": ["hedging_language", "hedging_language"]}, "repeats a param"),
    ],
)
def test_avoid_bad_overlaps_raise(tmp_path: Path, overlaps: dict, match: str) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "avoid.yaml"
    data = _load_yaml(path)
    data["overlaps"] = overlaps
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match=match):
        _check(catalogue)


def test_voice_line_with_a_banned_word_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "voices.yaml"
    data = _load_yaml(path)
    data["voices"][0]["line"] = "quietly follows the herd on every call"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="herd"):
        _check(catalogue)


def test_voice_line_with_a_banned_word_raises_case_insensitively(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "voices.yaml"
    data = _load_yaml(path)
    data["voices"][0]["line"] = "quietly follows the HERD on every call"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="herd"):
        _check(catalogue)


def test_voice_line_naming_a_bias_param_raises(tmp_path: Path) -> None:
    # exit_deficiency has no BANNED_STANCE_WORDS stem, so this exercises only
    # the param-name check, not the separate banned-word check above.
    _copy_shipped(tmp_path)
    path = tmp_path / "voices.yaml"
    data = _load_yaml(path)
    data["voices"][0]["line"] = "explains the exit_deficiency behind every call"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="exit_deficiency"):
        _check(catalogue)


def test_voice_line_naming_a_preference_param_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "voices.yaml"
    data = _load_yaml(path)
    data["voices"][0]["line"] = "casual register, drops articles"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="register"):
        _check(catalogue)


def test_blank_voice_line_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "voices.yaml"
    data = _load_yaml(path)
    data["voices"][0]["line"] = "   "
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="v_01"):
        _check(catalogue)


def test_duplicate_voice_id_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "voices.yaml"
    data = _load_yaml(path)
    data["voices"][1]["voice_id"] = data["voices"][0]["voice_id"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="duplicate"):
        _check(catalogue)


def test_fewer_than_six_voices_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "voices.yaml"
    data = _load_yaml(path)
    data["voices"] = data["voices"][:5]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="6"):
        _check(catalogue)


def test_missing_voices_file_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    (tmp_path / "voices.yaml").unlink()
    with pytest.raises(CatalogueError):
        load_catalogue(tmp_path)


def test_matched_params_is_whole_word_and_returns_every_match() -> None:
    assert matched_params("my register is fine", ("register",)) == ("register",)
    assert matched_params("registered", ("register",)) == ()
    assert matched_params("loss aversion lambda", ("loss_aversion_lambda",)) == (
        "loss_aversion_lambda",
    )
    assert matched_params("Pushback_Style and REGISTER", ("register", "pushback_style")) == (
        "register",
        "pushback_style",
    )


def test_banned_words_in_is_a_case_insensitive_substring_scan() -> None:
    assert banned_words_in("Loss Aversion and herding") == ("loss aversion", "herd")
    assert banned_words_in("a plain line") == ()


# --- check_gate2_catalogue: bias definition catalogue for gate 2's judge prompt ---


def test_bias_definitions_missing_a_param_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "bias_definitions.yaml"
    data = _load_yaml(path)
    del data["definitions"]["exit_deficiency"]
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="bias_definitions.*exit_deficiency"):
        check_gate2_catalogue(catalogue)


def test_bias_definitions_blank_line_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "bias_definitions.yaml"
    data = _load_yaml(path)
    data["definitions"]["exit_deficiency"] = "   "
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="exit_deficiency.*blank"):
        check_gate2_catalogue(catalogue)


def test_bias_definitions_line_naming_a_param_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "bias_definitions.yaml"
    data = _load_yaml(path)
    data["definitions"]["loss_aversion_lambda"] = "shows herding weight on every trade"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="herding_weight"):
        check_gate2_catalogue(catalogue)

    _copy_shipped(tmp_path)
    data = _load_yaml(path)
    data["definitions"]["loss_aversion_lambda"] = "shows herding_weight on every trade"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="herding_weight"):
        check_gate2_catalogue(catalogue)


def test_bias_definitions_em_dash_raises(tmp_path: Path) -> None:
    _copy_shipped(tmp_path)
    path = tmp_path / "bias_definitions.yaml"
    data = _load_yaml(path)
    data["definitions"]["exit_deficiency"] = "does not act — or acts late"
    _dump_yaml(path, data)
    catalogue = load_catalogue(tmp_path)
    with pytest.raises(CatalogueError, match="bias_definitions.*exit_deficiency"):
        check_gate2_catalogue(catalogue)
