"""Tests for the catalogue YAML files shipped inside the package."""

import re
from importlib import resources
from typing import Any

import pytest
import yaml

from pm_traitbench.catalogues.loader import check_catalogue, load_catalogue, render_template
from pm_traitbench.catalogues.models import Catalogue, RuleVariant
from pm_traitbench.enums import AssetClass

_REPEATED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

_N_PREFERENCES_MAX = 8
_BANNED_WORDS = ("really", "actually", "truly", "genuinely")
_BIAS_WORDS = (
    "bias",
    "anchor",
    "herd",
    "overconfiden",
    "loss avers",
    "disposition",
    "extrapolat",
    "miscalibrat",
)

_EXPECTED_SUB_STYLES = {
    AssetClass.EQUITIES: (
        ("equity_long_short", "pct_nav", "cash"),
        ("long_only_equity", "pct_nav", "broad_equity_index"),
    ),
    AssetClass.RATES_CREDIT: (
        ("sovereign_rates", "dv01", "sovereign_index"),
        ("long_short_credit", "dv01", "agg"),
    ),
    AssetClass.COMMODITIES: (
        ("commodity_futures_directional", "contracts", "commodity_index"),
        ("curve_and_spread", "contracts", "cash"),
    ),
    AssetClass.MULTI_ASSET: (
        ("global_macro", "pct_nav_sleeve", "cash"),
        ("balanced_allocation", "pct_nav_sleeve", "sixty_forty"),
    ),
}

_EXPECTED_RULE_PARAMS = {
    "stop_loss",
    "trim_at_target",
    "no_add_before_trigger",
    "min_holding_period",
    "exclusion",
    "max_positions",
    "roll_before_expiry",
}


def _iter_strings(value: Any) -> list[str]:
    """Recursively collect every string found in a loaded YAML structure."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for k, v in value.items():
            if isinstance(k, str):
                strings.append(k)
            strings.extend(_iter_strings(v))
        return strings
    if isinstance(value, list | tuple):
        strings = []
        for item in value:
            strings.extend(_iter_strings(item))
        return strings
    return []


def _load_shipped_yaml(name: str) -> dict[str, Any]:
    text = resources.files("pm_traitbench.catalogues").joinpath(name).read_text()
    return yaml.safe_load(text)


def _representative_levels(variant: RuleVariant, is_cap: bool) -> tuple[float | str, ...]:
    """Levels worth rendering for one rule variant: min, max, and each choice."""
    if is_cap:
        return (10.0,)
    if variant.level_choices:
        return variant.level_choices
    return (variant.level_min, variant.level_max)


def _assert_every_rule_template_renders_cleanly(catalogue: Catalogue) -> None:
    """Render every template of every rule variant and assert it reads as clean PM text."""
    for entry in (catalogue.rules.mandate_cap, *catalogue.rules.entries):
        is_cap = entry is catalogue.rules.mandate_cap
        for variant in entry.variants:
            for level in _representative_levels(variant, is_cap):
                for template in variant.templates:
                    rendered = render_template(template, level, variant.unit)
                    context = (entry.param, variant.asset_class, template, rendered)
                    assert "{" not in rendered and "}" not in rendered, context
                    assert "pct" not in rendered.lower(), context
                    assert "None" not in rendered, context
                    assert "  " not in rendered, context
                    assert "_" not in rendered, context
                    assert rendered == rendered.strip(), context
                    assert not _REPEATED_WORD.search(rendered), context


def test_packaged_yaml_files_exist() -> None:
    base = resources.files("pm_traitbench.catalogues")
    for name in ("preferences.yaml", "rules.yaml", "mandates.yaml", "self_descriptions.yaml"):
        assert base.joinpath(name).is_file()


def test_load_catalogue_succeeds_from_packaged_files() -> None:
    catalogue = load_catalogue()
    check_catalogue(catalogue, list(AssetClass), n_preferences_max=_N_PREFERENCES_MAX)


def test_preference_param_count_between_28_and_34() -> None:
    catalogue = load_catalogue()
    assert 28 <= len(catalogue.preferences) <= 34


def test_every_preference_has_2_to_4_values() -> None:
    catalogue = load_catalogue()
    for entry in catalogue.preferences:
        assert 2 <= len(entry.values) <= 4, entry.param


def test_mandate_sub_style_table_matches_exactly() -> None:
    catalogue = load_catalogue()
    for asset_class, expected in _EXPECTED_SUB_STYLES.items():
        styles = catalogue.sub_styles[asset_class]
        actual = tuple((style.name, style.risk_unit, style.benchmark) for style in styles)
        assert actual == expected


def test_rule_params_equal_expected_set_plus_cap() -> None:
    catalogue = load_catalogue()
    assert catalogue.rules.mandate_cap.param == "max_risk_pct"
    actual_params = {entry.param for entry in catalogue.rules.entries}
    assert actual_params == _EXPECTED_RULE_PARAMS


def test_self_description_phrasings_are_lowercase_no_period_3_to_8_words() -> None:
    catalogue = load_catalogue()
    for param, phrasings in catalogue.self_descriptions.items():
        for fragment in (*phrasings.agree, *phrasings.contradict):
            assert fragment == fragment.lower(), (param, fragment)
            assert not fragment.endswith("."), (param, fragment)
            word_count = len(fragment.split())
            assert 3 <= word_count <= 8, (param, fragment, word_count)


@pytest.mark.parametrize(
    "name",
    ["preferences.yaml", "rules.yaml", "mandates.yaml", "self_descriptions.yaml"],
)
def test_no_em_dash_or_banned_words_in_any_shipped_file(name: str) -> None:
    data = _load_shipped_yaml(name)
    for text in _iter_strings(data):
        assert "—" not in text, (name, text)
        lowered = text.lower()
        for banned in _BANNED_WORDS:
            assert not re.search(rf"\b{banned}\b", lowered), (name, banned, text)


def test_self_descriptions_never_name_a_bias() -> None:
    catalogue = load_catalogue()
    for param, phrasings in catalogue.self_descriptions.items():
        for fragment in (*phrasings.agree, *phrasings.contradict):
            lowered = fragment.lower()
            for word in _BIAS_WORDS:
                assert word not in lowered, (param, word, fragment)


def test_every_rule_template_renders_cleanly() -> None:
    catalogue = load_catalogue()
    _assert_every_rule_template_renders_cleanly(catalogue)
