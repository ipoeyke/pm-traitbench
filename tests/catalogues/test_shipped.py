"""Tests for the catalogue YAML files shipped inside the package."""

import re
from importlib import resources
from typing import Any

import pytest
import yaml

from pm_traitbench.catalogues.loader import (
    CLOSER_PHRASES,
    check_catalogue,
    check_dialogue_catalogue,
    load_catalogue,
    render_signpost,
    render_template,
    render_thesis,
)
from pm_traitbench.catalogues.models import ADAPTER_FORMS, Catalogue, RuleVariant
from pm_traitbench.enums import AssetClass, Expression

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


def _assert_clean(rendered: str, context: Any) -> None:
    assert "{" not in rendered and "}" not in rendered, context
    assert "None" not in rendered, context
    assert "_" not in rendered, context
    assert "  " not in rendered, context


# Slot values as the engine passes them: levels arrive rendered (a bare price for a
# price-quoted outright, else the series level with its unit), relative moves as magnitudes.
_ENGINE_LEVELS: dict[AssetClass, tuple[str, ...]] = {
    AssetClass.EQUITIES: ("55.00", "-4.55%"),
    AssetClass.RATES_CREDIT: ("420.1bp", "35.0bp"),
    AssetClass.COMMODITIES: ("15823.82", "1.33%"),
}
_ENGINE_RELATIVE: dict[AssetClass, tuple[str, str]] = {
    AssetClass.EQUITIES: ("7.15%", "information technology"),
    AssetClass.RATES_CREDIT: ("30.2bp", "the BBB band"),
    AssetClass.COMMODITIES: ("4.55%", "industrial metals"),
}
_ENGINE_THESIS: dict[AssetClass, dict[Expression, dict[str, str]]] = {
    AssetClass.EQUITIES: {
        Expression.OUTRIGHT: {"name": "Equity 0001", "entry": "21.84", "target": "34.09"},
        Expression.PAIR: {"name": "Equity 0001 versus Equity 0002", "entry": "3.10%"},
    },
    AssetClass.RATES_CREDIT: {
        Expression.OUTRIGHT: {"name": "USD 10Y", "entry": "420.1bp", "target": "380.0bp"},
        Expression.CURVE: {"name": "USD 2Y versus 10Y", "entry": "35.0bp"},
    },
    AssetClass.COMMODITIES: {
        Expression.OUTRIGHT: {"name": "silver", "entry": "31.20", "target": "37.26"},
        Expression.CALENDAR_SPREAD: {"name": "crude M1 versus M5", "entry": "1.33%"},
    },
}


def _assert_engine_text(rendered: str, context: Any) -> None:
    _assert_clean(rendered, context)
    assert not re.search(r"\d\.\d{3}", rendered), context


def _assert_every_engine_template_renders_cleanly(catalogue: Catalogue) -> None:
    """Render every signpost, thesis and outcome template with engine-realistic slots."""
    for asset_class in ADAPTER_FORMS:
        signpost = catalogue.signposts[asset_class]
        for template in signpost.event:
            _assert_clean(
                render_signpost(template, event="rating_downgrade"), (asset_class, "event")
            )
        for level in _ENGINE_LEVELS[asset_class]:
            for template in signpost.level:
                rendered = render_signpost(template, level=level, window=5)
                _assert_engine_text(rendered, (asset_class, "level", template))
        magnitude, peer = _ENGINE_RELATIVE[asset_class]
        for template in signpost.relative:
            rendered = render_signpost(template, level=magnitude, peer=peer)
            _assert_engine_text(rendered, (asset_class, "relative", template))
            assert "-" not in rendered, (asset_class, template)
        for expression in ADAPTER_FORMS[asset_class]:
            slots = _ENGINE_THESIS[asset_class][expression]
            for template in catalogue.theses.theses[asset_class][expression]:
                rendered = render_thesis(
                    template,
                    name=slots["name"],
                    entry=slots["entry"],
                    target=slots.get("target", slots["entry"]),
                    move="+5.00",
                    unit="bp" if asset_class == AssetClass.RATES_CREDIT else "pct",
                    horizon=20,
                    side="long",
                )
                _assert_engine_text(rendered, (asset_class, expression, template))
    for kind, templates in catalogue.theses.outcomes.items():
        for closer in CLOSER_PHRASES:
            for template in templates:
                rendered = render_thesis(template, pnl=3.2, unit="pct", closer=closer)
                _assert_clean(rendered, (kind, closer, template))


def test_packaged_yaml_files_exist() -> None:
    base = resources.files("pm_traitbench.catalogues")
    for name in (
        "preferences.yaml",
        "rules.yaml",
        "mandates.yaml",
        "self_descriptions.yaml",
        "signposts.yaml",
        "theses.yaml",
        "stances.yaml",
        "voices.yaml",
        "avoid.yaml",
    ):
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


def test_no_preference_asks_the_copilot_to_start_a_conversation() -> None:
    # The copilot only ever replies, so a value needs a moment inside a session to act on.
    push_verbs = ("send ", "notify ", "ping ", "alert me", "message me", "email ")
    catalogue = load_catalogue()
    for entry in catalogue.preferences:
        for value in entry.values:
            assert not any(verb in value.lower() for verb in push_verbs), (entry.param, value)


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


def _iter_stance_line_lists(value: Any) -> list[list[str]]:
    """Collect every list of stance lines in a loaded stance bank.

    A leaf maps "all" or an asset-class key to a list of line strings; anything else
    (a bias's `revealed`, keyed one level deeper by engine action pattern) is a dict of
    further dicts, so the recursion stops exactly at the line lists themselves.
    """
    if not isinstance(value, dict):
        return []
    if value and all(isinstance(v, list) for v in value.values()):
        return list(value.values())
    collected: list[list[str]] = []
    for v in value.values():
        collected.extend(_iter_stance_line_lists(v))
    return collected


def test_stance_bank_lines_are_lowercase_no_period_and_2_to_4_per_key() -> None:
    data = _load_shipped_yaml("stances.yaml")
    for lines in _iter_stance_line_lists(data):
        assert 2 <= len(lines) <= 4, lines
        for line in lines:
            assert not line.endswith("."), line
            first_char = line[0]
            if first_char != "{":
                assert first_char == first_char.lower(), line


@pytest.mark.parametrize(
    "name",
    [
        "preferences.yaml",
        "rules.yaml",
        "mandates.yaml",
        "self_descriptions.yaml",
        "signposts.yaml",
        "theses.yaml",
        "stances.yaml",
        "voices.yaml",
        "avoid.yaml",
    ],
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


@pytest.mark.parametrize("name", ["signposts.yaml", "theses.yaml"])
def test_engine_template_banks_never_name_a_bias(name: str) -> None:
    data = _load_shipped_yaml(name)
    for text in _iter_strings(data):
        lowered = text.lower()
        for word in _BIAS_WORDS:
            assert word not in lowered, (name, word, text)


def test_every_rule_template_renders_cleanly() -> None:
    catalogue = load_catalogue()
    _assert_every_rule_template_renders_cleanly(catalogue)


def test_every_engine_template_renders_cleanly() -> None:
    catalogue = load_catalogue()
    _assert_every_engine_template_renders_cleanly(catalogue)


def test_every_direct_asset_class_has_all_three_signpost_kinds() -> None:
    catalogue = load_catalogue()
    for asset_class in ADAPTER_FORMS:
        signpost = catalogue.signposts[asset_class]
        assert len(signpost.event) >= 2
        assert len(signpost.level) >= 2
        assert len(signpost.relative) >= 2


def test_every_adapter_forms_cell_has_a_thesis_template() -> None:
    catalogue = load_catalogue()
    for asset_class, expressions in ADAPTER_FORMS.items():
        for expression in expressions:
            templates = catalogue.theses.theses[asset_class][expression]
            assert len(templates) >= 2


def test_equities_and_commodities_outright_theses_have_no_unit_on_entry_or_target() -> None:
    # Entry and target for these cells are a quoted price, not a percentage.
    catalogue = load_catalogue()
    for asset_class in (AssetClass.EQUITIES, AssetClass.COMMODITIES):
        for template in catalogue.theses.theses[asset_class][Expression.OUTRIGHT]:
            assert "{entry}{unit}" not in template, (asset_class, template)
            assert "{target}{unit}" not in template, (asset_class, template)


def test_shipped_dialogue_catalogue_passes_its_checks() -> None:
    catalogue = load_catalogue()
    check_dialogue_catalogue(catalogue)
    assert len(catalogue.voices) == 8
