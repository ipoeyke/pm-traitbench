import math

import pytest

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass, RuleScope, RuleSource
from pm_traitbench.rng import stream
from pm_traitbench.sampling.rules import sample_rules
from pm_traitbench.tables.schema import Mandate


def _mandate(asset_class: AssetClass, sub_style: str) -> Mandate:
    return Mandate(
        asset_class=asset_class,
        sub_style=sub_style,
        book_size=1e8,
        risk_unit="pct_nav",
        benchmark="cash",
    )


def _first_sub_style(catalogue: Catalogue, asset_class: AssetClass) -> str:
    """Any valid sub-style for an asset class, for tests that do not care which one."""
    return catalogue.sub_styles[asset_class][0].name


def _draw(config: Config, catalogue: Catalogue, mandate: Mandate, i: int):
    rng = stream(1, "t", i, "rules")
    return sample_rules("pm_001", mandate, config, catalogue, rng)


def test_first_rule_is_the_mandate_cap(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    for i in range(100):
        rules = _draw(config, catalogue, mandate, i)
        cap = rules[0]
        assert cap.rule_id == "r_01"
        assert cap.source == RuleSource.MANDATE
        assert cap.level in config.rules.max_risk_pct_choices


def test_self_rule_count_is_within_configured_bounds(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    for i in range(300):
        rules = _draw(config, catalogue, mandate, i)
        self_rules = rules[1:]
        assert config.rules.n_self_rules_min <= len(self_rules) <= config.rules.n_self_rules_max


def test_stop_loss_is_always_present(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    for i in range(300):
        rules = _draw(config, catalogue, mandate, i)
        assert any(r.param == "stop_loss" for r in rules[1:])


def test_at_least_one_discipline_rule_is_present(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    discipline_params = {e.param for e in catalogue.rules.entries if e.discipline}
    for i in range(300):
        rules = _draw(config, catalogue, mandate, i)
        params = {r.param for r in rules[1:]}
        assert params & discipline_params


def test_rule_ids_are_sequential_with_no_gaps(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    for i in range(100):
        rules = _draw(config, catalogue, mandate, i)
        assert [r.rule_id for r in rules] == [f"r_{n:02d}" for n in range(1, len(rules) + 1)]


def test_self_rules_are_pm_scoped_with_no_trade_idea(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    for i in range(100):
        rules = _draw(config, catalogue, mandate, i)
        for r in rules[1:]:
            assert r.scope == RuleScope.PM
            assert r.trade_idea_id is None
            assert r.source == RuleSource.SELF


def test_numeric_levels_are_within_range_and_multiples_of_round_to(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    by_param = {entry.param: entry for entry in catalogue.rules.entries}
    for i in range(300):
        rules = _draw(config, catalogue, mandate, i)
        for r in rules[1:]:
            entry = by_param[r.param]
            variant = entry.variant_for(mandate.asset_class, mandate.sub_style)
            if variant.level_choices:
                continue
            assert variant.level_min <= r.level <= variant.level_max
            ratio = r.level / variant.round_to
            assert math.isclose(ratio, round(ratio), abs_tol=1e-6)


def test_exclusion_level_is_a_string_from_choices(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    entry = next(e for e in catalogue.rules.entries if e.param == "exclusion")
    variant = entry.variant_for(mandate.asset_class, mandate.sub_style)
    seen = False
    for i in range(300):
        rules = _draw(config, catalogue, mandate, i)
        for r in rules[1:]:
            if r.param == "exclusion":
                seen = True
                assert isinstance(r.level, str)
                assert r.level in variant.level_choices
    assert seen


def test_sub_style_selects_the_matching_stop_variant(catalogue: Catalogue):
    config = Config()
    stop_loss = next(e for e in catalogue.rules.entries if e.param == "stop_loss")
    sub_styles = catalogue.sub_styles[AssetClass.RATES_CREDIT]
    fields_seen = set()
    for sub_style in sub_styles:
        mandate = _mandate(AssetClass.RATES_CREDIT, sub_style.name)
        expected_field = stop_loss.variant_for(AssetClass.RATES_CREDIT, sub_style.name).field
        for i in range(30):
            rules = _draw(config, catalogue, mandate, i)
            stop = next(r for r in rules if r.param == "stop_loss")
            assert stop.field == expected_field
        fields_seen.add(expected_field)
    if len(fields_seen) < 2:
        pytest.skip("catalogue has no two rates_credit sub-styles with different stop_loss fields")


def test_text_has_no_unfilled_template_slots(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    for i in range(100):
        rules = _draw(config, catalogue, mandate, i)
        for r in rules:
            assert "{" not in r.text


def test_trim_at_target_is_included_more_often_than_min_holding_period(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    by_param = {entry.param: entry for entry in catalogue.rules.entries}
    trim, min_hold = by_param["trim_at_target"], by_param["min_holding_period"]
    if trim.share == min_hold.share:
        pytest.skip("trim_at_target and min_holding_period have equal share in this catalogue")
    higher, lower = (trim, min_hold) if trim.share > min_hold.share else (min_hold, trim)
    higher_count = 0
    lower_count = 0
    n = 2000
    for i in range(n):
        rules = _draw(config, catalogue, mandate, i)
        params = {r.param for r in rules[1:]}
        higher_count += higher.param in params
        lower_count += lower.param in params
    assert higher_count > lower_count


def test_only_a_commodities_pm_ever_receives_roll_before_expiry(catalogue: Catalogue):
    config = Config()
    seen_on_commodities = False
    for asset_class in AssetClass:
        mandate = _mandate(asset_class, _first_sub_style(catalogue, asset_class))
        for i in range(300):
            rules = _draw(config, catalogue, mandate, i)
            has_roll = any(r.param == "roll_before_expiry" for r in rules[1:])
            if has_roll:
                assert asset_class == AssetClass.COMMODITIES
                seen_on_commodities = True
    assert seen_on_commodities


def test_roll_before_expiry_level_and_text_are_well_formed(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.COMMODITIES, _first_sub_style(catalogue, AssetClass.COMMODITIES))
    seen = False
    for i in range(300):
        rules = _draw(config, catalogue, mandate, i)
        for r in rules[1:]:
            if r.param == "roll_before_expiry":
                seen = True
                assert 3 <= r.level <= 10
                assert r.level == round(r.level)
                assert "{" not in r.text and "}" not in r.text
    assert seen


def test_non_applicable_entries_consume_no_draws_for_an_equities_pm(catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, _first_sub_style(catalogue, AssetClass.EQUITIES))
    without_roll_entries = tuple(
        entry for entry in catalogue.rules.entries if entry.param != "roll_before_expiry"
    )
    catalogue_without_roll = catalogue.model_copy(
        update={"rules": catalogue.rules.model_copy(update={"entries": without_roll_entries})}
    )
    for i in range(100):
        with_roll = _draw(config, catalogue, mandate, i)
        without_roll = _draw(config, catalogue_without_roll, mandate, i)
        assert with_roll == without_roll
