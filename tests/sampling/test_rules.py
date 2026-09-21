import math

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass, RuleScope, RuleSource
from pm_traitbench.rng import stream
from pm_traitbench.sampling.rules import sample_rules
from pm_traitbench.tables.schema import Mandate

_DISCIPLINE_PARAMS = {"no_add_before_trigger", "min_holding_period"}


def _mandate(asset_class: AssetClass, sub_style: str) -> Mandate:
    return Mandate(
        asset_class=asset_class,
        sub_style=sub_style,
        book_size=1e8,
        risk_unit="pct_nav",
        benchmark="cash",
    )


def _draw(config: Config, catalogue: Catalogue, mandate: Mandate, i: int):
    rng = stream(1, "t", i, "rules")
    return sample_rules("pm_001", mandate, config, catalogue, rng)


def test_first_rule_is_the_mandate_cap(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    for i in range(100):
        rules = _draw(config, fixture_catalogue, mandate, i)
        cap = rules[0]
        assert cap.rule_id == "r_01"
        assert cap.source == RuleSource.MANDATE
        assert cap.level in config.rules.max_risk_pct_choices


def test_self_rule_count_is_within_configured_bounds(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    for i in range(300):
        rules = _draw(config, fixture_catalogue, mandate, i)
        self_rules = rules[1:]
        assert config.rules.n_self_rules_min <= len(self_rules) <= config.rules.n_self_rules_max


def test_stop_loss_is_always_present(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    for i in range(300):
        rules = _draw(config, fixture_catalogue, mandate, i)
        assert any(r.param == "stop_loss" for r in rules[1:])


def test_at_least_one_discipline_rule_is_present(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    for i in range(300):
        rules = _draw(config, fixture_catalogue, mandate, i)
        params = {r.param for r in rules[1:]}
        assert params & _DISCIPLINE_PARAMS


def test_rule_ids_are_sequential_with_no_gaps(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    for i in range(100):
        rules = _draw(config, fixture_catalogue, mandate, i)
        assert [r.rule_id for r in rules] == [f"r_{n:02d}" for n in range(1, len(rules) + 1)]


def test_self_rules_are_pm_scoped_with_no_trade_idea(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    for i in range(100):
        rules = _draw(config, fixture_catalogue, mandate, i)
        for r in rules[1:]:
            assert r.scope == RuleScope.PM
            assert r.trade_idea_id is None
            assert r.source == RuleSource.SELF


def test_numeric_levels_are_within_range_and_multiples_of_round_to(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    by_param = {entry.param: entry for entry in fixture_catalogue.rules.entries}
    for i in range(300):
        rules = _draw(config, fixture_catalogue, mandate, i)
        for r in rules[1:]:
            entry = by_param[r.param]
            variant = entry.variant_for(mandate.asset_class, mandate.sub_style)
            if variant.level_choices:
                continue
            assert variant.level_min <= r.level <= variant.level_max
            ratio = r.level / variant.round_to
            assert math.isclose(ratio, round(ratio), abs_tol=1e-6)


def test_exclusion_level_is_a_string_from_choices(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    entry = next(e for e in fixture_catalogue.rules.entries if e.param == "exclusion")
    variant = entry.variant_for(mandate.asset_class, mandate.sub_style)
    seen = False
    for i in range(300):
        rules = _draw(config, fixture_catalogue, mandate, i)
        for r in rules[1:]:
            if r.param == "exclusion":
                seen = True
                assert isinstance(r.level, str)
                assert r.level in variant.level_choices
    assert seen


def test_sub_style_selects_the_matching_stop_variant(fixture_catalogue: Catalogue):
    config = Config()
    sovereign = _mandate(AssetClass.RATES_CREDIT, "sovereign_rates")
    credit = _mandate(AssetClass.RATES_CREDIT, "long_short_credit")
    for i in range(30):
        sovereign_rules = _draw(config, fixture_catalogue, sovereign, i)
        stop = next(r for r in sovereign_rules if r.param == "stop_loss")
        assert stop.field == "adverse_yield_move_bp"

        credit_rules = _draw(config, fixture_catalogue, credit, i)
        stop = next(r for r in credit_rules if r.param == "stop_loss")
        assert stop.field == "adverse_spread_move_bp"


def test_text_has_no_unfilled_template_slots(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    for i in range(100):
        rules = _draw(config, fixture_catalogue, mandate, i)
        for r in rules:
            assert "{" not in r.text


def test_trim_at_target_is_included_more_often_than_min_holding_period(
    fixture_catalogue: Catalogue,
):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    trim_count = 0
    min_hold_count = 0
    n = 2000
    for i in range(n):
        rules = _draw(config, fixture_catalogue, mandate, i)
        params = {r.param for r in rules[1:]}
        trim_count += "trim_at_target" in params
        min_hold_count += "min_holding_period" in params
    assert trim_count > min_hold_count


_SUB_STYLE_BY_ASSET_CLASS = {
    AssetClass.EQUITIES: "equity_long_short",
    AssetClass.RATES_CREDIT: "sovereign_rates",
    AssetClass.COMMODITIES: "commodity_futures_directional",
    AssetClass.MULTI_ASSET: "global_macro",
}


def test_only_a_commodities_pm_ever_receives_roll_before_expiry(fixture_catalogue: Catalogue):
    config = Config()
    seen_on_commodities = False
    for asset_class, sub_style in _SUB_STYLE_BY_ASSET_CLASS.items():
        mandate = _mandate(asset_class, sub_style)
        for i in range(300):
            rules = _draw(config, fixture_catalogue, mandate, i)
            has_roll = any(r.param == "roll_before_expiry" for r in rules[1:])
            if has_roll:
                assert asset_class == AssetClass.COMMODITIES
                seen_on_commodities = True
    assert seen_on_commodities


def test_roll_before_expiry_level_and_text_are_well_formed(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.COMMODITIES, "commodity_futures_directional")
    seen = False
    for i in range(300):
        rules = _draw(config, fixture_catalogue, mandate, i)
        for r in rules[1:]:
            if r.param == "roll_before_expiry":
                seen = True
                assert 3 <= r.level <= 10
                assert r.level == round(r.level)
                assert "{" not in r.text and "}" not in r.text
    assert seen


def test_non_applicable_entries_consume_no_draws_for_an_equities_pm(fixture_catalogue: Catalogue):
    config = Config()
    mandate = _mandate(AssetClass.EQUITIES, "equity_long_short")
    without_roll_entries = tuple(
        entry for entry in fixture_catalogue.rules.entries if entry.param != "roll_before_expiry"
    )
    catalogue_without_roll = fixture_catalogue.model_copy(
        update={
            "rules": fixture_catalogue.rules.model_copy(update={"entries": without_roll_entries})
        }
    )
    for i in range(100):
        with_roll = _draw(config, fixture_catalogue, mandate, i)
        without_roll = _draw(config, catalogue_without_roll, mandate, i)
        assert with_roll == without_roll
