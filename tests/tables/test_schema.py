import datetime

import pytest
from pydantic import ValidationError

from pm_traitbench.enums import (
    Action,
    AssetClass,
    DriftEventType,
    Kind,
    Op,
    Regime,
    RuleScope,
    RuleSource,
    Split,
    Typicality,
)
from pm_traitbench.tables.schema import (
    DriftEvent,
    Mandate,
    Persona,
    Rule,
    StatedProfile,
    Trait,
    multiplier_field,
    to_record,
)

ROW_MODELS = [Mandate, StatedProfile, Persona, Trait, Rule, DriftEvent]


def _mandate(**overrides) -> Mandate:
    fields = dict(
        asset_class=AssetClass.EQUITIES,
        sub_style="value",
        book_size=250_000_000.0,
        risk_unit="pct_nav",
        benchmark="MSCI World",
    )
    fields.update(overrides)
    return Mandate(**fields)


def _stated_profile(**overrides) -> StatedProfile:
    fields = dict(self_description="Disciplined value investor focused on downside protection.")
    fields.update(overrides)
    return StatedProfile(**fields)


def _persona(**overrides) -> Persona:
    fields = dict(
        pm_id="pm_001",
        market_seed="A",
        split=Split.PILOT,
        mandate=_mandate(),
        stated_profile=_stated_profile(),
        typicality=Typicality.TYPICAL,
    )
    fields.update(overrides)
    return Persona(**fields)


def _bias_trait(**overrides) -> Trait:
    fields = dict(
        pm_id="pm_001",
        trait_id="t_01",
        kind=Kind.BIAS,
        param="loss_aversion_lambda",
        value=2.6,
        active=True,
        mult_range=1.1,
        mult_risk_off=1.3,
        mult_risk_on=0.9,
    )
    fields.update(overrides)
    return Trait(**fields)


def _preference_trait(**overrides) -> Trait:
    fields = dict(
        pm_id="pm_001",
        trait_id="t_02",
        kind=Kind.PREFERENCE,
        param="sector_focus",
        value="technology",
        active=True,
        mult_range=None,
        mult_risk_off=None,
        mult_risk_on=None,
    )
    fields.update(overrides)
    return Trait(**fields)


def _pm_rule(**overrides) -> Rule:
    fields = dict(
        pm_id="pm_001",
        rule_id="r_01",
        source=RuleSource.MANDATE,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="max_risk_pct",
        field="risk_pct",
        op=Op.LE,
        level=10.0,
        unit="pct_nav",
        window=1,
        action=Action.CAP,
        text="Cap risk at 10% of NAV.",
    )
    fields.update(overrides)
    return Rule(**fields)


def _idea_rule(**overrides) -> Rule:
    fields = dict(
        pm_id="pm_001",
        rule_id="r_02",
        source=RuleSource.SELF,
        scope=RuleScope.IDEA,
        trade_idea_id="idea_001",
        param="stop_loss",
        field="price",
        op=Op.LT,
        level=95.0,
        unit="usd",
        window=5,
        action=Action.EXIT,
        text="Exit if price falls below 95.",
    )
    fields.update(overrides)
    return Rule(**fields)


def _update_event(**overrides) -> DriftEvent:
    fields = dict(
        pm_id="pm_001",
        date=datetime.date(2026, 3, 2),
        event=DriftEventType.UPDATE,
        trait_id="t_01",
        from_value=1.1,
        to_value=1.4,
    )
    fields.update(overrides)
    return DriftEvent(**fields)


def _dormant_event(**overrides) -> DriftEvent:
    fields = dict(
        pm_id="pm_001",
        date=datetime.date(2026, 3, 2),
        event=DriftEventType.DORMANT,
        trait_id="t_01",
        from_value=None,
        to_value=None,
    )
    fields.update(overrides)
    return DriftEvent(**fields)


# --- valid construction, one per model ---


def test_mandate_valid() -> None:
    mandate = _mandate()
    assert mandate.asset_class == AssetClass.EQUITIES
    assert mandate.book_size == 250_000_000.0


def test_stated_profile_valid() -> None:
    assert _stated_profile().self_description.startswith("Disciplined")


def test_persona_valid() -> None:
    persona = _persona()
    assert persona.pm_id == "pm_001"
    assert isinstance(persona.mandate, Mandate)


def test_trait_bias_valid() -> None:
    trait = _bias_trait()
    assert trait.value == 2.6
    assert trait.mult_range == 1.1


def test_trait_preference_valid() -> None:
    trait = _preference_trait()
    assert trait.value == "technology"
    assert trait.mult_range is None


def test_rule_pm_scope_valid() -> None:
    rule = _pm_rule()
    assert rule.trade_idea_id is None


def test_rule_idea_scope_valid() -> None:
    rule = _idea_rule()
    assert rule.trade_idea_id == "idea_001"


def test_drift_event_update_valid() -> None:
    event = _update_event()
    assert event.from_value == 1.1
    assert event.to_value == 1.4


def test_drift_event_dormant_valid() -> None:
    event = _dormant_event()
    assert event.from_value is None
    assert event.to_value is None


# --- Trait bias/preference invariants ---


@pytest.mark.parametrize("text", ["2.6", "not-a-number"])
def test_trait_bias_rejects_a_text_value(text: str) -> None:
    with pytest.raises(ValidationError, match="float value"):
        _bias_trait(value=text)


def test_trait_bias_requires_positive_multipliers() -> None:
    with pytest.raises(ValidationError):
        _bias_trait(mult_risk_on=0.0)


def test_trait_bias_requires_non_null_multipliers() -> None:
    with pytest.raises(ValidationError):
        _bias_trait(mult_range=None)


def test_trait_preference_requires_str_value() -> None:
    with pytest.raises(ValidationError):
        _preference_trait(value=1.0)


def test_trait_preference_requires_active_true() -> None:
    with pytest.raises(ValidationError):
        _preference_trait(active=False)


def test_trait_preference_requires_null_multipliers() -> None:
    with pytest.raises(ValidationError):
        _preference_trait(mult_range=1.0)


def test_trait_preference_value_stays_str_when_numeric_looking() -> None:
    trait = _preference_trait(value="123")
    assert trait.value == "123"
    assert isinstance(trait.value, str)


# --- Rule.level typing and scope invariants ---


def test_rule_level_number_stays_float() -> None:
    rule = _pm_rule(level=-15.0)
    assert rule.level == -15.0
    assert isinstance(rule.level, float)


@pytest.mark.parametrize("text", ["energy", "-15", "1_0", "nan", " 5 "])
def test_rule_level_text_is_never_read_as_a_number(text: str) -> None:
    rule = _pm_rule(level=text)
    assert rule.level == text
    assert isinstance(rule.level, str)


def test_rule_pm_scope_rejects_trade_idea_id() -> None:
    with pytest.raises(ValidationError):
        _pm_rule(trade_idea_id="idea_001")


def test_rule_idea_scope_requires_trade_idea_id() -> None:
    with pytest.raises(ValidationError):
        _idea_rule(trade_idea_id=None)


# --- DriftEvent invariants and alias behaviour ---


def test_drift_event_update_requires_both_values() -> None:
    with pytest.raises(ValidationError):
        _update_event(to_value=None)


def test_drift_event_dormant_requires_both_null() -> None:
    with pytest.raises(ValidationError):
        _dormant_event(from_value=1.0)


def test_drift_event_revive_requires_both_null() -> None:
    with pytest.raises(ValidationError):
        _dormant_event(event=DriftEventType.REVIVE, to_value=2.0)


def test_drift_event_accepts_alias_input() -> None:
    by_alias = DriftEvent.model_validate(
        {
            "pm_id": "pm_001",
            "date": "2026-03-02",
            "event": "update",
            "trait_id": "t_01",
            "from": 1.0,
            "to": 2.0,
        }
    )
    by_name = DriftEvent(
        pm_id="pm_001",
        date=datetime.date(2026, 3, 2),
        event=DriftEventType.UPDATE,
        trait_id="t_01",
        from_value=1.0,
        to_value=2.0,
    )
    assert by_alias == by_name


def test_drift_event_endpoints_keep_the_type_they_were_given() -> None:
    event = _update_event(from_value=-3.0, to_value="-3")
    assert isinstance(event.from_value, float)
    assert event.to_value == "-3"


# --- id patterns ---


@pytest.mark.parametrize("pm_id", ["pm_001", "pm_1234"])
def test_pm_id_pattern_accepts(pm_id: str) -> None:
    _persona(pm_id=pm_id)


@pytest.mark.parametrize("pm_id", ["pm_1", "pm_12", "pm001", "PM_001"])
def test_pm_id_pattern_rejects(pm_id: str) -> None:
    with pytest.raises(ValidationError):
        _persona(pm_id=pm_id)


@pytest.mark.parametrize("trait_id", ["t_01", "t_123"])
def test_trait_id_pattern_accepts(trait_id: str) -> None:
    _bias_trait(trait_id=trait_id)


@pytest.mark.parametrize("trait_id", ["t_1", "t01", "T_01"])
def test_trait_id_pattern_rejects(trait_id: str) -> None:
    with pytest.raises(ValidationError):
        _bias_trait(trait_id=trait_id)


@pytest.mark.parametrize("rule_id", ["r_01", "r_123"])
def test_rule_id_pattern_accepts(rule_id: str) -> None:
    _pm_rule(rule_id=rule_id)


@pytest.mark.parametrize("rule_id", ["r_1", "r01", "R_01"])
def test_rule_id_pattern_rejects(rule_id: str) -> None:
    with pytest.raises(ValidationError):
        _pm_rule(rule_id=rule_id)


# --- to_record ---


def test_to_record_drift_event_uses_aliases_and_iso_date() -> None:
    record = to_record(_update_event())
    assert record["from"] == 1.1
    assert record["to"] == 1.4
    assert record["date"] == "2026-03-02"
    assert "from_value" not in record
    assert "to_value" not in record


def test_to_record_persona_nests_mandate_as_dict() -> None:
    record = to_record(_persona())
    assert isinstance(record["mandate"], dict)
    assert record["mandate"]["asset_class"] == "equities"


# --- frozen and extra="forbid" ---


def test_models_are_frozen() -> None:
    mandate = _mandate()
    with pytest.raises(ValidationError):
        mandate.sub_style = "growth"


def test_unknown_field_raises() -> None:
    with pytest.raises(ValidationError):
        _mandate(unknown_field="nope")


# --- descriptions ---


def test_every_field_has_description() -> None:
    for model in ROW_MODELS:
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name} is missing a description"


# --- helper functions ---


def test_multiplier_field() -> None:
    assert multiplier_field(Regime.RANGE) == "mult_range"
    assert multiplier_field(Regime.RISK_OFF) == "mult_risk_off"
    assert multiplier_field(Regime.RISK_ON) == "mult_risk_on"
