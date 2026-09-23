import datetime

import pytest
from pydantic import ValidationError

from pm_traitbench.enums import (
    Action,
    AssetClass,
    CommodityGroup,
    DriftEventType,
    EventType,
    ExpiryRule,
    Family,
    InstrumentKind,
    Kind,
    Op,
    Positioning,
    RatingBand,
    Regime,
    RuleScope,
    RuleSource,
    Split,
    StreetView,
    Tenor,
    Typicality,
)
from pm_traitbench.tables.introspect import columns
from pm_traitbench.tables.schema import (
    CalendarEvent,
    ConsensusRow,
    CurvePoint,
    DriftEvent,
    Instrument,
    Mandate,
    Persona,
    Price,
    RegimeSpan,
    Rule,
    StatedProfile,
    Trait,
    multiplier_field,
    to_record,
)

ROW_MODELS = [
    Mandate,
    StatedProfile,
    Persona,
    Trait,
    Rule,
    DriftEvent,
    Instrument,
    Price,
    CurvePoint,
    ConsensusRow,
    CalendarEvent,
    RegimeSpan,
]
MARKET_ROW_MODELS = [Instrument, Price, CurvePoint, ConsensusRow, CalendarEvent, RegimeSpan]


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


# --- market row model fixtures ---


def _equity(**overrides) -> Instrument:
    fields = dict(
        instrument_id="EQ-AAPL",
        family=Family.EQUITIES,
        kind=InstrumentKind.EQUITY,
        name="Apple Inc.",
        currency="USD",
        sector="technology",
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=1.2,
        expiry_rule=None,
    )
    fields.update(overrides)
    return Instrument(**fields)


def _credit_issuer(**overrides) -> Instrument:
    fields = dict(
        instrument_id="CR-XYZ",
        family=Family.CREDIT,
        kind=InstrumentKind.CREDIT_ISSUER,
        name="XYZ Corp",
        currency="USD",
        sector="industrials",
        rating_band=RatingBand.BBB,
        commodity_group=None,
        duration_years=5.0,
        beta=None,
        expiry_rule=None,
    )
    fields.update(overrides)
    return Instrument(**fields)


def _sovereign_curve(**overrides) -> Instrument:
    fields = dict(
        instrument_id="SV-US",
        family=Family.RATES,
        kind=InstrumentKind.SOVEREIGN_CURVE,
        name="US Treasury Curve",
        currency="USD",
        sector=None,
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=None,
        expiry_rule=None,
    )
    fields.update(overrides)
    return Instrument(**fields)


def _commodity(**overrides) -> Instrument:
    fields = dict(
        instrument_id="CM-CL",
        family=Family.COMMODITIES,
        kind=InstrumentKind.COMMODITY,
        name="WTI Crude",
        currency="USD",
        sector=None,
        rating_band=None,
        commodity_group=CommodityGroup.ENERGY,
        duration_years=None,
        beta=None,
        expiry_rule=ExpiryRule.MONTHLY_THIRD_FRIDAY,
    )
    fields.update(overrides)
    return Instrument(**fields)


def _fx_pair(**overrides) -> Instrument:
    fields = dict(
        instrument_id="FX-EURUSD",
        family=Family.FX,
        kind=InstrumentKind.FX_PAIR,
        name="EUR/USD",
        currency="USD",
        sector=None,
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=None,
        expiry_rule=None,
    )
    fields.update(overrides)
    return Instrument(**fields)


_INSTRUMENT_BUILDERS = {
    InstrumentKind.EQUITY: _equity,
    InstrumentKind.CREDIT_ISSUER: _credit_issuer,
    InstrumentKind.SOVEREIGN_CURVE: _sovereign_curve,
    InstrumentKind.COMMODITY: _commodity,
    InstrumentKind.FX_PAIR: _fx_pair,
}


def _price(**overrides) -> Price:
    fields = dict(
        seed="A",
        date=datetime.date(2026, 1, 5),
        instrument_id="EQ-AAPL",
        price=150.0,
        spread_bp=None,
    )
    fields.update(overrides)
    return Price(**fields)


def _curve_point(**overrides) -> CurvePoint:
    fields = dict(
        seed="A", date=datetime.date(2026, 1, 5), curve_id="SV-US", tenor=Tenor.Y10, level=4.25
    )
    fields.update(overrides)
    return CurvePoint(**fields)


def _consensus_row(**overrides) -> ConsensusRow:
    fields = dict(
        seed="A",
        date=datetime.date(2026, 1, 5),
        instrument_id="EQ-AAPL",
        street_score=0.4,
        street_view=StreetView.OVERWEIGHT,
        positioning_pct=62.0,
        positioning=Positioning.CROWDED_LONG,
    )
    fields.update(overrides)
    return ConsensusRow(**fields)


def _calendar_event(**overrides) -> CalendarEvent:
    fields = dict(
        seed="A",
        date=datetime.date(2026, 1, 5),
        instrument_id="EQ-AAPL",
        event=EventType.EARNINGS,
        surprise=0.3,
        affected=Family.EQUITIES.value,
    )
    fields.update(overrides)
    return CalendarEvent(**fields)


def _regime_span(**overrides) -> RegimeSpan:
    fields = dict(
        seed="A",
        regime=Regime.RANGE,
        date_start=datetime.date(2026, 1, 1),
        date_end=datetime.date(2026, 1, 31),
    )
    fields.update(overrides)
    return RegimeSpan(**fields)


# --- Instrument: valid rows and kind invariants ---


@pytest.mark.parametrize("kind", list(InstrumentKind))
def test_instrument_valid_per_kind(kind: InstrumentKind) -> None:
    instrument = _INSTRUMENT_BUILDERS[kind]()
    assert instrument.kind == kind


_INSTRUMENT_EXTRA_FIELD_CASES = [
    (InstrumentKind.EQUITY, "rating_band", RatingBand.BBB),
    (InstrumentKind.CREDIT_ISSUER, "beta", 1.1),
    (InstrumentKind.SOVEREIGN_CURVE, "sector", "government"),
    (InstrumentKind.COMMODITY, "sector", "energy"),
    (InstrumentKind.FX_PAIR, "beta", 1.0),
]


@pytest.mark.parametrize(("kind", "field", "value"), _INSTRUMENT_EXTRA_FIELD_CASES)
def test_instrument_rejects_extra_optional_field(
    kind: InstrumentKind, field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        _INSTRUMENT_BUILDERS[kind](**{field: value})


_INSTRUMENT_MISSING_FIELD_CASES = [
    (InstrumentKind.EQUITY, "beta"),
    (InstrumentKind.CREDIT_ISSUER, "duration_years"),
    (InstrumentKind.COMMODITY, "expiry_rule"),
]


@pytest.mark.parametrize(("kind", "field"), _INSTRUMENT_MISSING_FIELD_CASES)
def test_instrument_rejects_missing_required_field(kind: InstrumentKind, field: str) -> None:
    with pytest.raises(ValidationError):
        _INSTRUMENT_BUILDERS[kind](**{field: None})


def test_instrument_rejects_family_kind_mismatch() -> None:
    with pytest.raises(ValidationError):
        _equity(family=Family.CREDIT)


# --- Price ---


def test_price_valid() -> None:
    assert _price().price == 150.0


def test_price_rejects_non_positive_price() -> None:
    with pytest.raises(ValidationError):
        _price(price=0.0)


def test_price_rejects_spread_on_non_credit_instrument() -> None:
    with pytest.raises(ValidationError):
        _price(spread_bp=120.0)


def test_price_rejects_missing_spread_on_credit_instrument() -> None:
    with pytest.raises(ValidationError):
        _price(instrument_id="CR-XYZ", spread_bp=None)


def test_price_rejects_non_positive_spread_on_credit_instrument() -> None:
    with pytest.raises(ValidationError):
        _price(instrument_id="CR-XYZ", spread_bp=-5.0)


def test_price_credit_instrument_valid() -> None:
    price = _price(instrument_id="CR-XYZ", spread_bp=120.0)
    assert price.spread_bp == 120.0


# --- CurvePoint ---


def test_curve_point_valid() -> None:
    assert _curve_point().tenor == Tenor.Y10


# --- ConsensusRow ---


def test_consensus_row_valid() -> None:
    assert _consensus_row().street_view == StreetView.OVERWEIGHT


def test_consensus_row_rejects_street_score_out_of_range() -> None:
    with pytest.raises(ValidationError):
        _consensus_row(street_score=1.2)


def test_consensus_row_rejects_negative_positioning_pct() -> None:
    with pytest.raises(ValidationError):
        _consensus_row(positioning_pct=-1.0)


# --- CalendarEvent ---


def test_calendar_event_earnings_valid() -> None:
    assert _calendar_event().surprise == 0.3


def test_calendar_event_macro_print_valid() -> None:
    event = _calendar_event(
        instrument_id=None, event=EventType.MACRO_PRINT, surprise=0.2, affected="all"
    )
    assert event.instrument_id is None


def test_calendar_event_contract_expiry_valid() -> None:
    assert _calendar_event(event=EventType.CONTRACT_EXPIRY, surprise=None).surprise is None


def test_calendar_event_rating_downgrade_valid() -> None:
    assert _calendar_event(event=EventType.RATING_DOWNGRADE, surprise=-0.3).surprise == -0.3


def test_calendar_event_rating_upgrade_valid() -> None:
    assert _calendar_event(event=EventType.RATING_UPGRADE, surprise=0.3).surprise == 0.3


def test_calendar_event_rejects_surprise_on_contract_expiry() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(event=EventType.CONTRACT_EXPIRY, surprise=0.1)


def test_calendar_event_rejects_missing_surprise_on_earnings() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(event=EventType.EARNINGS, surprise=None)


def test_calendar_event_rejects_positive_surprise_on_rating_downgrade() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(event=EventType.RATING_DOWNGRADE, surprise=0.2)


def test_calendar_event_rejects_negative_surprise_on_rating_upgrade() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(event=EventType.RATING_UPGRADE, surprise=-0.2)


def test_calendar_event_rejects_instrument_id_on_macro_print() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(
            event=EventType.MACRO_PRINT,
            surprise=0.1,
            affected=Family.EQUITIES.value,
            instrument_id="EQ-AAPL",
        )


def test_calendar_event_rejects_null_instrument_id_on_earnings() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(event=EventType.EARNINGS, instrument_id=None, affected="all")


def test_calendar_event_rejects_affected_all_with_instrument() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(affected="all")


def test_calendar_event_rejects_invalid_affected_value() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(affected="bonds")


def test_calendar_event_rejects_surprise_out_of_range() -> None:
    with pytest.raises(ValidationError):
        _calendar_event(surprise=1.5)


# --- RegimeSpan ---


def test_regime_span_valid() -> None:
    span = _regime_span()
    assert span.date_start <= span.date_end


def test_regime_span_rejects_start_after_end() -> None:
    with pytest.raises(ValidationError):
        _regime_span(date_start=datetime.date(2026, 2, 1), date_end=datetime.date(2026, 1, 1))


# --- Tenor ordering ---


def test_tenor_values_in_order() -> None:
    assert [tenor.value for tenor in Tenor] == [
        "2Y",
        "5Y",
        "10Y",
        "30Y",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "M10",
        "M11",
        "M12",
    ]


# --- round-trip and introspection ---


def test_to_record_calendar_event_null_instrument_id_is_none() -> None:
    event = _calendar_event(
        instrument_id=None, event=EventType.MACRO_PRINT, surprise=0.2, affected="all"
    )
    record = to_record(event)
    assert record["instrument_id"] is None


def test_market_models_survive_columns_introspection() -> None:
    for model in MARKET_ROW_MODELS:
        infos = columns(model)
        assert len(infos) == len(model.model_fields)
