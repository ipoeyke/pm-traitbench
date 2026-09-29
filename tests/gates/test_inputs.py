"""Tests for `build_inputs`: partitioning the engine's tables into one `PmInputs` per PM."""

from datetime import date

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.enums import (
    AssetClass,
    DriftEventType,
    Expression,
    Kind,
    PositionAction,
    RuleResponse,
    RuleScope,
    RuleSource,
    Side,
    Split,
    Tenor,
    Typicality,
)
from pm_traitbench.errors import Gate1Error
from pm_traitbench.gates.gate1.inputs import build_inputs
from pm_traitbench.tables.schema import (
    DriftEvent,
    Leg,
    Mandate,
    Persona,
    Rule,
    StatedProfile,
    Trait,
)
from tests.gates.fixtures import idea_row, ledger_row, position_day, rule_event


def _persona(
    pm_id: str, asset_class: AssetClass = AssetClass.EQUITIES, market_seed: str = "T"
) -> Persona:
    return Persona(
        pm_id=pm_id,
        market_seed=market_seed,
        split=Split.PILOT,
        mandate=Mandate(
            asset_class=asset_class,
            sub_style="value",
            book_size=1e8,
            risk_unit="pct_nav",
            benchmark="cash",
        ),
        stated_profile=StatedProfile(self_description="disciplined"),
        typicality=Typicality.TYPICAL,
    )


def _bias_traits(pm_id: str) -> list[Trait]:
    return [
        Trait(
            pm_id=pm_id,
            trait_id=f"t_{i:02d}",
            kind=Kind.BIAS,
            param=param,
            value=0.3,
            active=False,
            mult_range=1.0,
            mult_risk_off=1.0,
            mult_risk_on=1.0,
        )
        for i, param in enumerate(BIAS_PARAMS, start=1)
    ]


def _preference_trait(pm_id: str, trait_id: str = "t_09") -> Trait:
    return Trait(
        pm_id=pm_id,
        trait_id=trait_id,
        kind=Kind.PREFERENCE,
        param="pair_vs_outright",
        value="express the view as a pair trade",
        active=True,
        mult_range=None,
        mult_risk_off=None,
        mult_risk_on=None,
    )


def test_build_inputs_returns_only_direct_asset_pms_sorted(fixture_view) -> None:
    result = build_inputs(
        config=Config(),
        personas=[
            _persona("pm_002"),
            _persona("pm_001"),
            _persona("pm_099", asset_class=AssetClass.MULTI_ASSET),
        ],
        traits=[*_bias_traits("pm_001"), *_bias_traits("pm_002")],
        drift_events=[],
        rules=[],
        ideas=[],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}, "pm_002": {"c": 0}},
        skipped=set(),
    )
    assert [inputs.pm_id for inputs in result] == ["pm_001", "pm_002"]


def test_build_inputs_skips_pm_ids_in_skipped(fixture_view) -> None:
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001"), _persona("pm_010")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        rules=[],
        ideas=[],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped={"pm_010"},
    )
    assert [inputs.pm_id for inputs in result] == ["pm_001"]


def _no_add_rule(pm_id: str, scope: RuleScope) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id="r_04",
        source=RuleSource.SELF,
        scope=scope,
        trade_idea_id=None if scope == RuleScope.PM else "ti_001",
        param="no_add_before_trigger",
        field="triggers_fired",
        op="==",
        level=0.0,
        unit=None,
        window=1,
        action="no_add",
        text="I don't add to the position before a trigger fires",
    )


def test_build_inputs_flags_only_a_pm_scope_no_add_rule(fixture_view) -> None:
    pm_ids = ("pm_001", "pm_002", "pm_003")
    result = build_inputs(
        config=Config(),
        personas=[_persona(pm_id) for pm_id in pm_ids],
        traits=[trait for pm_id in pm_ids for trait in _bias_traits(pm_id)],
        drift_events=[],
        rules=[_no_add_rule("pm_001", RuleScope.PM), _no_add_rule("pm_002", RuleScope.IDEA)],
        ideas=[],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={pm_id: {"c": 0} for pm_id in pm_ids},
        skipped=set(),
    )
    assert [inputs.no_add_before_trigger for inputs in result] == [True, False, False]


def test_build_inputs_partitions_rows_per_pm(fixture_view) -> None:
    param = BIAS_PARAMS[0]
    idea_1 = idea_row(pm_id="pm_001", trade_idea_id="ti_001")
    idea_2 = idea_row(pm_id="pm_002", trade_idea_id="ti_002", instrument_id="EQ-0002")
    ledger_1 = ledger_row(pm_id="pm_001", trade_idea_id="ti_001", risk_amount=1.0)
    ledger_2 = ledger_row(pm_id="pm_002", trade_idea_id="ti_002", risk_amount=2.0)
    position_1 = position_day(
        pm_id="pm_001", trade_idea_id="ti_001", date=date(2026, 1, 6), action=PositionAction.CUT
    )
    position_2 = position_day(
        pm_id="pm_002", trade_idea_id="ti_002", date=date(2026, 1, 7), action=PositionAction.TRIM
    )
    rule_event_1 = rule_event(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        response=RuleResponse.ACTED,
        response_date=date(2026, 1, 8),
    )
    rule_event_2 = rule_event(
        pm_id="pm_002",
        trade_idea_id="ti_002",
        response=RuleResponse.ACKED_NO_ACTION,
        response_date=date(2026, 1, 9),
    )
    drift_1 = DriftEvent(
        pm_id="pm_001",
        date=date(2026, 1, 12),
        event=DriftEventType.UPDATE,
        trait_id="t_01",
        **{"from": 0.3, "to": 0.7},
    )
    drift_2 = DriftEvent(
        pm_id="pm_002",
        date=date(2026, 1, 20),
        event=DriftEventType.UPDATE,
        trait_id="t_01",
        **{"from": 0.3, "to": 0.2},
    )
    traits_1 = _bias_traits("pm_001")
    traits_1[0] = traits_1[0].model_copy(update={"value": 0.7})
    traits_2 = _bias_traits("pm_002")
    traits_2[0] = traits_2[0].model_copy(update={"value": 0.2})

    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001"), _persona("pm_002")],
        traits=[*traits_1, *traits_2],
        drift_events=[drift_1, drift_2],
        rules=[],
        ideas=[idea_1, idea_2],
        ledger=[ledger_1, ledger_2],
        rule_events=[rule_event_1, rule_event_2],
        position_days=[position_1, position_2],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}, "pm_002": {"c": 0}},
        skipped=set(),
    )
    by_pm = {inputs.pm_id: inputs for inputs in result}

    assert by_pm["pm_001"].ideas == (idea_1,)
    assert by_pm["pm_002"].ideas == (idea_2,)
    assert "ti_001" in by_pm["pm_001"].series
    assert "ti_002" not in by_pm["pm_001"].series

    assert by_pm["pm_001"].entry_risk == {"ti_001": 1.0}
    assert by_pm["pm_002"].entry_risk == {"ti_002": 2.0}

    assert by_pm["pm_001"].position_days == (position_1,)
    assert by_pm["pm_002"].position_days == (position_2,)
    assert by_pm["pm_001"].sell_dates == {date(2026, 1, 6)}
    assert by_pm["pm_002"].sell_dates == {date(2026, 1, 7)}

    assert by_pm["pm_001"].rule_events == (rule_event_1,)
    assert by_pm["pm_002"].rule_events == (rule_event_2,)
    assert by_pm["pm_001"].acted == {("ti_001", date(2026, 1, 8))}
    assert by_pm["pm_002"].acted == frozenset()

    assert by_pm["pm_001"].drift_dates[param] == (date(2026, 1, 12),)
    assert by_pm["pm_002"].drift_dates[param] == (date(2026, 1, 20),)

    assert by_pm["pm_001"].traits[param].value == 0.7
    assert by_pm["pm_002"].traits[param].value == 0.2


def test_build_inputs_entry_risk_reads_lead_leg_row_not_later_dates(fixture_view) -> None:
    idea = idea_row(pm_id="pm_001", trade_idea_id="ti_001", entry_date=date(2026, 1, 5))
    entry_row = ledger_row(
        pm_id="pm_001", trade_idea_id="ti_001", date=date(2026, 1, 5), risk_amount=1.0
    )
    later_row = ledger_row(
        pm_id="pm_001", trade_idea_id="ti_001", date=date(2026, 1, 6), risk_amount=100.0
    )
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        rules=[],
        ideas=[idea],
        ledger=[entry_row, later_row],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    assert result[0].entry_risk["ti_001"] == 1.0


def test_build_inputs_entry_risk_pair_reads_lead_leg_not_the_sum(fixture_view) -> None:
    idea = idea_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        entry_date=date(2026, 1, 5),
        expression=Expression.PAIR,
        legs=(
            Leg(instrument_id="EQ-0001", tenor=None, side=Side.BUY, weight=1.0),
            Leg(instrument_id="EQ-0002", tenor=None, side=Side.SELL, weight=1.0),
        ),
    )
    lead_leg_row = ledger_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        date=date(2026, 1, 5),
        instrument_id="EQ-0001",
        risk_amount=1.0,
    )
    other_leg_row = ledger_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        date=date(2026, 1, 5),
        instrument_id="EQ-0002",
        risk_amount=2.0,
    )
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        rules=[],
        ideas=[idea],
        ledger=[other_leg_row, lead_leg_row],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    assert result[0].entry_risk["ti_001"] == 1.0


def test_build_inputs_entry_risk_curve_reads_lead_leg_tenor(fixture_view) -> None:
    idea = idea_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        instrument_id="RT-USD",
        entry_date=date(2026, 1, 5),
        expression=Expression.CURVE,
        legs=(
            Leg(instrument_id="RT-USD", tenor=Tenor.Y2, side=Side.BUY, weight=1.0),
            Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.SELL, weight=1.0),
        ),
    )
    long_tenor_row = ledger_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        date=date(2026, 1, 5),
        instrument_id="RT-USD",
        tenor=Tenor.Y10,
        risk_amount=2.0,
    )
    lead_tenor_row = ledger_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        date=date(2026, 1, 5),
        instrument_id="RT-USD",
        tenor=Tenor.Y2,
        risk_amount=1.0,
    )
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        rules=[],
        ideas=[idea],
        ledger=[long_tenor_row, lead_tenor_row],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    assert result[0].entry_risk["ti_001"] == 1.0


def test_build_inputs_entry_risk_falls_back_to_first_row_when_no_leg_matches(
    fixture_view,
) -> None:
    idea = idea_row(pm_id="pm_001", trade_idea_id="ti_001", entry_date=date(2026, 1, 5))
    first_row = ledger_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        date=date(2026, 1, 5),
        instrument_id="EQ-9999",
        risk_amount=1.0,
    )
    second_row = ledger_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        date=date(2026, 1, 5),
        instrument_id="EQ-8888",
        risk_amount=2.0,
    )
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        rules=[],
        ideas=[idea],
        ledger=[first_row, second_row],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    assert result[0].entry_risk["ti_001"] == 1.0


def test_build_inputs_entry_conviction_reads_first_entry_date_ledger_row(fixture_view) -> None:
    idea = idea_row(pm_id="pm_001", trade_idea_id="ti_001", entry_date=date(2026, 1, 5))
    first_row = ledger_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        date=date(2026, 1, 5),
        stated_conviction=4,
    )
    second_row = ledger_row(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        date=date(2026, 1, 5),
        stated_conviction=2,
    )
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        rules=[],
        ideas=[idea],
        ledger=[first_row, second_row],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    assert result[0].entry_conviction["ti_001"] == 4


def test_build_inputs_sell_dates_and_acted(fixture_view) -> None:
    hold = position_day(date=date(2026, 1, 5), action=PositionAction.HOLD)
    cut = position_day(date=date(2026, 1, 6), action=PositionAction.CUT)
    trim = position_day(date=date(2026, 1, 7), action=PositionAction.TRIM)
    exit_ = position_day(date=date(2026, 1, 8), action=PositionAction.EXIT)
    acted_event = rule_event(
        trade_idea_id="ti_001",
        response=RuleResponse.ACTED,
        response_date=date(2026, 1, 9),
    )
    acked_event = rule_event(
        trade_idea_id="ti_001",
        response=RuleResponse.ACKED_NO_ACTION,
        response_date=date(2026, 1, 10),
    )
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        rules=[],
        ideas=[],
        ledger=[],
        rule_events=[acted_event, acked_event],
        position_days=[hold, cut, trim, exit_],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    inputs = result[0]
    assert inputs.sell_dates == {date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)}
    assert inputs.acted == {("ti_001", date(2026, 1, 9))}


def test_build_inputs_drift_dates_maps_trait_ids_and_ignores_preferences(fixture_view) -> None:
    param = BIAS_PARAMS[0]
    bias_event = DriftEvent(
        pm_id="pm_001",
        date=date(2026, 1, 15),
        event=DriftEventType.UPDATE,
        trait_id="t_01",
        **{"from": 0.3, "to": 0.5},
    )
    preference_event = DriftEvent(
        pm_id="pm_001",
        date=date(2026, 1, 20),
        event=DriftEventType.UPDATE,
        trait_id="t_09",
        **{
            "from": "express the view as a pair trade",
            "to": "express the view as an outright position",
        },
    )
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=[*_bias_traits("pm_001"), _preference_trait("pm_001")],
        drift_events=[bias_event, preference_event],
        rules=[],
        ideas=[],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    inputs = result[0]
    assert inputs.drift_dates[param] == (date(2026, 1, 15),)
    for other_param in BIAS_PARAMS[1:]:
        assert inputs.drift_dates[other_param] == ()
    assert all(date(2026, 1, 20) not in dates for dates in inputs.drift_dates.values())


def test_build_inputs_missing_bias_trait_raises(fixture_view) -> None:
    incomplete_traits = _bias_traits("pm_001")[:-1]
    with pytest.raises(Gate1Error, match="pm_001"):
        build_inputs(
            config=Config(),
            personas=[_persona("pm_001")],
            traits=incomplete_traits,
            drift_events=[],
            rules=[],
            ideas=[],
            ledger=[],
            rule_events=[],
            position_days=[],
            views={"T": fixture_view},
            engine_counts={"pm_001": {"c": 0}},
            skipped=set(),
        )


def test_build_inputs_pm_missing_engine_counts_raises(fixture_view) -> None:
    with pytest.raises(Gate1Error, match="pm_001"):
        build_inputs(
            config=Config(),
            personas=[_persona("pm_001")],
            traits=_bias_traits("pm_001"),
            drift_events=[],
            rules=[],
            ideas=[],
            ledger=[],
            rule_events=[],
            position_days=[],
            views={"T": fixture_view},
            engine_counts={},
            skipped=set(),
        )


def test_build_inputs_is_real_seed_reflects_config(fixture_view) -> None:
    config = Config()
    real_seed = next(iter(config.market.real.seeds))
    real_view = MarketView.build(
        seed=real_seed,
        dates=fixture_view.dates,
        instruments=[],
        prices=[],
        curves=[],
        consensus=[],
        calendar=[],
        regimes=[],
    )
    result = build_inputs(
        config=config,
        personas=[
            _persona("pm_001", market_seed="T"),
            _persona("pm_002", market_seed=real_seed),
        ],
        traits=[*_bias_traits("pm_001"), *_bias_traits("pm_002")],
        drift_events=[],
        rules=[],
        ideas=[],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view, real_seed: real_view},
        engine_counts={"pm_001": {"c": 0}, "pm_002": {"c": 0}},
        skipped=set(),
    )
    by_pm = {inputs.pm_id: inputs for inputs in result}
    assert by_pm["pm_001"].is_real_seed is False
    assert by_pm["pm_002"].is_real_seed is True


def test_build_inputs_engine_counts_carried_per_pm(fixture_view) -> None:
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001"), _persona("pm_002")],
        traits=[*_bias_traits("pm_001"), *_bias_traits("pm_002")],
        drift_events=[],
        rules=[],
        ideas=[],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"ideas": 3}, "pm_002": {"ideas": 7}},
        skipped=set(),
    )
    by_pm = {inputs.pm_id: inputs for inputs in result}
    assert by_pm["pm_001"].engine_counts == {"ideas": 3}
    assert by_pm["pm_002"].engine_counts == {"ideas": 7}


def test_build_inputs_idea_with_no_entry_date_ledger_row_has_no_conviction(fixture_view) -> None:
    idea = idea_row(pm_id="pm_001", trade_idea_id="ti_001", entry_date=date(2026, 1, 5))
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        rules=[],
        ideas=[idea],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    inputs = result[0]
    assert inputs.entry_risk["ti_001"] == 0.0
    assert "ti_001" not in inputs.entry_conviction


def test_build_inputs_missing_view_for_seed_raises(fixture_view) -> None:
    with pytest.raises(Gate1Error) as excinfo:
        build_inputs(
            config=Config(),
            personas=[_persona("pm_001", market_seed="ZZZ")],
            traits=_bias_traits("pm_001"),
            drift_events=[],
            rules=[],
            ideas=[],
            ledger=[],
            rule_events=[],
            position_days=[],
            views={"T": fixture_view},
            engine_counts={"pm_001": {"c": 0}},
            skipped=set(),
        )
    assert "pm_001" in str(excinfo.value)
    assert "ZZZ" in str(excinfo.value)
