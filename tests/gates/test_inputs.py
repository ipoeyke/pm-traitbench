"""Tests for `build_inputs`: partitioning the engine's tables into one `PmInputs` per PM."""

from datetime import date

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.enums import (
    AssetClass,
    DriftEventType,
    Kind,
    PositionAction,
    RuleResponse,
    Split,
    Typicality,
)
from pm_traitbench.errors import Gate1Error
from pm_traitbench.gates.gate1.inputs import build_inputs
from pm_traitbench.tables.schema import DriftEvent, Mandate, Persona, StatedProfile, Trait
from tests.engine.conftest import fixture_market, fixture_view  # noqa: F401
from tests.gates.conftest import idea_row, ledger_row, position_day, rule_event


def _persona(pm_id: str, asset_class: AssetClass = AssetClass.EQUITIES) -> Persona:
    return Persona(
        pm_id=pm_id,
        market_seed="T",
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
        ideas=[],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped={"pm_010"},
    )
    assert [inputs.pm_id for inputs in result] == ["pm_001"]


def test_build_inputs_partitions_rows_per_pm(fixture_view) -> None:
    idea_1 = idea_row(pm_id="pm_001", trade_idea_id="ti_001")
    idea_2 = idea_row(pm_id="pm_002", trade_idea_id="ti_002", instrument_id="EQ-0002")
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001"), _persona("pm_002")],
        traits=[*_bias_traits("pm_001"), *_bias_traits("pm_002")],
        drift_events=[],
        ideas=[idea_1, idea_2],
        ledger=[],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}, "pm_002": {"c": 0}},
        skipped=set(),
    )
    by_pm = {inputs.pm_id: inputs for inputs in result}
    assert by_pm["pm_001"].ideas == (idea_1,)
    assert by_pm["pm_002"].ideas == (idea_2,)
    assert "ti_001" in by_pm["pm_001"].series
    assert "ti_002" not in by_pm["pm_001"].series


def test_build_inputs_entry_risk_sums_only_entry_date_ledger_rows(fixture_view) -> None:
    idea = idea_row(pm_id="pm_001", trade_idea_id="ti_001", entry_date=date(2026, 1, 5))
    entry_row_a = ledger_row(
        pm_id="pm_001", trade_idea_id="ti_001", date=date(2026, 1, 5), risk_amount=1.0
    )
    entry_row_b = ledger_row(
        pm_id="pm_001", trade_idea_id="ti_001", date=date(2026, 1, 5), risk_amount=2.0
    )
    later_row = ledger_row(
        pm_id="pm_001", trade_idea_id="ti_001", date=date(2026, 1, 6), risk_amount=100.0
    )
    result = build_inputs(
        config=Config(),
        personas=[_persona("pm_001")],
        traits=_bias_traits("pm_001"),
        drift_events=[],
        ideas=[idea],
        ledger=[entry_row_a, entry_row_b, later_row],
        rule_events=[],
        position_days=[],
        views={"T": fixture_view},
        engine_counts={"pm_001": {"c": 0}},
        skipped=set(),
    )
    assert result[0].entry_risk["ti_001"] == 3.0


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
            ideas=[],
            ledger=[],
            rule_events=[],
            position_days=[],
            views={"T": fixture_view},
            engine_counts={},
            skipped=set(),
        )
