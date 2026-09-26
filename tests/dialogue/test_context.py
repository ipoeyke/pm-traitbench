"""Tests for session context assembly: PM filtering and per-skeleton context building."""

from datetime import date, timedelta

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import Voice
from pm_traitbench.config import Config, PmFilter
from pm_traitbench.dialogue.context import PmTables, build_contexts, select_pms
from pm_traitbench.enums import (
    Action,
    DriftEventType,
    DriftStatus,
    InstrumentKind,
    Op,
    RuleScope,
    RuleSource,
    SessionKind,
    Split,
    Typicality,
)
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import Rule, Skeleton
from tests.gates.conftest import DEFAULT_DATE, PM_ID, idea_row, ledger_row, position_day
from tests.signals.conftest import bias_trait, drift_event, persona

_VOICE = Voice(voice_id="v_01", line="terse trader shorthand, drops articles")


@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue()


def _persona(pm_id=PM_ID, split=Split.PILOT, typicality=Typicality.TYPICAL):
    return persona().model_copy(update={"pm_id": pm_id, "split": split, "typicality": typicality})


def _pm_tables(
    *,
    pm_id: str = PM_ID,
    split: Split = Split.PILOT,
    typicality: Typicality = Typicality.TYPICAL,
    drift_events: tuple = (),
    rules: tuple = (),
    ideas: dict | None = None,
    ledger: tuple = (),
    position_days: tuple = (),
    skeletons: tuple = (),
) -> PmTables:
    return PmTables(
        persona=_persona(pm_id=pm_id, split=split, typicality=typicality),
        traits=(),
        drift_events=drift_events,
        rules=rules,
        ideas=ideas or {},
        ledger=ledger,
        position_days=position_days,
        skeletons=skeletons,
    )


def _rule(rule_id: str, text: str, *, scope=RuleScope.PM, trade_idea_id=None, pm_id=PM_ID) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id=rule_id,
        source=RuleSource.SELF,
        scope=scope,
        trade_idea_id=trade_idea_id,
        param="stop_loss",
        field="pnl_pct",
        op=Op.LE,
        level=-5.0,
        unit="pct",
        window=1,
        action=Action.EXIT,
        text=text,
    )


def _skeleton(
    *,
    session_id: str,
    pm_id: str = PM_ID,
    session_date: date = DEFAULT_DATE,
    kind: SessionKind = SessionKind.DECISION,
    trade_idea_ids: tuple = (),
    stances: tuple = (),
    advisor_violation: str | None = None,
    forbidden_trait_ids: tuple = (),
    forbidden_pref_params: tuple = (),
) -> Skeleton:
    return Skeleton(
        session_id=session_id,
        pm_id=pm_id,
        date=session_date,
        kind=kind,
        trade_idea_ids=trade_idea_ids,
        stances=stances,
        advisor_violation=advisor_violation,
        forbidden_trait_ids=forbidden_trait_ids,
        forbidden_pref_params=forbidden_pref_params,
    )


def test_select_pms_applies_each_filter():
    pilot_typical_static = _pm_tables(
        pm_id="pm_001", split=Split.PILOT, typicality=Typicality.TYPICAL
    )
    pilot_anti_static = _pm_tables(
        pm_id="pm_002", split=Split.PILOT, typicality=Typicality.ANTI_TYPICAL
    )
    full_typical_drift = _pm_tables(
        pm_id="pm_003",
        split=Split.FULL,
        typicality=Typicality.TYPICAL,
        drift_events=(
            drift_event(
                "t_01", date(2026, 5, 1), DriftEventType.UPDATE, from_value=0.3, to_value=0.5
            ),
        ),
    )
    all_pms = [full_typical_drift, pilot_anti_static, pilot_typical_static]

    def ids(pm_filter: PmFilter) -> list[str]:
        return [pm.persona.pm_id for pm in select_pms(all_pms, pm_filter)]

    assert ids(PmFilter()) == ["pm_001", "pm_002", "pm_003"]  # unfiltered, pm_id order
    assert ids(PmFilter(split=Split.PILOT)) == ["pm_001", "pm_002"]
    assert ids(PmFilter(typicality=Typicality.ANTI_TYPICAL)) == ["pm_002"]
    assert ids(PmFilter(drift=DriftStatus.DRIFT)) == ["pm_003"]
    assert ids(PmFilter(drift=DriftStatus.STATIC)) == ["pm_001", "pm_002"]
    assert ids(PmFilter(pm_ids=("pm_002",))) == ["pm_002"]
    assert ids(PmFilter(split=Split.PILOT, typicality=Typicality.TYPICAL)) == ["pm_001"]


def test_contexts_follow_skeleton_order(market_lookup, catalogue):
    skeleton_b = _skeleton(
        session_id="s_pm001_2026-01-05_b", kind=SessionKind.SILENCE, session_date=DEFAULT_DATE
    )
    skeleton_a = _skeleton(
        session_id="s_pm001_2026-01-05_a", kind=SessionKind.SILENCE, session_date=DEFAULT_DATE
    )
    pm = _pm_tables(skeletons=(skeleton_b, skeleton_a))

    contexts = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())

    assert [ctx.skeleton.session_id for ctx in contexts] == [
        "s_pm001_2026-01-05_a",
        "s_pm001_2026-01-05_b",
    ]


def test_day_trades_are_the_session_date_rows_for_session_ideas_only(market_lookup, catalogue):
    idea_1 = idea_row(trade_idea_id="ti_001")
    idea_2 = idea_row(trade_idea_id="ti_002", instrument_id="EQ-0002")
    in_session_row = ledger_row(trade_idea_id="ti_001", date=DEFAULT_DATE)
    other_idea_row = ledger_row(trade_idea_id="ti_002", date=DEFAULT_DATE, instrument_id="EQ-0002")
    other_date_row = ledger_row(trade_idea_id="ti_001", date=DEFAULT_DATE + timedelta(days=1))
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a", trade_idea_ids=("ti_001",), session_date=DEFAULT_DATE
    )
    pm = _pm_tables(
        ideas={"ti_001": idea_1, "ti_002": idea_2},
        ledger=(in_session_row, other_idea_row, other_date_row),
        skeletons=(skeleton,),
    )

    contexts = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())

    assert contexts[0].day_trades == (in_session_row,)


def test_open_positions_come_from_position_days_on_the_date(market_lookup, catalogue):
    idea_1 = idea_row(trade_idea_id="ti_001")
    idea_2 = idea_row(trade_idea_id="ti_002", instrument_id="EQ-0002")
    on_date = position_day(trade_idea_id="ti_001", date=DEFAULT_DATE)
    off_date = position_day(trade_idea_id="ti_002", date=DEFAULT_DATE + timedelta(days=1))
    skeleton = _skeleton(session_id="s_pm001_2026-01-05_a", session_date=DEFAULT_DATE)
    pm = _pm_tables(
        ideas={"ti_001": idea_1, "ti_002": idea_2},
        position_days=(on_date, off_date),
        skeletons=(skeleton,),
    )

    contexts = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())

    assert contexts[0].open_positions == (idea_1,)


def test_silence_session_draws_a_question_instrument_of_the_pm_asset_class(
    market_lookup, catalogue
):
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a", kind=SessionKind.SILENCE, session_date=DEFAULT_DATE
    )
    pm = _pm_tables(skeletons=(skeleton,))

    contexts = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())

    instrument = contexts[0].question_instrument
    assert instrument is not None
    assert instrument.kind == InstrumentKind.EQUITY
    assert instrument.instrument_id in contexts[0].instrument_names

    # Same inputs, same draw.
    again = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())
    assert again[0].question_instrument == instrument


def test_avoid_lines_render_forbidden_traits_and_preference_params(market_lookup, catalogue):
    idea = idea_row(trade_idea_id="ti_001")
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a",
        trade_idea_ids=("ti_001",),
        session_date=DEFAULT_DATE,
        forbidden_trait_ids=("t_01",),
        forbidden_pref_params=("register",),
    )
    pm = PmTables(
        persona=_persona(),
        traits=(bias_trait("loss_aversion_lambda"),),
        drift_events=(),
        rules=(),
        ideas={"ti_001": idea},
        ledger=(),
        position_days=(),
        skeletons=(skeleton,),
    )

    contexts = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())

    assert contexts[0].avoid_lines == tuple(
        sorted(
            (
                catalogue.avoid.biases["loss_aversion_lambda"],
                catalogue.avoid.preferences["register"],
            )
        )
    )


def test_unknown_trade_idea_raises(market_lookup, catalogue):
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a", trade_idea_ids=("ti_999",), session_date=DEFAULT_DATE
    )
    pm = _pm_tables(skeletons=(skeleton,))

    with pytest.raises(DialogueError):
        build_contexts(pm, _VOICE, market_lookup, catalogue, Config())
