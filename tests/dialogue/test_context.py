"""Tests for session context assembly: PM filtering and per-skeleton context building."""

from datetime import date, timedelta

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import Voice
from pm_traitbench.config import Config, PmFilter
from pm_traitbench.dialogue.context import (
    PmTables,
    TradeNote,
    annotate_day_trades,
    build_contexts,
    select_pms,
    trade_key,
)
from pm_traitbench.dialogue.prompts import narrator_system
from pm_traitbench.dialogue.tools import MarketLookup
from pm_traitbench.enums import (
    AssetClass,
    DriftEventType,
    DriftStatus,
    Expression,
    InstrumentKind,
    PositionAction,
    RuleScope,
    SessionKind,
    Side,
    Split,
    Tenor,
    Typicality,
)
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import Leg, Skeleton
from tests.dialogue.fixtures import rule
from tests.gates.fixtures import DEFAULT_DATE, PM_ID, idea_row, ledger_row, position_day
from tests.signals.fixtures import bias_trait, drift_event, persona

_VOICE = Voice(voice_id="v_01", line="terse trader shorthand, drops articles")


@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue()


def _persona(
    pm_id=PM_ID,
    split=Split.PILOT,
    typicality=Typicality.TYPICAL,
    asset_class=AssetClass.EQUITIES,
):
    return persona(asset_class=asset_class).model_copy(
        update={"pm_id": pm_id, "split": split, "typicality": typicality}
    )


def _pm_tables(
    *,
    pm_id: str = PM_ID,
    split: Split = Split.PILOT,
    typicality: Typicality = Typicality.TYPICAL,
    asset_class: AssetClass = AssetClass.EQUITIES,
    traits: tuple = (),
    drift_events: tuple = (),
    rules: tuple = (),
    ideas: dict | None = None,
    ledger: tuple = (),
    position_days: tuple = (),
    skeletons: tuple = (),
) -> PmTables:
    return PmTables(
        persona=_persona(pm_id=pm_id, split=split, typicality=typicality, asset_class=asset_class),
        traits=traits,
        drift_events=drift_events,
        rules=rules,
        ideas=ideas or {},
        ledger=ledger,
        position_days=position_days,
        skeletons=skeletons,
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


def test_day_trades_use_the_full_ledger_key_with_a_two_leg_idea_in_reverse_order(
    market_lookup, catalogue
):
    idea = idea_row(
        trade_idea_id="ti_001",
        instrument_id="RT-USD",
        expression=Expression.CURVE,
        legs=(
            Leg(instrument_id="RT-USD", tenor=Tenor.Y10, side=Side.BUY, weight=1.0),
            Leg(instrument_id="RT-USD", tenor=Tenor.Y30, side=Side.SELL, weight=1.0),
        ),
    )
    leg_30y = ledger_row(
        trade_idea_id="ti_001",
        date=DEFAULT_DATE,
        instrument_id="RT-USD",
        tenor=Tenor.Y30,
        instrument_type=InstrumentKind.SOVEREIGN_CURVE,
        side=Side.SELL,
    )
    leg_10y = ledger_row(
        trade_idea_id="ti_001",
        date=DEFAULT_DATE,
        instrument_id="RT-USD",
        tenor=Tenor.Y10,
        instrument_type=InstrumentKind.SOVEREIGN_CURVE,
        side=Side.BUY,
    )
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a", trade_idea_ids=("ti_001",), session_date=DEFAULT_DATE
    )
    # Rows given in reverse order relative to the ledger table's own key
    # (trade_idea_id, instrument_id, tenor, side): "10Y" sorts before "30Y".
    pm = _pm_tables(
        asset_class=AssetClass.RATES_CREDIT,
        ideas={"ti_001": idea},
        ledger=(leg_30y, leg_10y),
        skeletons=(skeleton,),
    )

    contexts = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())

    assert contexts[0].day_trades == (leg_10y, leg_30y)


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


def test_session_with_no_instruments_of_its_own_names_the_whole_asset_class(
    market_lookup, catalogue
):
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a", kind=SessionKind.CHECK_IN, session_date=DEFAULT_DATE
    )
    pm = _pm_tables(skeletons=(skeleton,))

    contexts = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())

    equities = {
        i.instrument_id: i.name
        for i in market_lookup.instruments.values()
        if i.kind == InstrumentKind.EQUITY
    }
    assert equities
    assert contexts[0].instrument_names == equities


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


def test_silence_session_question_instrument_covers_both_rates_credit_kinds(
    fixture_market, catalogue
):
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a", kind=SessionKind.SILENCE, session_date=DEFAULT_DATE
    )
    pm = _pm_tables(asset_class=AssetClass.RATES_CREDIT, skeletons=(skeleton,))

    curve_only_lookup = MarketLookup.build(
        seed="T",
        instruments=fixture_market["instruments"],
        prices=[],
        curves=[c for c in fixture_market["curves"] if c.curve_id == "RT-USD"],
        consensus=fixture_market["consensus"],
        calendar=fixture_market["calendar"],
    )
    credit_only_lookup = MarketLookup.build(
        seed="T",
        instruments=fixture_market["instruments"],
        prices=[
            p for p in fixture_market["prices"] if p.instrument_id in {"CR-IG-001", "CR-IG-002"}
        ],
        curves=[],
        consensus=fixture_market["consensus"],
        calendar=fixture_market["calendar"],
    )

    curve_contexts = build_contexts(pm, _VOICE, curve_only_lookup, catalogue, Config())
    credit_contexts = build_contexts(pm, _VOICE, credit_only_lookup, catalogue, Config())

    assert curve_contexts[0].question_instrument.kind == InstrumentKind.SOVEREIGN_CURVE
    assert credit_contexts[0].question_instrument.kind == InstrumentKind.CREDIT_ISSUER


def test_silence_session_with_no_matching_instrument_raises(fixture_market, catalogue):
    equities_only_lookup = MarketLookup.build(
        seed="T",
        instruments=fixture_market["instruments"],
        prices=[p for p in fixture_market["prices"] if p.instrument_id.startswith("EQ-")],
        curves=[],
        consensus=fixture_market["consensus"],
        calendar=fixture_market["calendar"],
    )
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a", kind=SessionKind.SILENCE, session_date=DEFAULT_DATE
    )
    pm = _pm_tables(asset_class=AssetClass.COMMODITIES, skeletons=(skeleton,))

    with pytest.raises(DialogueError):
        build_contexts(pm, _VOICE, equities_only_lookup, catalogue, Config())


def test_avoid_lines_render_forbidden_traits_and_preference_params(market_lookup, catalogue):
    idea = idea_row(trade_idea_id="ti_001")
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a",
        trade_idea_ids=("ti_001",),
        session_date=DEFAULT_DATE,
        forbidden_trait_ids=("t_01",),
        forbidden_pref_params=("register",),
    )
    pm = _pm_tables(
        traits=(bias_trait("loss_aversion_lambda"),),
        ideas={"ti_001": idea},
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


def test_position_day_for_unknown_idea_raises(market_lookup, catalogue):
    skeleton = _skeleton(session_id="s_pm001_2026-01-05_a", session_date=DEFAULT_DATE)
    stray = position_day(trade_idea_id="ti_999", date=DEFAULT_DATE)
    pm = _pm_tables(position_days=(stray,), skeletons=(skeleton,))

    with pytest.raises(DialogueError):
        build_contexts(pm, _VOICE, market_lookup, catalogue, Config())


def test_forbidden_trait_id_the_pm_lacks_raises(market_lookup, catalogue):
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a",
        session_date=DEFAULT_DATE,
        forbidden_trait_ids=("t_99",),
    )
    pm = _pm_tables(skeletons=(skeleton,))

    with pytest.raises(DialogueError):
        build_contexts(pm, _VOICE, market_lookup, catalogue, Config())


def test_forbidden_pref_param_missing_from_the_catalogue_raises(market_lookup, catalogue):
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a",
        session_date=DEFAULT_DATE,
        forbidden_pref_params=("not_a_real_param",),
    )
    pm = _pm_tables(skeletons=(skeleton,))

    with pytest.raises(DialogueError):
        build_contexts(pm, _VOICE, market_lookup, catalogue, Config())


def test_rules_are_filtered_to_scope_and_the_sessions_ideas_and_appear_in_the_system_prompt(
    market_lookup, catalogue
):
    idea_1 = idea_row(trade_idea_id="ti_001")
    idea_2 = idea_row(trade_idea_id="ti_002", instrument_id="EQ-0002")
    pm_rule_a = rule(rule_id="r_02", text="Cap risk at the mandate ceiling.")
    pm_rule_b = rule(rule_id="r_01", text="Never average into a loser.")
    idea_1_rule = rule(
        rule_id="r_03",
        text="Trim ti_001 by half at target.",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_001",
    )
    other_idea_rule = rule(
        rule_id="r_04",
        text="Roll ti_002 before expiry.",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_002",
    )
    skeleton = _skeleton(
        session_id="s_pm001_2026-01-05_a", trade_idea_ids=("ti_001",), session_date=DEFAULT_DATE
    )
    pm = _pm_tables(
        rules=(pm_rule_a, pm_rule_b, idea_1_rule, other_idea_rule),
        ideas={"ti_001": idea_1, "ti_002": idea_2},
        skeletons=(skeleton,),
    )

    contexts = build_contexts(pm, _VOICE, market_lookup, catalogue, Config())
    ctx = contexts[0]

    assert ctx.pm_rules == (pm_rule_b, pm_rule_a)  # rule_id order: r_01 before r_02
    assert ctx.idea_rules == (idea_1_rule,)  # the other idea's rule is excluded

    system = narrator_system(ctx, None)
    assert pm_rule_a.text in system
    assert pm_rule_b.text in system
    assert idea_1_rule.text in system
    assert other_idea_rule.text not in system


# --- annotate_day_trades ----------------------------------------------------------------------

_LATER = DEFAULT_DATE + timedelta(days=7)


def _notes(trades, position_days=(), rules=(), horizon_end=None, ideas=None):
    idea = idea_row(entry_date=DEFAULT_DATE)
    return annotate_day_trades(
        trades, ideas or {idea.trade_idea_id: idea}, position_days, rules, horizon_end
    )


def test_trade_on_the_entry_date_opens_the_position_with_no_trigger():
    trade = ledger_row(date=DEFAULT_DATE)

    assert _notes((trade,)) == {trade_key(trade): TradeNote("opens the position", None)}


@pytest.mark.parametrize(
    ("action", "kind"),
    [
        (PositionAction.ADD, "adds to the position"),
        (PositionAction.TRIM, "trims the position"),
        (PositionAction.EXIT, "closes the whole position"),
        (PositionAction.CUT, "closes the whole position"),
        (PositionAction.ROLL, "rolls the position"),
    ],
)
def test_trade_kind_follows_the_position_day_action(action, kind):
    trade = ledger_row(date=_LATER, side=Side.SELL)
    day = position_day(date=_LATER, action=action)

    note = _notes((trade,), position_days=(day,))[trade_key(trade)]

    assert note.kind == kind
    assert note.trigger == "on your own call"


def test_trade_trigger_quotes_the_rule_behind_rule_id():
    trade = ledger_row(date=_LATER, side=Side.SELL, rule_id="r_01")
    stop = rule(rule_id="r_01", text="Exit a position after a 5 percent drawdown from entry.")

    note = _notes((trade,), rules=(stop,))[trade_key(trade)]

    assert note.trigger == "on your rule: Exit a position after a 5 percent drawdown from entry."


def test_trade_on_the_horizon_end_is_triggered_by_the_horizon():
    trade = ledger_row(date=_LATER, side=Side.SELL)

    note = _notes((trade,), horizon_end=_LATER)[trade_key(trade)]

    assert note.trigger == "at the end of the horizon"


def test_trade_without_a_position_day_is_read_off_its_side():
    sell = ledger_row(date=_LATER, side=Side.SELL)
    buy = ledger_row(date=_LATER, side=Side.BUY)

    notes = _notes((sell, buy))

    assert notes[trade_key(sell)].kind == "closes the whole position"
    assert notes[trade_key(buy)].kind == "adds to the position"


def test_trade_naming_an_unknown_idea_or_rule_raises():
    with pytest.raises(DialogueError, match="trade idea 'ti_009'"):
        _notes((ledger_row(trade_idea_id="ti_009"),))
    with pytest.raises(DialogueError, match="rule 'r_99'"):
        _notes((ledger_row(date=_LATER, rule_id="r_99"),))
