"""Tests for `skeleton`: rendering planned sessions into hidden `Skeleton` rows."""

import numpy as np

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.enums import (
    CarrierSource,
    DriftEventType,
    Ownership,
    SessionKind,
    SignalMode,
    StanceEntry,
    Valence,
)
from pm_traitbench.signals.assemble import Assembly, PlacedSignal, PlannedSession, assemble
from pm_traitbench.signals.assemble import session_id as make_session_id
from pm_traitbench.signals.carriers import Carrier, carrier_pools
from pm_traitbench.signals.quotas import DateWindow, PlannedSignal, plan_quotas
from pm_traitbench.signals.skeleton import forbidden_sets, format_level, render_skeletons
from pm_traitbench.tables.schema import Skeleton
from tests.signals.conftest import (
    TRADING_DAYS,
    bias_trait,
    drift_event,
    idea_row,
    ledger_row,
    plan_inputs,
    pref_trait,
)

_CATALOGUE = load_catalogue()
_PM_ID = "pm_001"


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sid(day, letter: int = 0) -> str:
    return make_session_id(_PM_ID, day, letter)


# --- forbidden_sets -------------------------------------------------------------------------


def test_forbidden_sets_lists_only_inactive_biases_and_unheld_asset_class_prefs():
    active_param = BIAS_PARAMS[0]
    traits = tuple(
        bias_trait(param, active=(param == active_param), trait_id=f"t_{i:02d}")
        for i, param in enumerate(BIAS_PARAMS, start=1)
    )
    traits = (*traits, pref_trait("response_format", "short bullets"))
    inputs = plan_inputs(traits=traits)

    inactive, unheld = forbidden_sets(inputs, _CATALOGUE)

    inactive_params = {p for p in BIAS_PARAMS if p != active_param}
    assert len(inactive) == len(inactive_params)
    assert inactive == tuple(sorted(inactive))
    assert "t_01" not in inactive  # the active trait's own id must not show up

    assert "response_format" not in unheld  # held by the PM
    assert "pushback_style" in unheld  # applicable to equities, never held
    assert "duration_expression" not in unheld  # rates_credit only, not this PM's asset class
    assert unheld == tuple(sorted(unheld))


def test_forbidden_sets_appear_on_every_skeleton_even_with_a_third_party_stance():
    inactive_trait_id = "t_01"
    traits = tuple(
        bias_trait(param, active=(param != "loss_aversion_lambda"), trait_id=f"t_{i:02d}")
        for i, param in enumerate(BIAS_PARAMS, start=1)
    )
    inputs = plan_inputs(traits=traits)
    expected_inactive, expected_unheld = forbidden_sets(inputs, _CATALOGUE)
    assert inactive_trait_id in expected_inactive

    day, silence_day = TRADING_DAYS[0], TRADING_DAYS[1]
    planned = PlannedSignal(
        trait_id=inactive_trait_id,
        mode=SignalMode.STATED,
        valence=Valence.CONFIRM,
        ownership=Ownership.COLLEAGUE,
        entry=StanceEntry.THIRD_PARTY,
        window=DateWindow(day, day),
        needs_carrier=False,
    )
    ps = PlacedSignal(
        signal_id="sg_001",
        planned=planned,
        date=day,
        trade_idea_id=None,
        claim_date=None,
        carrier=None,
    )
    third_party_session = PlannedSession(
        session_id=_sid(day),
        date=day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(),
        signals=(ps,),
        claims=(),
    )
    silence_session = PlannedSession(
        session_id=_sid(silence_day),
        date=silence_day,
        kind=SessionKind.SILENCE,
        trade_idea_ids=(),
        signals=(),
        claims=(),
    )
    assembly = Assembly(
        sessions=(third_party_session, silence_session), signals=(), warnings=(), counts={}
    )

    skeletons = render_skeletons(inputs, assembly, _CATALOGUE, _rng())

    assert len(skeletons) == 2
    for skeleton in skeletons:
        assert skeleton.forbidden_trait_ids == expected_inactive
        assert skeleton.forbidden_pref_params == expected_unheld

    # the third-party stance still names the forbidden trait; it stays forbidden regardless.
    third_party_skeleton = skeletons[0]
    assert third_party_skeleton.stances[0].trait_id == inactive_trait_id
    assert inactive_trait_id in third_party_skeleton.forbidden_trait_ids


# --- revealed bias stance ---------------------------------------------------------------------


def _revealed_bias_stance(trait_param: str, trait_id: str, pattern: str, idea):
    day = idea.entry_date
    carrier = Carrier(trait_id, idea.trade_idea_id, day, CarrierSource.LEDGER, pattern)
    planned = PlannedSignal(
        trait_id=trait_id,
        mode=SignalMode.REVEALED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.REVEALED,
        window=DateWindow(day, day),
        needs_carrier=True,
    )
    ps = PlacedSignal(
        signal_id="sg_001",
        planned=planned,
        date=day,
        trade_idea_id=idea.trade_idea_id,
        claim_date=None,
        carrier=carrier,
    )
    session = PlannedSession(
        session_id=_sid(day),
        date=day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(idea.trade_idea_id,),
        signals=(ps,),
        claims=(),
    )
    inputs = plan_inputs(
        traits=(bias_trait(trait_param, trait_id=trait_id),), ideas={idea.trade_idea_id: idea}
    )
    assembly = Assembly(sessions=(session,), signals=(), warnings=(), counts={})
    skeletons = render_skeletons(inputs, assembly, _CATALOGUE, _rng())
    return skeletons[0].stances[0].stance


def test_revealed_bias_stance_names_carrier_instrument_and_entry_level():
    # Every 'add' line for loss_aversion_lambda quotes both {instrument} and {entry}.
    idea = idea_row(
        trade_idea_id="ti_001",
        entry_date=TRADING_DAYS[0],
        instrument_id="EQ-0007",
        entry_level=123.456,
    )
    text = _revealed_bias_stance("loss_aversion_lambda", "t_01", "add", idea)
    assert "EQ-0007" in text
    assert format_level(123.456) in text
    assert "{" not in text


def test_revealed_bias_stance_names_carrier_target_level():
    # Every 'realise_gain_early' line for disposition_ratio quotes {instrument} and {target}.
    idea = idea_row(
        trade_idea_id="ti_002",
        entry_date=TRADING_DAYS[0],
        instrument_id="EQ-0008",
        target_level=150.0,
    )
    text = _revealed_bias_stance("disposition_ratio", "t_02", "realise_gain_early", idea)
    assert "EQ-0008" in text
    assert format_level(150.0) in text
    assert "{" not in text


# --- claim and revealed pair on separate sessions ---------------------------------------------


def test_claim_and_carrier_stances_share_the_contradiction_signal_id():
    claim_day = TRADING_DAYS[0]
    carrier_day = TRADING_DAYS[5]
    idea = idea_row(trade_idea_id="ti_001", entry_date=carrier_day)
    carrier = Carrier("t_01", "ti_001", carrier_day, CarrierSource.LEDGER, "add")
    planned = PlannedSignal(
        trait_id="t_01",
        mode=SignalMode.CONTRADICTION,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.REVEALED,
        window=DateWindow(carrier_day, carrier_day),
        needs_carrier=True,
    )
    ps = PlacedSignal(
        signal_id="sg_002",
        planned=planned,
        date=carrier_day,
        trade_idea_id="ti_001",
        claim_date=claim_day,
        carrier=carrier,
    )
    claim_session = PlannedSession(
        session_id=_sid(claim_day),
        date=claim_day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(),
        signals=(),
        claims=(ps,),
    )
    main_session = PlannedSession(
        session_id=_sid(carrier_day),
        date=carrier_day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=("ti_001",),
        signals=(ps,),
        claims=(),
    )
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),), ideas={"ti_001": idea}
    )
    assembly = Assembly(sessions=(claim_session, main_session), signals=(), warnings=(), counts={})

    skeletons = render_skeletons(inputs, assembly, _CATALOGUE, _rng())

    claim_stance = skeletons[0].stances[0]
    revealed_stance = skeletons[1].stances[0]
    assert claim_stance.entry == StanceEntry.CLAIM
    assert claim_stance.mode == SignalMode.CONTRADICTION
    assert claim_stance.signal_id == "sg_002"
    assert revealed_stance.entry == StanceEntry.REVEALED
    assert revealed_stance.signal_id == "sg_002"


# --- preference drift -------------------------------------------------------------------------


def test_preference_stance_after_drift_uses_new_value_and_note_names_both():
    drift_day = TRADING_DAYS[10]
    after_day = TRADING_DAYS[20]

    trait = pref_trait("response_format", "short bullets", trait_id="t_90")
    event = drift_event(
        "t_90",
        drift_day,
        DriftEventType.UPDATE,
        from_value="short bullets",
        to_value="a table with columns",
    )
    inputs = plan_inputs(traits=(trait,), drift_events=(event,))

    stated_planned = PlannedSignal(
        trait_id="t_90",
        mode=SignalMode.STATED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.STATED,
        window=DateWindow(after_day, after_day),
        needs_carrier=False,
    )
    stated_ps = PlacedSignal(
        signal_id="sg_001",
        planned=stated_planned,
        date=after_day,
        trade_idea_id=None,
        claim_date=None,
        carrier=None,
    )
    stated_session = PlannedSession(
        session_id=_sid(after_day),
        date=after_day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(),
        signals=(stated_ps,),
        claims=(),
    )

    note_planned = PlannedSignal(
        trait_id="t_90",
        mode=SignalMode.STATED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.DRIFT_UPDATE,
        window=DateWindow(drift_day, after_day),
        needs_carrier=False,
        drift_date=drift_day,
    )
    note_ps = PlacedSignal(
        signal_id="sg_002",
        planned=note_planned,
        date=drift_day,
        trade_idea_id=None,
        claim_date=None,
        carrier=None,
    )
    note_session = PlannedSession(
        session_id=_sid(drift_day),
        date=drift_day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(),
        signals=(note_ps,),
        claims=(),
    )

    assembly = Assembly(sessions=(note_session, stated_session), signals=(), warnings=(), counts={})
    skeletons = render_skeletons(inputs, assembly, _CATALOGUE, _rng())

    note_text = skeletons[0].stances[0].stance
    stated_text = skeletons[1].stances[0].stance
    assert "short bullets" in note_text
    assert "a table with columns" in note_text
    assert "a table with columns" in stated_text


# --- third-party preference --------------------------------------------------------------------


def test_third_party_preference_stance_names_value_and_who():
    day = TRADING_DAYS[0]

    trait = pref_trait("response_format", "short bullets", trait_id="t_90")
    planned = PlannedSignal(
        trait_id="t_90",
        mode=SignalMode.STATED,
        valence=Valence.CONFIRM,
        ownership=Ownership.CLIENT,
        entry=StanceEntry.THIRD_PARTY,
        window=DateWindow(day, day),
        needs_carrier=False,
        third_party_value="a table with columns",
    )
    ps = PlacedSignal(
        signal_id="sg_001",
        planned=planned,
        date=day,
        trade_idea_id=None,
        claim_date=None,
        carrier=None,
    )
    session = PlannedSession(
        session_id=_sid(day),
        date=day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(),
        signals=(ps,),
        claims=(),
    )
    inputs = plan_inputs(traits=(trait,))
    assembly = Assembly(sessions=(session,), signals=(), warnings=(), counts={})

    skeletons = render_skeletons(inputs, assembly, _CATALOGUE, _rng())
    text = skeletons[0].stances[0].stance
    assert "a table with columns" in text
    assert "a client" in text


# --- advisor violation ------------------------------------------------------------------------


def test_advisor_violation_set_exactly_with_revealed_reaction_and_names_value():
    day = TRADING_DAYS[0]

    trait = pref_trait("response_format", "short bullets", trait_id="t_90")
    reaction_planned = PlannedSignal(
        trait_id="t_90",
        mode=SignalMode.REVEALED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.REVEALED_REACTION,
        window=DateWindow(day, day),
        needs_carrier=False,
    )
    reaction_ps = PlacedSignal(
        signal_id="sg_001",
        planned=reaction_planned,
        date=day,
        trade_idea_id=None,
        claim_date=None,
        carrier=None,
    )
    reaction_session = PlannedSession(
        session_id=_sid(day),
        date=day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(),
        signals=(reaction_ps,),
        claims=(),
    )
    other_day = TRADING_DAYS[1]
    stated_planned = PlannedSignal(
        trait_id="t_90",
        mode=SignalMode.STATED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.STATED,
        window=DateWindow(other_day, other_day),
        needs_carrier=False,
    )
    stated_ps = PlacedSignal(
        signal_id="sg_002",
        planned=stated_planned,
        date=other_day,
        trade_idea_id=None,
        claim_date=None,
        carrier=None,
    )
    plain_session = PlannedSession(
        session_id=_sid(other_day),
        date=other_day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(),
        signals=(stated_ps,),
        claims=(),
    )
    inputs = plan_inputs(traits=(trait,))
    assembly = Assembly(
        sessions=(reaction_session, plain_session), signals=(), warnings=(), counts={}
    )

    skeletons = render_skeletons(inputs, assembly, _CATALOGUE, _rng())

    assert skeletons[0].advisor_violation is not None
    assert "short bullets" in skeletons[0].advisor_violation
    assert skeletons[1].advisor_violation is None


# --- validation, silence and determinism -------------------------------------------------------


def test_silence_skeleton_has_no_stances_and_full_run_validates():
    days = TRADING_DAYS[:30]
    ledger_rows = tuple(
        ledger_row(date=d, trade_idea_id=f"ti_{i:03d}", bias_flag="loss_aversion:add")
        for i, d in enumerate(days, start=1)
    )
    ideas = {
        row.trade_idea_id: idea_row(trade_idea_id=row.trade_idea_id, entry_date=row.date)
        for row in ledger_rows
    }
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        ledger=ledger_rows,
        ideas=ideas,
    )
    pools = carrier_pools(inputs)
    knobs = Config().plan
    planned = plan_quotas(inputs, pools, _CATALOGUE, knobs, _rng(1))
    assembly = assemble(inputs, planned, pools, knobs, _rng(2))

    silence_session = PlannedSession(
        session_id=_sid(TRADING_DAYS[100]),
        date=TRADING_DAYS[100],
        kind=SessionKind.SILENCE,
        trade_idea_ids=(),
        signals=(),
        claims=(),
    )
    assembly = Assembly(
        sessions=(*assembly.sessions, silence_session),
        signals=assembly.signals,
        warnings=assembly.warnings,
        counts=assembly.counts,
    )

    skeletons = render_skeletons(inputs, assembly, _CATALOGUE, _rng(3))

    assert all(isinstance(s, Skeleton) for s in skeletons)
    by_id = {s.session_id: s for s in skeletons}
    silence_skeleton = by_id[silence_session.session_id]
    assert silence_skeleton.kind == SessionKind.SILENCE
    assert silence_skeleton.stances == ()
    assert silence_skeleton.trade_idea_ids == ()
    assert silence_skeleton.advisor_violation is None


def test_equal_seeds_give_equal_output():
    days = TRADING_DAYS[:30]
    ledger_rows = tuple(
        ledger_row(date=d, trade_idea_id=f"ti_{i:03d}", bias_flag="loss_aversion:add")
        for i, d in enumerate(days, start=1)
    )
    ideas = {
        row.trade_idea_id: idea_row(trade_idea_id=row.trade_idea_id, entry_date=row.date)
        for row in ledger_rows
    }
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        ledger=ledger_rows,
        ideas=ideas,
    )
    pools = carrier_pools(inputs)
    knobs = Config().plan
    planned = plan_quotas(inputs, pools, _CATALOGUE, knobs, _rng(1))
    assembly = assemble(inputs, planned, pools, knobs, _rng(2))

    skeletons_a = render_skeletons(inputs, assembly, _CATALOGUE, _rng(7))
    skeletons_b = render_skeletons(inputs, assembly, _CATALOGUE, _rng(7))

    assert skeletons_a == skeletons_b


# --- revealed preference (expression) stance -----------------------------------------------------


def test_revealed_preference_stance_names_value_and_instrument():
    day = TRADING_DAYS[0]

    trait = pref_trait("pair_vs_outright", "express the view as a pair trade", trait_id="t_90")
    idea = idea_row(trade_idea_id="ti_001", entry_date=day, instrument_id="EQ-0009")
    carrier = Carrier("t_90", "ti_001", day, CarrierSource.IDEA, None)
    planned = PlannedSignal(
        trait_id="t_90",
        mode=SignalMode.REVEALED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.REVEALED,
        window=DateWindow(day, day),
        needs_carrier=True,
    )
    ps = PlacedSignal(
        signal_id="sg_001",
        planned=planned,
        date=day,
        trade_idea_id="ti_001",
        claim_date=None,
        carrier=carrier,
    )
    session = PlannedSession(
        session_id=_sid(day),
        date=day,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=("ti_001",),
        signals=(ps,),
        claims=(),
    )
    inputs = plan_inputs(traits=(trait,), ideas={"ti_001": idea})
    assembly = Assembly(sessions=(session,), signals=(), warnings=(), counts={})

    skeletons = render_skeletons(inputs, assembly, _CATALOGUE, _rng())
    text = skeletons[0].stances[0].stance
    assert "EQ-0009" in text
    assert "express the view as a pair trade" in text
