"""Tests for `assemble`: placing one PM's planned signals on dated sessions."""

import re

import numpy as np
import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.enums import (
    CarrierSource,
    DriftEventType,
    Ownership,
    SessionKind,
    SignalMode,
    StanceEntry,
    Valence,
)
from pm_traitbench.errors import PlanError
from pm_traitbench.signals.assemble import Assembly, assemble, session_id
from pm_traitbench.signals.carriers import Carrier, carrier_pools
from pm_traitbench.signals.quotas import DateWindow, PlannedSignal, plan_quotas
from pm_traitbench.tables.schema import _SESSION_ID_PATTERN, Signal
from tests.signals.conftest import (
    TRADING_DAYS,
    bias_trait,
    drift_event,
    ledger_row,
    plan_inputs,
)

_SESSION_ID_RE = re.compile(_SESSION_ID_PATTERN)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _stated(
    trait_id: str, window: DateWindow, entry: StanceEntry = StanceEntry.STATED
) -> PlannedSignal:
    return PlannedSignal(
        trait_id=trait_id,
        mode=SignalMode.STATED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=entry,
        window=window,
        needs_carrier=False,
    )


def _revealed(
    trait_id: str, window: DateWindow, mode: SignalMode = SignalMode.REVEALED
) -> PlannedSignal:
    return PlannedSignal(
        trait_id=trait_id,
        mode=mode,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.REVEALED,
        window=window,
        needs_carrier=True,
    )


# --- carrier placement -------------------------------------------------------------------


def test_carrier_signals_land_on_carrier_dates_with_no_reuse():
    days = TRADING_DAYS[:30]
    ledger_rows = tuple(
        ledger_row(date=d, trade_idea_id=f"ti_{i:03d}", bias_flag="loss_aversion:add")
        for i, d in enumerate(days, start=1)
    )
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),), ledger=ledger_rows
    )
    pools = carrier_pools(inputs)
    knobs = Config().plan
    catalogue = load_catalogue()
    planned = plan_quotas(inputs, pools, catalogue, knobs, _rng(1))
    assembly = assemble(inputs, planned, pools, knobs, _rng(2))

    carrier_placements = [
        ps for session in assembly.sessions for ps in session.signals if ps.carrier is not None
    ]
    assert carrier_placements
    for ps in carrier_placements:
        assert ps.date == ps.carrier.date
        assert ps.carrier in pools[ps.planned.trait_id]
    assert len(carrier_placements) == len({ps.carrier for ps in carrier_placements})


def test_two_carriers_and_six_signals_places_two_and_warns_four_times():
    inputs = plan_inputs()
    trait_id = "t_01"
    c1 = Carrier(trait_id, "ti_001", TRADING_DAYS[0], CarrierSource.LEDGER, "add")
    c2 = Carrier(trait_id, "ti_002", TRADING_DAYS[1], CarrierSource.LEDGER, "add")
    pools = {trait_id: (c1, c2)}
    window = DateWindow(TRADING_DAYS[0], TRADING_DAYS[5])
    planned = [_revealed(trait_id, window) for _ in range(6)]
    knobs = Config().plan
    assembly = assemble(inputs, planned, pools, knobs, _rng(0))

    placed = [row for row in assembly.signals if row.trait_id == trait_id]
    assert len(placed) == 2
    drop_warnings = [w for w in assembly.warnings if "dropped, no carrier" in w]
    assert len(drop_warnings) == 4
    assert all(
        w == f"{inputs.persona.pm_id} {trait_id}: revealed signal dropped, no carrier in "
        f"{window.first}..{window.last}"
        for w in drop_warnings
    )


# --- contradiction claims ------------------------------------------------------------------


def test_contradiction_carrier_filter_respects_claim_lead_days_floor():
    """A carrier inside the window but too close to the horizon start cannot back a
    contradiction (no room left for its claim), even though the very same carrier is
    eligible for a plain revealed signal.
    """
    inputs = plan_inputs()
    trait_id = "t_01"
    knobs = Config().plan
    early_date = TRADING_DAYS[2]  # trading-day index 2 < claim_lead_days[0] (10)
    carrier = Carrier(trait_id, "ti_001", early_date, CarrierSource.LEDGER, "add")
    pools = {trait_id: (carrier,)}
    window = DateWindow(TRADING_DAYS[0], TRADING_DAYS[-1])

    contradiction_only = [_revealed(trait_id, window, mode=SignalMode.CONTRADICTION)]
    dropped = assemble(inputs, contradiction_only, pools, knobs, _rng(32))
    assert not any(row.trait_id == trait_id for row in dropped.signals)
    assert any("dropped, no carrier" in w for w in dropped.warnings)

    revealed_only = [_revealed(trait_id, window, mode=SignalMode.REVEALED)]
    placed_assembly = assemble(inputs, revealed_only, pools, knobs, _rng(33))
    placed = [row for row in placed_assembly.signals if row.trait_id == trait_id]
    assert len(placed) == 1
    assert placed[0].date == early_date


def test_contradiction_dropped_for_no_claim_day_leaves_carrier_unused():
    """Every trading day in the carrier's whole claim window is dormant for the trait, so no
    claim day qualifies even though the carrier itself clears the `claim_lead_days[0]` floor.
    """
    trait_id = "t_01"
    inputs = plan_inputs(
        drift_events=(drift_event(trait_id, TRADING_DAYS[0], DriftEventType.DORMANT),)
    )
    knobs = Config().plan
    carrier_date = TRADING_DAYS[15]
    carrier = Carrier(trait_id, "ti_001", carrier_date, CarrierSource.LEDGER, "add")
    pools = {trait_id: (carrier,)}
    window = DateWindow(TRADING_DAYS[0], TRADING_DAYS[-1])
    planned = [_revealed(trait_id, window, mode=SignalMode.CONTRADICTION)]
    assembly = assemble(inputs, planned, pools, knobs, _rng(30))

    assert not any(row.trait_id == trait_id for row in assembly.signals)
    assert any(
        "contradiction signal dropped, no claim day available" in w for w in assembly.warnings
    )
    # a session created only to hold the dropped contradiction's stance is removed, so the
    # carrier's own date is left with no session at all: the carrier was never consumed.
    assert not any(s.date == carrier_date for s in assembly.sessions)


def test_contradiction_claim_precedes_carrier_by_10_to_40_trading_days():
    inputs = plan_inputs()
    trait_id = "t_01"
    knobs = Config().plan
    carrier_date = TRADING_DAYS[100]
    carrier = Carrier(trait_id, "ti_001", carrier_date, CarrierSource.LEDGER, "add")
    pools = {trait_id: (carrier,)}
    window = DateWindow(TRADING_DAYS[0], TRADING_DAYS[-1])
    planned = [_revealed(trait_id, window, mode=SignalMode.CONTRADICTION)]
    assembly = assemble(inputs, planned, pools, knobs, _rng(3))

    row = next(r for r in assembly.signals if r.trait_id == trait_id)
    assert row.mode == SignalMode.CONTRADICTION
    assert row.claim_session_id is not None
    claim_session = next(s for s in assembly.sessions if s.session_id == row.claim_session_id)
    assert claim_session.date < carrier_date
    lead = TRADING_DAYS.index(carrier_date) - TRADING_DAYS.index(claim_session.date)
    assert 10 <= lead <= 40
    assert any(ps.signal_id == row.signal_id for ps in claim_session.claims)


# --- capacity --------------------------------------------------------------------------


def test_capacity_rules_hold_across_sessions():
    inputs = plan_inputs()
    knobs = Config().plan
    window = DateWindow(TRADING_DAYS[0], TRADING_DAYS[0])
    planned = [_stated("t_01", window) for _ in range(3)]
    planned += [_stated(f"t_0{i}", window, entry=StanceEntry.REVEALED_REACTION) for i in (2, 3, 4)]
    assembly = assemble(inputs, planned, {}, knobs, _rng(5))

    for session in assembly.sessions:
        stances = [*session.signals, *session.claims]
        assert len(stances) <= knobs.max_signals_per_session
        trait_ids = [ps.planned.trait_id for ps in stances]
        assert len(trait_ids) == len(set(trait_ids))
        reactions = [ps for ps in stances if ps.planned.entry == StanceEntry.REVEALED_REACTION]
        assert len(reactions) <= 1


def test_cap_of_one_isolates_every_stance():
    inputs = plan_inputs()
    knobs = Config().plan.model_copy(update={"max_signals_per_session": 1})
    window = DateWindow(TRADING_DAYS[0], TRADING_DAYS[3])
    planned = [_stated(f"t_0{i}", window) for i in range(1, 5)]
    assembly = assemble(inputs, planned, {}, knobs, _rng(6))

    signal_sessions = [s for s in assembly.sessions if s.signals or s.claims]
    assert signal_sessions
    for session in signal_sessions:
        assert len(session.signals) + len(session.claims) == 1


# --- packable ----------------------------------------------------------------------------


def test_packable_signals_avoid_dormant_windows_and_stay_in_window():
    inputs = plan_inputs(
        drift_events=(
            drift_event("t_01", TRADING_DAYS[10], DriftEventType.DORMANT),
            drift_event("t_01", TRADING_DAYS[20], DriftEventType.REVIVE),
        )
    )
    knobs = Config().plan
    window = DateWindow(TRADING_DAYS[5], TRADING_DAYS[25])
    planned = [_stated("t_01", window) for _ in range(8)]
    assembly = assemble(inputs, planned, {}, knobs, _rng(7))

    placed = [row for row in assembly.signals if row.trait_id == "t_01"]
    assert placed
    for row in placed:
        assert window.first <= row.date <= window.last
        assert not inputs.is_dormant("t_01", row.date)


# --- drift notes -------------------------------------------------------------------------


def test_drift_note_opens_new_session_when_none_has_room():
    inputs = plan_inputs()
    knobs = Config().plan.model_copy(update={"max_signals_per_session": 1})
    drift_date = TRADING_DAYS[30]
    other = _stated("t_02", DateWindow(drift_date, drift_date))
    note = PlannedSignal(
        trait_id="t_01",
        mode=SignalMode.STATED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.DRIFT_UPDATE,
        window=DateWindow(drift_date, TRADING_DAYS[-1]),
        needs_carrier=False,
        drift_date=drift_date,
    )
    assembly = assemble(inputs, [other, note], {}, knobs, _rng(31))

    note_row = next(r for r in assembly.signals if r.trait_id == "t_01")
    other_row = next(r for r in assembly.signals if r.trait_id == "t_02")
    assert note_row.date == drift_date
    assert note_row.session_id != other_row.session_id
    assert len([s for s in assembly.sessions if s.date == drift_date]) >= 2


def test_drift_note_joins_earliest_existing_session_on_or_after_its_date():
    inputs = plan_inputs()
    knobs = Config().plan
    drift_date = TRADING_DAYS[50]
    later_day = TRADING_DAYS[55]
    other = _stated("t_02", DateWindow(later_day, later_day))
    note = PlannedSignal(
        trait_id="t_01",
        mode=SignalMode.STATED,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        entry=StanceEntry.DRIFT_UPDATE,
        window=DateWindow(drift_date, TRADING_DAYS[-1]),
        needs_carrier=False,
        drift_date=drift_date,
    )
    assembly = assemble(inputs, [other, note], {}, knobs, _rng(8))

    note_row = next(r for r in assembly.signals if r.trait_id == "t_01")
    other_row = next(r for r in assembly.signals if r.trait_id == "t_02")
    assert note_row.date == later_day
    assert note_row.session_id == other_row.session_id


# --- ledger sessions ---------------------------------------------------------------------


def test_ledger_sessions_hold_qualifying_orders_and_are_decisions():
    days = TRADING_DAYS[:10]
    rows = tuple(
        ledger_row(date=d, trade_idea_id=f"ti_{i:03d}", risk_amount=float(i))
        for i, d in enumerate(days, start=1)
    )
    inputs = plan_inputs(ledger=rows)
    knobs = Config().plan
    assembly = assemble(inputs, [], {}, knobs, _rng(9))

    threshold = np.percentile([r.risk_amount for r in rows], knobs.ledger_session_percentile)
    qualifying = {r.date: r.trade_idea_id for r in rows if r.risk_amount >= threshold}
    assert qualifying
    for day, idea_id in qualifying.items():
        matches = [s for s in assembly.sessions if s.date == day and idea_id in s.trade_idea_ids]
        assert matches
        assert all(s.kind == SessionKind.DECISION for s in matches)


# --- filler ------------------------------------------------------------------------------


def test_filler_keeps_signal_session_share_within_cap_when_free_days_suffice():
    inputs = plan_inputs()
    knobs = Config().plan
    window = DateWindow(TRADING_DAYS[0], TRADING_DAYS[50])
    planned = [_stated(f"t_{i % 8 + 1:02d}", window) for i in range(20)]
    assembly = assemble(inputs, planned, {}, knobs, _rng(10))

    signal_sessions = sum(1 for s in assembly.sessions if s.signals or s.claims)
    total = len(assembly.sessions)
    assert signal_sessions / total <= knobs.signal_session_cap + 1e-9


def test_filler_warns_when_cap_cannot_be_met_on_a_short_horizon():
    days = TRADING_DAYS[:5]
    inputs = plan_inputs(trading_days=days)
    knobs = Config().plan
    planned = [_stated(f"t_{i:02d}", DateWindow(d, d)) for i, d in enumerate(days, start=1)]
    assembly = assemble(inputs, planned, {}, knobs, _rng(11))

    assert any("signal_session_cap cannot be met" in w for w in assembly.warnings)


# --- ids ---------------------------------------------------------------------------------


def test_session_and_signal_ids():
    inputs = plan_inputs()
    knobs = Config().plan
    window = DateWindow(TRADING_DAYS[0], TRADING_DAYS[0])
    planned = [_stated(f"t_{i:02d}", window) for i in range(1, 5)]
    assembly = assemble(inputs, planned, {}, knobs, _rng(12))

    for session in assembly.sessions:
        assert _SESSION_ID_RE.fullmatch(session.session_id)
    same_day = sorted(
        (s for s in assembly.sessions if s.date == TRADING_DAYS[0]), key=lambda s: s.session_id
    )
    assert same_day[0].session_id.endswith("_a")

    ids = [row.signal_id for row in assembly.signals]
    assert ids == [f"sg_{n:03d}" for n in range(1, len(ids) + 1)]


def test_session_id_past_z_raises_plan_error():
    with pytest.raises(PlanError):
        session_id("pm_001", TRADING_DAYS[0], 26)
    assert session_id("pm_001", TRADING_DAYS[0], 0).endswith("_a")


# --- counts ------------------------------------------------------------------------------


def test_counts_are_internally_consistent():
    days = TRADING_DAYS[:200]
    ledger_rows = tuple(
        ledger_row(
            date=d,
            trade_idea_id=f"ti_{i:03d}",
            bias_flag="loss_aversion:add",
            risk_amount=float(i),
        )
        for i, d in enumerate(days, start=1)
    )
    update_date = days[100]
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        drift_events=(
            drift_event("t_01", update_date, DriftEventType.UPDATE, from_value=0.7, to_value=1.2),
        ),
        ledger=ledger_rows,
        trading_days=days,
    )
    pools = carrier_pools(inputs)
    knobs = Config().plan
    catalogue = load_catalogue()
    planned = plan_quotas(inputs, pools, catalogue, knobs, _rng(20))
    assembly = assemble(inputs, planned, pools, knobs, _rng(21))
    counts = assembly.counts

    assert list(counts.keys()) == sorted(counts.keys())

    assert sum(counts["sessions_by_kind"].values()) == len(assembly.sessions)

    signal_sessions = sum(1 for s in assembly.sessions if s.signals or s.claims)
    assert counts["signal_session_share"] == round(signal_sessions / len(assembly.sessions), 4)

    assert counts["revealed_planned"]
    for trait_id, placed in counts["revealed_placed"].items():
        assert placed <= counts["revealed_planned"][trait_id]

    assert "t_01" in counts["segments"]
    assert len(counts["segments"]["t_01"]) == 2


# --- validity and determinism -------------------------------------------------------------


def test_rows_validate_and_same_seed_gives_equal_assembly():
    days = TRADING_DAYS[:120]
    ledger_rows = tuple(
        ledger_row(
            date=d,
            trade_idea_id=f"ti_{i:03d}",
            bias_flag="loss_aversion:add",
            risk_amount=float(i),
        )
        for i, d in enumerate(days[:20], start=1)
    )
    inputs = plan_inputs(
        traits=(
            bias_trait("loss_aversion_lambda", trait_id="t_01"),
            bias_trait("disposition_ratio", trait_id="t_02", active=False),
        ),
        drift_events=(
            drift_event("t_01", days[60], DriftEventType.DORMANT),
            drift_event("t_01", days[80], DriftEventType.REVIVE),
        ),
        ledger=ledger_rows,
        trading_days=days,
    )
    pools = carrier_pools(inputs)
    knobs = Config().plan
    catalogue = load_catalogue()
    planned = plan_quotas(inputs, pools, catalogue, knobs, _rng(42))

    first = assemble(inputs, planned, pools, knobs, np.random.default_rng(42))
    second = assemble(inputs, planned, pools, knobs, np.random.default_rng(42))

    assert isinstance(first, Assembly)
    for row in first.signals:
        assert isinstance(row, Signal)
    assert first == second
