"""Tests for `plan_quotas`: per-trait signal counts, modes and date windows for one PM."""

from collections import Counter

import numpy as np

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.enums import (
    CarrierSource,
    DriftEventType,
    Ownership,
    SignalMode,
    StanceEntry,
    Valence,
)
from pm_traitbench.signals.carriers import Carrier
from pm_traitbench.signals.quotas import (
    DateWindow,
    PlannedSignal,
    largest_remainder,
    plan_quotas,
    round_half_up,
)
from tests.signals.conftest import TRADING_DAYS, bias_trait, drift_event, plan_inputs, pref_trait


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _confirm(signals: list[PlannedSignal]) -> list[PlannedSignal]:
    return [s for s in signals if s.entry not in _NOTE_ENTRIES and s.valence == Valence.CONFIRM]


_NOTE_ENTRIES = {
    StanceEntry.DRIFT_UPDATE,
    StanceEntry.DRIFT_DORMANT,
    StanceEntry.DRIFT_REVIVE,
    StanceEntry.RETRACT,
    StanceEntry.THIRD_PARTY,
}


# --- largest_remainder / round_half_up -------------------------------------------------


def test_largest_remainder_sums_to_total() -> None:
    weights = [("a", 0.65), ("b", 0.175), ("c", 0.10)]
    result = largest_remainder(37, weights)
    assert sum(result.values()) == 37


def test_largest_remainder_known_case() -> None:
    weights = [("a", 0.65), ("b", 0.175), ("c", 0.10)]
    assert largest_remainder(10, weights) == {"a": 7, "b": 2, "c": 1}


def test_largest_remainder_ties_go_to_earlier_item() -> None:
    # 3 equal-weight items splitting 10: floors are 3, 3, 3 with all three fractional
    # parts tied at 1/3, so the one leftover unit goes to the earliest item, "a".
    weights = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
    result = largest_remainder(10, weights)
    assert sum(result.values()) == 10
    assert result["a"] == 4
    assert result["b"] == 3
    assert result["c"] == 3


def test_largest_remainder_zero_weight_gets_zero() -> None:
    weights = [("a", 0.5), ("b", 0.0), ("c", 0.5)]
    result = largest_remainder(9, weights)
    assert result["b"] == 0
    assert sum(result.values()) == 9


def test_round_half_up() -> None:
    assert round_half_up(0.5) == 1
    assert round_half_up(2.5) == 3
    assert round_half_up(2.4) == 2


# --- confirm counts ----------------------------------------------------------------------


def test_active_bias_gets_confirm_signals_in_range_with_mode_split() -> None:
    inputs = plan_inputs(traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),))
    knobs = Config().plan
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(1))
    confirm = _confirm(signals)
    assert 8 <= len(confirm) <= 10

    counts = Counter(s.mode for s in confirm)
    expected = largest_remainder(
        len(confirm),
        [
            (SignalMode.REVEALED, knobs.bias_revealed_weight),
            (SignalMode.STATED, knobs.bias_stated_weight),
            (SignalMode.CONTRADICTION, knobs.bias_contradiction_weight),
        ],
    )
    for mode, count in expected.items():
        assert counts.get(mode, 0) == count

    for s in confirm:
        assert s.valence == Valence.CONFIRM
        assert s.ownership == Ownership.SELF


def test_inactive_bias_gets_no_confirm_signals() -> None:
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01", active=False),)
    )
    knobs = Config().plan
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(1))
    assert _confirm(signals) == []


def test_fully_dormant_trait_gets_no_confirm_signals() -> None:
    """A trait dormant for its whole horizon (no revive) has zero segments, and must not
    crash trying to divide the confirm count across them.
    """
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        drift_events=(drift_event("t_01", TRADING_DAYS[0], DriftEventType.DORMANT),),
    )
    knobs = Config().plan
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(1))
    assert [s for s in _confirm(signals) if s.trait_id == "t_01"] == []
    dormant_note = next(s for s in signals if s.entry == StanceEntry.DRIFT_DORMANT)
    assert dormant_note.drift_date == TRADING_DAYS[0]


def test_preference_gets_exactly_pref_signals_count() -> None:
    inputs = plan_inputs(
        traits=(pref_trait("duration_expression", "steepeners over outright duration"),)
    )
    knobs = Config().plan
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(1))
    confirm = _confirm(signals)
    assert len(confirm) == knobs.pref_signals


def test_preference_without_carrier_pool_plans_revealed_reaction() -> None:
    inputs = plan_inputs(
        traits=(pref_trait("duration_expression", "steepeners over outright duration"),)
    )
    knobs = Config().plan
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(1))
    confirm = _confirm(signals)
    revealed = [s for s in confirm if s.mode == SignalMode.REVEALED]
    assert revealed
    for s in revealed:
        assert s.entry == StanceEntry.REVEALED_REACTION
        assert s.needs_carrier is False


def test_preference_with_nonempty_pool_plans_revealed_with_carrier() -> None:
    trait_id = "t_90"
    inputs = plan_inputs(
        traits=(pref_trait("duration_expression", "steepeners over outright duration"),)
    )
    knobs = Config().plan
    carrier = Carrier(trait_id, "ti_001", TRADING_DAYS[0], CarrierSource.IDEA, None)
    pools = {trait_id: (carrier,)}
    signals = plan_quotas(inputs, pools, load_catalogue(), knobs, _rng(1))
    confirm = _confirm(signals)
    revealed = [s for s in confirm if s.mode == SignalMode.REVEALED]
    assert revealed
    for s in revealed:
        assert s.entry == StanceEntry.REVEALED
        assert s.needs_carrier is True


def test_preference_drift_from_mapped_to_unmapped_form_uses_reaction_after_update() -> None:
    """Deciding needs_carrier once per trait would leave every revealed signal of the
    post-drift segment looking for a carrier the pool never has there; deciding it per
    segment instead falls back to an advisor reaction for that segment alone.
    """
    trait_id = "t_90"
    update_date = TRADING_DAYS[130]
    inputs = plan_inputs(
        traits=(pref_trait("curve_trade_expression", "express curve views as calendar spreads"),),
        drift_events=(
            drift_event(
                trait_id,
                update_date,
                DriftEventType.UPDATE,
                from_value="express curve views as calendar spreads",
                to_value="express curve views as spread ratios",
            ),
        ),
    )
    knobs = Config().plan
    carrier = Carrier(trait_id, "ti_001", TRADING_DAYS[0], CarrierSource.IDEA, None)
    pools = {trait_id: (carrier,)}
    signals = plan_quotas(inputs, pools, load_catalogue(), knobs, _rng(1))
    confirm = [s for s in _confirm(signals) if s.trait_id == trait_id]
    revealed = [s for s in confirm if s.mode == SignalMode.REVEALED]
    before = [s for s in revealed if s.window.last < update_date]
    after = [s for s in revealed if s.window.first >= update_date]
    assert before
    assert after
    for s in before:
        assert s.entry == StanceEntry.REVEALED
        assert s.needs_carrier is True
    for s in after:
        assert s.entry == StanceEntry.REVEALED_REACTION
        assert s.needs_carrier is False


def test_bias_entry_and_carrier_need() -> None:
    inputs = plan_inputs(traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),))
    knobs = Config().plan
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(1))
    for s in _confirm(signals):
        if s.mode in (SignalMode.REVEALED, SignalMode.CONTRADICTION):
            assert s.entry == StanceEntry.REVEALED
            assert s.needs_carrier is True
        else:
            assert s.entry == StanceEntry.STATED
            assert s.needs_carrier is False


# --- retracted / third-party ---------------------------------------------------------------


def _base_inputs():
    return plan_inputs(
        traits=(
            bias_trait("loss_aversion_lambda", trait_id="t_01"),
            bias_trait("disposition_ratio", trait_id="t_02", active=False),
            pref_trait("duration_expression", "steepeners over outright duration"),
        )
    )


def test_retracted_and_third_party_are_additional_and_match_formula() -> None:
    inputs = _base_inputs()
    default_knobs = Config().plan
    zero_knobs = default_knobs.model_copy(update={"retracted_share": 0.0, "third_party_share": 0.0})

    zero_signals = plan_quotas(inputs, {}, load_catalogue(), zero_knobs, _rng(5))
    default_signals = plan_quotas(inputs, {}, load_catalogue(), default_knobs, _rng(5))

    confirm_zero = _confirm(zero_signals)
    confirm_default = _confirm(default_signals)
    assert len(confirm_zero) == len(confirm_default)

    retracted = [s for s in default_signals if s.entry == StanceEntry.RETRACT]
    third_party = [s for s in default_signals if s.entry == StanceEntry.THIRD_PARTY]
    expected_r = round_half_up(default_knobs.retracted_share * len(confirm_default))
    expected_k = round_half_up(
        default_knobs.third_party_share * (len(confirm_default) + expected_r)
    )
    assert len(retracted) == expected_r
    assert len(third_party) == expected_k
    for s in retracted:
        assert s.mode == SignalMode.STATED
        assert s.valence == Valence.RETRACTED
        assert s.ownership == Ownership.SELF
        assert s.needs_carrier is False
        assert s.window == DateWindow(TRADING_DAYS[0], TRADING_DAYS[-1])

    zero_retracted = [s for s in zero_signals if s.entry == StanceEntry.RETRACT]
    zero_third_party = [s for s in zero_signals if s.entry == StanceEntry.THIRD_PARTY]
    assert zero_retracted == []
    assert zero_third_party == []


def test_third_party_preference_rows_carry_never_held_value() -> None:
    inputs = _base_inputs()
    # A high third-party share so k >= 2: with an inactive bias and a preference target
    # both present, k_pref = k - ceil(k / 2) is only non-zero once k reaches 2.
    knobs = Config().plan.model_copy(update={"third_party_share": 0.5})
    catalogue = load_catalogue()
    signals = plan_quotas(inputs, {}, catalogue, knobs, _rng(3))
    third_party = [s for s in signals if s.entry == StanceEntry.THIRD_PARTY]
    pref_rows = [s for s in third_party if s.trait_id == "t_90"]
    entry = next(e for e in catalogue.preferences if e.param == "duration_expression")
    assert pref_rows
    for s in pref_rows:
        assert s.third_party_value is not None
        assert s.third_party_value != "steepeners over outright duration"
        assert s.third_party_value in entry.values
        assert s.ownership in (Ownership.COLLEAGUE, Ownership.CLIENT)

    bias_rows = [s for s in third_party if s.trait_id != "t_90"]
    assert all(s.trait_id == "t_02" for s in bias_rows)
    assert all(s.third_party_value is None for s in bias_rows)


def test_third_party_never_held_value_excludes_drift_from_and_to() -> None:
    """A value the trait passed through via a drift event is not "never held", even when
    it differs from the trait's current value.
    """
    inputs = plan_inputs(
        traits=(
            bias_trait("loss_aversion_lambda", trait_id="t_01"),
            bias_trait("disposition_ratio", trait_id="t_02", active=False),
            pref_trait("duration_expression", "steepeners over outright duration"),
        ),
        drift_events=(
            drift_event(
                "t_90",
                TRADING_DAYS[50],
                DriftEventType.UPDATE,
                from_value="outright duration over curve trades",
                to_value="steepeners over outright duration",
            ),
        ),
    )
    knobs = Config().plan.model_copy(update={"third_party_share": 0.5})
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(3))
    pref_rows = [s for s in signals if s.entry == StanceEntry.THIRD_PARTY and s.trait_id == "t_90"]
    assert pref_rows
    for s in pref_rows:
        assert s.third_party_value == "butterflies over outright duration"


def test_no_inactive_bias_sends_all_third_party_to_preferences() -> None:
    inputs = plan_inputs(
        traits=(
            bias_trait("loss_aversion_lambda", trait_id="t_01"),
            pref_trait("duration_expression", "steepeners over outright duration"),
        )
    )
    knobs = Config().plan
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(4))
    third_party = [s for s in signals if s.entry == StanceEntry.THIRD_PARTY]
    assert third_party
    assert all(s.trait_id == "t_90" for s in third_party)


# --- drift ---------------------------------------------------------------------------------


def test_update_event_gives_every_segment_the_drift_minimum() -> None:
    knobs = Config().plan
    update_date = TRADING_DAYS[130]
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        drift_events=(
            drift_event("t_01", update_date, DriftEventType.UPDATE, from_value=0.7, to_value=1.2),
        ),
    )
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(7))
    confirm = [s for s in _confirm(signals) if s.trait_id == "t_01"]
    note = next(s for s in signals if s.entry == StanceEntry.DRIFT_UPDATE)
    assert note.drift_date == update_date
    assert note.window == DateWindow(update_date, TRADING_DAYS[-1])

    before = [s for s in confirm if s.window.last < update_date]
    after = [s for s in confirm if s.window.first >= update_date]
    assert len(before) >= knobs.drift_min_per_side
    # the note itself counts toward the after side
    assert len(after) + 1 >= knobs.drift_min_per_side


def test_dormant_revive_pair_gives_two_segments_excluding_dormant_window() -> None:
    knobs = Config().plan
    dormant_date = TRADING_DAYS[50]
    revive_date = TRADING_DAYS[100]
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        drift_events=(
            drift_event("t_01", dormant_date, DriftEventType.DORMANT),
            drift_event("t_01", revive_date, DriftEventType.REVIVE),
        ),
    )
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(9))
    confirm = [s for s in _confirm(signals) if s.trait_id == "t_01"]
    windows = sorted({s.window for s in confirm}, key=lambda w: w.first)
    assert len(windows) == 2
    before_window, after_window = windows
    for window in windows:
        assert not (dormant_date <= window.first < revive_date)
        assert not (dormant_date <= window.last < revive_date)
        assert window.first < dormant_date or window.first >= revive_date

    before = [s for s in confirm if s.window == before_window]
    after = [s for s in confirm if s.window == after_window]
    assert len(before) >= knobs.drift_min_per_side
    # the revive note itself counts toward the after side's minimum
    assert len(after) + 1 >= knobs.drift_min_per_side

    dormant_note = next(s for s in signals if s.entry == StanceEntry.DRIFT_DORMANT)
    revive_note = next(s for s in signals if s.entry == StanceEntry.DRIFT_REVIVE)
    assert dormant_note.drift_date == dormant_date
    assert revive_note.drift_date == revive_date


def test_one_note_per_drift_event_with_right_entry_and_date() -> None:
    update_date = TRADING_DAYS[20]
    dormant_date = TRADING_DAYS[60]
    revive_date = TRADING_DAYS[90]
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        drift_events=(
            drift_event("t_01", update_date, DriftEventType.UPDATE, from_value=0.7, to_value=1.0),
            drift_event("t_01", dormant_date, DriftEventType.DORMANT),
            drift_event("t_01", revive_date, DriftEventType.REVIVE),
        ),
    )
    knobs = Config().plan
    signals = plan_quotas(inputs, {}, load_catalogue(), knobs, _rng(11))
    notes = [
        s
        for s in signals
        if s.entry in _NOTE_ENTRIES - {StanceEntry.RETRACT, StanceEntry.THIRD_PARTY}
    ]
    assert len(notes) == 3
    by_entry = {s.entry: s for s in notes}
    assert by_entry[StanceEntry.DRIFT_UPDATE].drift_date == update_date
    assert by_entry[StanceEntry.DRIFT_DORMANT].drift_date == dormant_date
    assert by_entry[StanceEntry.DRIFT_REVIVE].drift_date == revive_date


# --- determinism -----------------------------------------------------------------------


def test_same_seed_gives_equal_lists() -> None:
    inputs = _base_inputs()
    knobs = Config().plan
    catalogue = load_catalogue()
    first = plan_quotas(inputs, {}, catalogue, knobs, _rng(42))
    second = plan_quotas(inputs, {}, catalogue, knobs, _rng(42))
    assert first == second
