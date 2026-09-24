"""Tests for idea generation: `attempt_entry` and `entries_for_day`."""

from collections.abc import Callable

import numpy as np
import pytest

from pm_traitbench.config import Config, EngineConfig
from pm_traitbench.engine import ideas as ideas_module
from pm_traitbench.engine.adapters.equities import EquitiesAdapter
from pm_traitbench.engine.biases import herding
from pm_traitbench.engine.biases.herding import HerdingDecision
from pm_traitbench.engine.constants import NO_ENTRY_LAST_SESSIONS
from pm_traitbench.engine.ideas import attempt_entry, entries_for_day
from pm_traitbench.engine.own_signal import SignalDraw
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position, idea_id, rule_id
from pm_traitbench.enums import AssetClass, Expression, Kind, Op, Side
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import Leg, Trait

_PM_ID = "pm_001"
_SUB_STYLE = "equity_long_short"
_T = 10


def _rng_for(config: Config, pm_id: str) -> Callable[..., np.random.Generator]:
    def rng_for(purpose: str, *keys) -> np.random.Generator:
        return stream(config.seed.root, "engine", pm_id, purpose, *keys)

    return rng_for


def _params(
    *, theta: float = 0.0, coverage: float = 0.8, herding_weight: float = 0.0, mis: float = 0.0
) -> EffectiveParams:
    values = {
        "extrapolation_theta": theta,
        "overconfidence_coverage": coverage,
        "herding_weight": herding_weight,
        "conviction_size_miscalibration": mis,
    }
    return EffectiveParams(values=values, active={k: False for k in values})


def _strong_signal(own_signal: float = 2.0) -> SignalDraw:
    return SignalDraw(
        own_signal=own_signal,
        sd_h=1.0,
        thesis_move=own_signal,
        forecast=own_signal * 0.8,
        interval_lo=-1.0,
        interval_hi=3.0,
        conviction=5,
    )


def _force_no_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch herding.decide so the resolved side always matches the PM's own signal."""

    def _decide(own_side, street, params, rng) -> HerdingDecision:
        return HerdingDecision(conflict=False, followed_street=None, side=own_side, flag=None)

    monkeypatch.setattr(herding, "decide", _decide)


def _dummy_position(instrument_id: str, n: int) -> Position:
    leg = Leg(instrument_id=instrument_id, tenor=None, side=Side.BUY, weight=1.0)
    series = Series(legs=(LegRef(instrument_id, None, 1.0),), bullish_sign=1, unit="pct")
    return Position(
        trade_idea_id=idea_id(n),
        expression=Expression.OUTRIGHT,
        instrument_id=instrument_id,
        legs=(leg,),
        series=series,
        side=Side.BUY,
        entry_t=0,
        entry_level=0.0,
        target_level=1.0,
        stop_level=-1.0,
        sd_h_at_entry=1.0,
        forecast=0.0,
        size_pct_book=1.0,
        original_size_pct_book=1.0,
        conviction=1,
        size_rank=1,
        triggers_fired=0,
        consumed_rule_ids=frozenset(),
        run_counters=(),
        size_changed_t=0,
    )


@pytest.fixture
def equities_setup(fixture_view, neutral_pm, catalogue):
    persona, traits, rules = neutral_pm(AssetClass.EQUITIES, _SUB_STYLE)
    adapter = EquitiesAdapter(sub_style=_SUB_STYLE, horizon_days=20)
    universe = adapter.universe(fixture_view.instruments, rules)
    config = Config()
    state = PmState(pm_id=_PM_ID, positions=(), next_idea=1, next_rule=1)
    return {
        "view": fixture_view,
        "persona": persona,
        "traits": traits,
        "rules": rules,
        "adapter": adapter,
        "universe": universe,
        "config": config,
        "catalogue": catalogue,
        "state": state,
    }


def test_no_entry_in_last_sessions(equities_setup) -> None:
    setup = equities_setup
    t = setup["view"].n_days - NO_ENTRY_LAST_SESSIONS
    new_state, new_idea = attempt_entry(
        setup["state"],
        t,
        setup["view"],
        setup["adapter"],
        _params(),
        setup["persona"],
        setup["rules"],
        setup["traits"],
        setup["universe"],
        setup["config"],
        setup["catalogue"],
        _rng_for(setup["config"], _PM_ID),
        attempt=0,
    )
    assert new_idea is None
    assert new_state == setup["state"]


def test_no_entry_below_threshold(equities_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    setup = equities_setup
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(0.5))
    new_state, new_idea = attempt_entry(
        setup["state"],
        _T,
        setup["view"],
        setup["adapter"],
        _params(),
        setup["persona"],
        setup["rules"],
        setup["traits"],
        setup["universe"],
        setup["config"],
        setup["catalogue"],
        _rng_for(setup["config"], _PM_ID),
        attempt=0,
    )
    assert new_idea is None
    assert new_state == setup["state"]


def test_strong_signal_produces_consistent_stop_and_target(
    equities_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = equities_setup
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(2.0))
    _force_no_conflict(monkeypatch)
    new_state, new_idea = attempt_entry(
        setup["state"],
        _T,
        setup["view"],
        setup["adapter"],
        _params(),
        setup["persona"],
        setup["rules"],
        setup["traits"],
        setup["universe"],
        setup["config"],
        setup["catalogue"],
        _rng_for(setup["config"], _PM_ID),
        attempt=0,
    )
    assert new_idea is not None
    idea = new_idea.idea
    assert idea.side == Side.BUY
    assert idea.stop_level < idea.entry_level
    assert idea.target_level > idea.entry_level
    rr_lo, rr_hi = setup["config"].engine.rr_range
    rr = (idea.target_level - idea.entry_level) / (idea.entry_level - idea.stop_level)
    assert rr_lo <= rr <= rr_hi
    assert (idea.target_level - idea.entry_level) == pytest.approx(
        rr * (idea.entry_level - idea.stop_level), abs=1e-9
    )


def test_cap_never_exceeded_with_a_large_overconfidence_factor(
    equities_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = equities_setup
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(2.0))
    _force_no_conflict(monkeypatch)
    params = _params(coverage=0.3)  # low stated coverage inflates the overconfidence factor
    new_state, new_idea = attempt_entry(
        setup["state"],
        _T,
        setup["view"],
        setup["adapter"],
        params,
        setup["persona"],
        setup["rules"],
        setup["traits"],
        setup["universe"],
        setup["config"],
        setup["catalogue"],
        _rng_for(setup["config"], _PM_ID),
        attempt=0,
    )
    assert new_idea is not None
    cap_rule = next(r for r in setup["rules"] if r.param == "max_risk_pct")
    assert new_idea.position.size_pct_book <= float(cap_rule.level) + 1e-9


def test_signposts_count_ids_and_event_kind(
    equities_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = equities_setup
    # Hold every equity but EQ-0002 (no calendar events) so it is the only candidate.
    held = [
        _dummy_position("EQ-0001", 1),
        _dummy_position("EQ-0003", 2),
        _dummy_position("EQ-0004", 3),
    ]
    state = PmState(pm_id=_PM_ID, positions=tuple(held), next_idea=4, next_rule=10)
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(2.0))
    _force_no_conflict(monkeypatch)
    new_state, new_idea = attempt_entry(
        state,
        _T,
        setup["view"],
        setup["adapter"],
        _params(),
        setup["persona"],
        setup["rules"],
        setup["traits"],
        setup["universe"],
        setup["config"],
        setup["catalogue"],
        _rng_for(setup["config"], _PM_ID),
        attempt=0,
    )
    assert new_idea is not None
    assert new_idea.idea.instrument_id == "EQ-0002"
    n_signposts = len(new_idea.rules) - 2
    assert n_signposts in (2, 3)
    expected_ids = [rule_id(10 + i) for i in range(len(new_idea.rules))]
    assert [r.rule_id for r in new_idea.rules] == expected_ids
    assert new_state.next_rule == 10 + len(new_idea.rules)
    signposts = new_idea.rules[2:]
    assert all(r.field != "event" for r in signposts)


def test_event_signpost_fires_when_candidate_has_event_types(
    equities_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = equities_setup
    # Hold every equity but EQ-0001, which has earnings events, so it is the only candidate.
    held = [
        _dummy_position("EQ-0002", 1),
        _dummy_position("EQ-0003", 2),
        _dummy_position("EQ-0004", 3),
    ]
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(2.0))
    _force_no_conflict(monkeypatch)
    found_event_row = None
    for attempt in range(20):
        state = PmState(pm_id=_PM_ID, positions=tuple(held), next_idea=4, next_rule=1)
        _, new_idea = attempt_entry(
            state,
            _T,
            setup["view"],
            setup["adapter"],
            _params(),
            setup["persona"],
            setup["rules"],
            setup["traits"],
            setup["universe"],
            setup["config"],
            setup["catalogue"],
            _rng_for(setup["config"], _PM_ID),
            attempt=attempt,
        )
        assert new_idea is not None
        assert new_idea.idea.instrument_id == "EQ-0001"
        event_rows = [r for r in new_idea.rules[2:] if r.field == "event"]
        if event_rows:
            found_event_row = event_rows[0]
            break
    assert found_event_row is not None
    assert found_event_row.op == Op.EQ
    assert found_event_row.level == "earnings"
    assert found_event_row.unit is None
    assert found_event_row.window == 1
    assert "earnings" in found_event_row.text


def test_preferred_form_pair_with_weight_one_always_draws_pair(
    equities_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = equities_setup
    # Hold EQ-0004 (the lone sector_02 equity) so every remaining candidate has a pair partner.
    state = PmState(
        pm_id=_PM_ID, positions=(_dummy_position("EQ-0004", 1),), next_idea=2, next_rule=1
    )
    traits = [
        *setup["traits"],
        Trait(
            pm_id=_PM_ID,
            trait_id="t_99",
            kind=Kind.PREFERENCE,
            param="pair_vs_outright",
            value="express the view as a pair trade",
            active=True,
            mult_range=None,
            mult_risk_off=None,
            mult_risk_on=None,
        ),
    ]
    config = Config(engine=EngineConfig(preferred_form_weight=1.0))
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(2.0))
    _force_no_conflict(monkeypatch)
    for attempt in range(5):
        _, new_idea = attempt_entry(
            state,
            _T,
            setup["view"],
            setup["adapter"],
            _params(),
            setup["persona"],
            setup["rules"],
            traits,
            setup["universe"],
            config,
            setup["catalogue"],
            _rng_for(config, _PM_ID),
            attempt=attempt,
        )
        assert new_idea is not None
        assert new_idea.idea.expression == Expression.PAIR
        assert new_idea.ledger_rows[0].side != new_idea.ledger_rows[1].side


def test_max_positions_rule_stops_entries(equities_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    setup = equities_setup
    rules = [
        r.model_copy(update={"level": 0.0}) if r.param == "max_positions" else r
        for r in setup["rules"]
    ]
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(2.0))
    _force_no_conflict(monkeypatch)
    new_state, new_idea = attempt_entry(
        setup["state"],
        _T,
        setup["view"],
        setup["adapter"],
        _params(),
        setup["persona"],
        rules,
        setup["traits"],
        setup["universe"],
        setup["config"],
        setup["catalogue"],
        _rng_for(setup["config"], _PM_ID),
        attempt=0,
    )
    assert new_idea is None
    assert new_state == setup["state"]


def test_returned_state_has_new_position_and_input_state_is_unchanged(
    equities_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = equities_setup
    original_state = setup["state"]
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(2.0))
    _force_no_conflict(monkeypatch)
    new_state, new_idea = attempt_entry(
        original_state,
        _T,
        setup["view"],
        setup["adapter"],
        _params(),
        setup["persona"],
        setup["rules"],
        setup["traits"],
        setup["universe"],
        setup["config"],
        setup["catalogue"],
        _rng_for(setup["config"], _PM_ID),
        attempt=0,
    )
    assert new_idea is not None
    assert original_state.positions == ()
    assert original_state.next_idea == 1
    assert new_state.n_positions == 1
    assert new_state.position(idea_id(1)) == new_idea.position
    assert new_state.next_idea == 2


def test_entries_for_day_stops_in_last_sessions_without_attempting(equities_setup) -> None:
    setup = equities_setup
    t = setup["view"].n_days - 1
    new_state, new_ideas = entries_for_day(
        setup["state"],
        t,
        setup["view"],
        setup["adapter"],
        _params(),
        setup["persona"],
        setup["rules"],
        setup["traits"],
        setup["universe"],
        setup["config"],
        setup["catalogue"],
        _rng_for(setup["config"], _PM_ID),
    )
    assert new_ideas == []
    assert new_state == setup["state"]


def test_entries_for_day_collects_multiple_attempts(
    equities_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = equities_setup
    monkeypatch.setattr(ideas_module, "draw_signal", lambda *a, **kw: _strong_signal(2.0))
    _force_no_conflict(monkeypatch)
    config = Config(engine=EngineConfig(arrival_rate=20.0))
    new_state, new_ideas = entries_for_day(
        setup["state"],
        _T,
        setup["view"],
        setup["adapter"],
        _params(),
        setup["persona"],
        setup["rules"],
        setup["traits"],
        setup["universe"],
        config,
        setup["catalogue"],
        _rng_for(config, _PM_ID),
    )
    assert len(new_ideas) == new_state.n_positions
    assert new_state.next_idea == 1 + len(new_ideas)
