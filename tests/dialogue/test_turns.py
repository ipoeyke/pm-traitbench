import datetime

import numpy as np
import pytest

from pm_traitbench.config import TurnRanges
from pm_traitbench.dialogue.turns import OPENING_BY_KIND, Opening, plan_turns
from pm_traitbench.enums import SessionKind, SignalMode, StanceEntry
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import Skeleton, Stance
from tests.gates.conftest import ledger_row

PM_ID = "pm_001"
SESSION_DATE = datetime.date(2026, 3, 2)
SESSION_ID = "s_pm001_2026-03-02_a"


def _stance(**overrides: object) -> Stance:
    fields = {
        "signal_id": "sg_001",
        "trait_id": "t_01",
        "mode": SignalMode.STATED,
        "entry": StanceEntry.STATED,
        "stance": "I focus on quality names.",
    }
    fields.update(overrides)
    return Stance(**fields)


def _skeleton(**overrides: object) -> Skeleton:
    fields = {
        "session_id": SESSION_ID,
        "pm_id": PM_ID,
        "date": SESSION_DATE,
        "kind": SessionKind.DECISION,
        "trade_idea_ids": (),
        "stances": (),
        "advisor_violation": None,
        "forbidden_trait_ids": (),
        "forbidden_pref_params": (),
    }
    fields.update(overrides)
    return Skeleton(**fields)


def _reaction_stance(**overrides: object) -> Stance:
    fields = {
        "signal_id": "sg_002",
        "trait_id": "t_02",
        "mode": SignalMode.REVEALED,
        "entry": StanceEntry.REVEALED_REACTION,
        "stance": "That's not what I asked the copilot to do.",
    }
    fields.update(overrides)
    return Stance(**fields)


def test_turn_count_is_drawn_from_the_kind_range() -> None:
    ranges = TurnRanges()
    for kind in SessionKind:
        allowed = ranges.for_kind(kind)
        skeleton = _skeleton(kind=kind, stances=(), trade_idea_ids=(), advisor_violation=None)
        for seed in range(50):
            rng = np.random.default_rng(seed)
            plan = plan_turns(skeleton, (), ranges, rng)
            assert plan.n_turns in allowed


def test_turn_count_is_raised_to_fit_every_stance() -> None:
    ranges = TurnRanges(decision=(2,))
    skeleton = _skeleton(
        kind=SessionKind.DECISION,
        stances=(_stance(), _stance(signal_id="sg_002", trait_id="t_02")),
    )
    rng = np.random.default_rng(0)
    plan = plan_turns(skeleton, (), ranges, rng)
    assert plan.n_turns == 4
    assert len(plan.pm_directives) == 2


def test_reaction_sits_on_pm_turn_two_or_later_with_violation_just_before() -> None:
    ranges = TurnRanges()
    skeleton = _skeleton(
        kind=SessionKind.DECISION,
        stances=(_stance(), _reaction_stance()),
        advisor_violation="The advisor recommended a size above the PM's cap.",
    )
    for seed in range(50):
        rng = np.random.default_rng(seed)
        plan = plan_turns(skeleton, (), ranges, rng)
        reaction_index = next(
            i
            for i, directive in enumerate(plan.pm_directives)
            if directive.stance is not None
            and directive.stance.entry == StanceEntry.REVEALED_REACTION
        )
        assert reaction_index >= 1
        assert plan.violation_advisor_index == reaction_index - 1


def test_each_stance_gets_its_own_pm_turn() -> None:
    ranges = TurnRanges()
    stances = (
        _stance(),
        _stance(signal_id="sg_002", trait_id="t_02"),
        _reaction_stance(signal_id="sg_003", trait_id="t_03"),
    )
    skeleton = _skeleton(
        kind=SessionKind.DECISION,
        stances=stances,
        advisor_violation="The advisor recommended a size above the PM's cap.",
    )
    rng = np.random.default_rng(3)
    plan = plan_turns(skeleton, (), ranges, rng)
    placed_indices = [
        i for i, directive in enumerate(plan.pm_directives) if directive.stance is not None
    ]
    placed_stances = [plan.pm_directives[i].stance for i in placed_indices]
    assert len(placed_indices) == len(stances)
    assert len(set(placed_indices)) == len(stances)
    assert set(placed_stances) == set(stances)


def test_day_trades_and_opening_go_on_pm_turn_zero_only() -> None:
    ranges = TurnRanges()
    skeleton = _skeleton(kind=SessionKind.CHECK_IN, stances=(), trade_idea_ids=())
    trades = (ledger_row(pm_id=PM_ID, date=SESSION_DATE),)
    rng = np.random.default_rng(7)
    plan = plan_turns(skeleton, trades, ranges, rng)

    assert plan.pm_directives[0].trades == trades
    assert plan.pm_directives[0].opening == OPENING_BY_KIND[SessionKind.CHECK_IN]
    for directive in plan.pm_directives[1:]:
        assert directive.trades == ()
        assert directive.opening is None


def test_opening_by_kind_covers_every_kind() -> None:
    assert OPENING_BY_KIND == {
        SessionKind.DECISION: Opening.SESSION_IDEAS,
        SessionKind.CHECK_IN: Opening.OPEN_POSITIONS,
        SessionKind.SILENCE: Opening.MARKET_QUESTION,
    }


def test_same_seed_gives_the_same_plan() -> None:
    ranges = TurnRanges()
    skeleton = _skeleton(
        kind=SessionKind.DECISION,
        stances=(_stance(), _reaction_stance()),
        advisor_violation="The advisor recommended a size above the PM's cap.",
    )
    trades = (ledger_row(pm_id=PM_ID, date=SESSION_DATE),)

    plan_a = plan_turns(skeleton, trades, ranges, np.random.default_rng(42))
    plan_b = plan_turns(skeleton, trades, ranges, np.random.default_rng(42))

    assert plan_a == plan_b


def test_violation_without_a_reaction_raises() -> None:
    ranges = TurnRanges()
    # Skeleton's own validator forbids this combination, so bypass it to exercise
    # plan_turns' defensive check on an assumed invariant.
    skeleton = Skeleton.model_construct(
        session_id=SESSION_ID,
        pm_id=PM_ID,
        date=SESSION_DATE,
        kind=SessionKind.DECISION,
        trade_idea_ids=(),
        stances=(_stance(),),
        advisor_violation="The advisor recommended a size above the PM's cap.",
        forbidden_trait_ids=(),
        forbidden_pref_params=(),
    )
    rng = np.random.default_rng(0)
    with pytest.raises(DialogueError, match=SESSION_ID):
        plan_turns(skeleton, (), ranges, rng)


def test_reaction_without_a_violation_raises() -> None:
    ranges = TurnRanges()
    skeleton = Skeleton.model_construct(
        session_id=SESSION_ID,
        pm_id=PM_ID,
        date=SESSION_DATE,
        kind=SessionKind.DECISION,
        trade_idea_ids=(),
        stances=(_reaction_stance(),),
        advisor_violation=None,
        forbidden_trait_ids=(),
        forbidden_pref_params=(),
    )
    rng = np.random.default_rng(0)
    with pytest.raises(DialogueError, match=SESSION_ID):
        plan_turns(skeleton, (), ranges, rng)


def test_two_reactions_raise() -> None:
    ranges = TurnRanges()
    skeleton = _skeleton(
        kind=SessionKind.DECISION,
        stances=(
            _reaction_stance(),
            _reaction_stance(signal_id="sg_003", trait_id="t_03"),
        ),
        advisor_violation="The advisor recommended a size above the PM's cap.",
    )
    rng = np.random.default_rng(0)
    with pytest.raises(DialogueError, match=SESSION_ID):
        plan_turns(skeleton, (), ranges, rng)
