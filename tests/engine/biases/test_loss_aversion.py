"""Tests for the loss-aversion cut/add hazard construction."""

import math

import pytest

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.biases.loss_aversion import LossSideChoice, choose
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import PositionAction
from pm_traitbench.rng import stream


def _params(lam: float, *, active: bool = True) -> EffectiveParams:
    """An EffectiveParams over all eight BIAS_PARAMS, lambda set and the rest neutral."""
    values = {param: 0.5 for param in BIAS_PARAMS}
    values["loss_aversion_lambda"] = lam
    return EffectiveParams(
        values=values,
        active={param: (param == "loss_aversion_lambda" and active) for param in BIAS_PARAMS},
    )


def _share(action: PositionAction, choices: list[LossSideChoice]) -> float:
    return sum(1 for c in choices if c.action == action) / len(choices)


def _se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def test_lambda_one_never_adds_and_cuts_at_the_base_hazard() -> None:
    n = 20_000
    params = _params(1.0)
    rng = stream(1, "loss-aversion-lam-one")
    choices = [choose(params, rng, add_allowed=True) for _ in range(n)]
    assert _share(PositionAction.ADD, choices) == 0.0
    assert _share(PositionAction.CUT, choices) == pytest.approx(0.05, abs=4 * _se(0.05, n))


def test_lambda_two_add_and_cut_shares_match_the_formula() -> None:
    n = 20_000
    params = _params(2.0)
    rng = stream(1, "loss-aversion-lam-two")
    choices = [choose(params, rng, add_allowed=True) for _ in range(n)]
    assert _share(PositionAction.ADD, choices) == pytest.approx(0.1, abs=4 * _se(0.1, n))
    assert _share(PositionAction.CUT, choices) == pytest.approx(0.025, abs=4 * _se(0.025, n))


def test_add_not_allowed_never_adds() -> None:
    n = 20_000
    params = _params(2.0)
    rng = stream(1, "loss-aversion-no-add")
    choices = [choose(params, rng, add_allowed=False) for _ in range(n)]
    assert _share(PositionAction.ADD, choices) == 0.0


def test_lambda_twenty_caps_the_add_share_at_one_half() -> None:
    n = 20_000
    params = _params(20.0)
    rng = stream(1, "loss-aversion-lam-cap")
    choices = [choose(params, rng, add_allowed=True) for _ in range(n)]
    assert _share(PositionAction.ADD, choices) == pytest.approx(0.5, abs=4 * _se(0.5, n))


def test_flags_match_action_when_lambda_active() -> None:
    params = _params(4.0, active=True)
    rng = stream(1, "loss-aversion-flags")
    seen_flags = set()
    for _ in range(500):
        choice = choose(params, rng, add_allowed=True)
        if choice.action == PositionAction.ADD:
            assert choice.flag == "loss_aversion:add"
        elif choice.action == PositionAction.HOLD:
            assert choice.flag == "loss_aversion:hold"
        else:
            assert choice.flag is None
        seen_flags.add(choice.flag)
    assert "loss_aversion:add" in seen_flags
    assert "loss_aversion:hold" in seen_flags


def test_no_flag_when_lambda_inactive() -> None:
    params = _params(4.0, active=False)
    rng = stream(1, "loss-aversion-inactive")
    for _ in range(500):
        choice = choose(params, rng, add_allowed=True)
        assert choice.flag is None
