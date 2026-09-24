"""Tests for the entry-side bias rules and the bias-param-to-module registry."""

import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import pearsonr

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.biases import BIAS_RULES, join_flags
from pm_traitbench.engine.biases import conviction as conviction_module
from pm_traitbench.engine.biases import extrapolation as extrapolation_module
from pm_traitbench.engine.biases import herding as herding_module
from pm_traitbench.engine.biases import overconfidence as overconfidence_module
from pm_traitbench.engine.biases.conviction import size_rank
from pm_traitbench.engine.biases.extrapolation import blend, entered_after_run
from pm_traitbench.engine.biases.herding import HerdingDecision, decide
from pm_traitbench.engine.biases.overconfidence import size_factor
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import Side, StreetView
from pm_traitbench.rng import stream


def _params(values: dict[str, float], active: set[str]) -> EffectiveParams:
    """An EffectiveParams over all eight BIAS_PARAMS, with only the given overrides."""
    return EffectiveParams(
        values={param: values.get(param, 0.5) for param in BIAS_PARAMS},
        active={param: param in active for param in BIAS_PARAMS},
    )


# --- registry ---------------------------------------------------------------


def test_bias_rules_registers_the_four_entry_side_modules() -> None:
    assert BIAS_RULES["extrapolation_theta"] is extrapolation_module
    assert BIAS_RULES["herding_weight"] is herding_module
    assert BIAS_RULES["overconfidence_coverage"] is overconfidence_module
    assert BIAS_RULES["conviction_size_miscalibration"] is conviction_module


def test_own_signal_imports_alone_without_a_circular_import() -> None:
    """own_signal imports biases.overconfidence, so it must load with no other module warm."""
    result = subprocess.run(
        [sys.executable, "-c", "import pm_traitbench.engine.own_signal"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# --- extrapolation.blend -----------------------------------------------------


def test_blend_at_theta_zero_is_pure_thesis() -> None:
    params = _params({"extrapolation_theta": 0.0}, set())
    assert blend(3.0, -7.0, params) == pytest.approx(3.0)


def test_blend_at_theta_one_is_pure_trailing() -> None:
    params = _params({"extrapolation_theta": 1.0}, set())
    assert blend(3.0, -7.0, params) == pytest.approx(-7.0)


# --- extrapolation.entered_after_run -----------------------------------------


@pytest.mark.parametrize(
    ("trailing_move", "sd_h", "side_sign", "bullish_sign", "expected"),
    [
        (5.0, 2.0, 1, 1, True),
        (1.0, 2.0, 1, 1, False),
        (-5.0, 2.0, -1, 1, True),
        (5.0, 2.0, -1, 1, False),
        (-5.0, 2.0, 1, -1, True),
        (5.0, 2.0, 1, -1, False),
    ],
)
def test_entered_after_run_sign_cases(
    trailing_move: float, sd_h: float, side_sign: int, bullish_sign: int, expected: bool
) -> None:
    assert entered_after_run(trailing_move, sd_h, side_sign, bullish_sign) is expected


# --- herding.decide -----------------------------------------------------------


def test_herding_no_conflict_when_street_is_none() -> None:
    params = _params({"herding_weight": 1.0}, {"herding_weight"})
    rng = stream(1, "herding-none")
    decision = decide(Side.BUY, None, params, rng)
    assert decision == HerdingDecision(False, None, Side.BUY, None)


def test_herding_no_conflict_when_street_is_neutral() -> None:
    params = _params({"herding_weight": 1.0}, {"herding_weight"})
    rng = stream(1, "herding-neutral")
    decision = decide(Side.SELL, StreetView.NEUTRAL, params, rng)
    assert decision == HerdingDecision(False, None, Side.SELL, None)


def test_herding_no_conflict_when_street_agrees() -> None:
    params = _params({"herding_weight": 1.0}, {"herding_weight"})
    rng = stream(1, "herding-agree")
    decision = decide(Side.BUY, StreetView.OVERWEIGHT, params, rng)
    assert decision == HerdingDecision(False, None, Side.BUY, None)


def test_herding_w_zero_never_follows_on_conflict() -> None:
    params = _params({"herding_weight": 0.0}, {"herding_weight"})
    rng = stream(1, "herding-w0")
    for _ in range(200):
        decision = decide(Side.BUY, StreetView.UNDERWEIGHT, params, rng)
        assert decision.conflict is True
        assert decision.followed_street is False
        assert decision.side == Side.BUY
        assert decision.flag is None


def test_herding_w_one_always_follows_on_conflict() -> None:
    params = _params({"herding_weight": 1.0}, {"herding_weight"})
    rng = stream(1, "herding-w1")
    for _ in range(200):
        decision = decide(Side.BUY, StreetView.UNDERWEIGHT, params, rng)
        assert decision.conflict is True
        assert decision.followed_street is True
        assert decision.side == Side.SELL
        assert decision.flag == "herding:followed_street"


def test_herding_flag_only_when_active() -> None:
    params = _params({"herding_weight": 1.0}, set())
    rng = stream(1, "herding-inactive")
    decision = decide(Side.BUY, StreetView.UNDERWEIGHT, params, rng)
    assert decision.followed_street is True
    assert decision.flag is None


def test_herding_follow_share_matches_w_over_many_draws() -> None:
    # False-alarm rate: tolerance 0.05 at SE 0.011 is about 4.5 SE, a two-sided
    # chance of a spurious failure at the true rate of about 1 in 150,000.
    params = _params({"herding_weight": 0.58}, {"herding_weight"})
    rng = stream(1, "herding-share")
    followed = sum(
        decide(Side.BUY, StreetView.UNDERWEIGHT, params, rng).followed_street for _ in range(2000)
    )
    share = followed / 2000
    assert share == pytest.approx(0.58, abs=0.05)


# --- overconfidence.size_factor -----------------------------------------------


def test_overconfidence_factor_is_one_at_reference_coverage() -> None:
    params = _params({"overconfidence_coverage": 0.8}, {"overconfidence_coverage"})
    factor, flag = size_factor(params)
    assert factor == pytest.approx(1.0)
    assert flag is None


def test_overconfidence_factor_exceeds_one_below_reference_coverage() -> None:
    params = _params({"overconfidence_coverage": 0.4}, {"overconfidence_coverage"})
    factor, flag = size_factor(params)
    assert factor > 1.0
    assert flag == "overconfidence:oversized"


def test_overconfidence_flag_only_when_active() -> None:
    params = _params({"overconfidence_coverage": 0.4}, set())
    factor, flag = size_factor(params)
    assert factor > 1.0
    assert flag is None


# --- conviction.size_rank -----------------------------------------------------


def test_conviction_m_zero_returns_the_conviction() -> None:
    params = _params({"conviction_size_miscalibration": 0.0}, {"conviction_size_miscalibration"})
    rng = stream(1, "conviction-m0")
    for conviction in range(1, 6):
        rank, flag = size_rank(conviction, params, rng)
        assert rank == conviction
        assert flag is None


def test_conviction_flag_only_when_active_and_different() -> None:
    active_params = _params(
        {"conviction_size_miscalibration": 1.0}, {"conviction_size_miscalibration"}
    )
    inactive_params = _params({"conviction_size_miscalibration": 1.0}, set())
    rng_active = stream(1, "conviction-active")
    rng_inactive = stream(1, "conviction-inactive")

    saw_flag = False
    for _ in range(200):
        rank, flag = size_rank(3, active_params, rng_active)
        if rank != 3:
            assert flag == "conviction:mis_sized"
            saw_flag = True
        else:
            assert flag is None
    assert saw_flag

    for _ in range(200):
        rank, flag = size_rank(3, inactive_params, rng_inactive)
        assert flag is None


def test_conviction_rank_uncorrelated_with_conviction_at_m_one() -> None:
    # m=1 draws the rank independently of conviction, so the sample correlation
    # over 2,000 draws should sit near zero; 0.15 gives ample margin over the
    # ~0.022 standard error implied by the sample size, with no seed hunting.
    params = _params({"conviction_size_miscalibration": 1.0}, {"conviction_size_miscalibration"})
    rng = stream(1, "conviction-corr")
    convictions = rng.integers(1, 6, size=2000)
    ranks = np.array([size_rank(int(c), params, rng)[0] for c in convictions])
    corr, _ = pearsonr(convictions, ranks)
    assert abs(corr) < 0.15


# --- join_flags ----------------------------------------------------------------


def test_join_flags_orders_by_bias_flag_order() -> None:
    flags = ["conviction:mis_sized", "herding:followed_street", "overconfidence:oversized"]
    expected = "herding:followed_street;overconfidence:oversized;conviction:mis_sized"
    assert join_flags(flags) == expected


def test_join_flags_drops_none() -> None:
    flags = [None, "herding:followed_street", None]
    assert join_flags(flags) == "herding:followed_street"


def test_join_flags_returns_none_when_empty() -> None:
    assert join_flags([]) is None
    assert join_flags([None, None]) is None


def test_join_flags_keeps_input_order_for_unknown_prefixes() -> None:
    flags = ["zzz:unknown", "herding:followed_street", "aaa:other"]
    assert join_flags(flags) == "herding:followed_street;zzz:unknown;aaa:other"
