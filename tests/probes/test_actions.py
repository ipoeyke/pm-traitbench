import numpy as np
import pytest

from pm_traitbench.catalogues.loader import PROBE_ACTIONS
from pm_traitbench.config import Config
from pm_traitbench.enums import McqAction, OptionSource
from pm_traitbench.probes.actions import (
    NEUTRAL_FACTS,
    PmFacts,
    action_for,
    assemble_action_options,
    assemble_value_options,
    horizons,
    loss_side_outcomes,
)

CONFIG = Config()
HORIZONS = horizons(CONFIG)


def outcome(param: str, value: float, facts: PmFacts = NEUTRAL_FACTS) -> McqAction:
    return action_for(param, value, facts, HORIZONS, CONFIG)


def test_default_horizons():
    assert horizons(Config()) == {"loss_aversion_lambda": 34, "disposition_ratio": 14}


def test_loss_aversion_typical_at_the_active_floor():
    assert outcome("loss_aversion_lambda", 1.5) == McqAction.ADD
    assert outcome("loss_aversion_lambda", 2.0) == McqAction.ADD
    assert outcome("loss_aversion_lambda", 1.1) == McqAction.CUT


def test_loss_side_outcomes_at_the_neutral_median():
    add, hold, cut = loss_side_outcomes(1.1, NEUTRAL_FACTS, HORIZONS["loss_aversion_lambda"])
    assert add + hold + cut == pytest.approx(1.0)
    assert (add, hold, cut) == pytest.approx((0.15, 0.14, 0.70), abs=0.01)


def test_no_add_rule_scales_the_add_hazard():
    low = PmFacts(no_add_rule=True, lambda_active=True, exit_deficiency=0.06)
    high = PmFacts(no_add_rule=True, lambda_active=True, exit_deficiency=1.0)
    add, hold, cut = loss_side_outcomes(2.0, low, HORIZONS["loss_aversion_lambda"])
    assert (add, hold, cut) == pytest.approx((0.13, 0.34, 0.53), abs=0.01)
    assert outcome("loss_aversion_lambda", 2.0, low) == McqAction.CUT
    assert outcome("loss_aversion_lambda", 2.0, high) == McqAction.ADD


def test_disposition_reads_typical_at_the_floor():
    assert outcome("disposition_ratio", 1.2) == McqAction.SELL_NOW
    assert outcome("disposition_ratio", 1.0) == McqAction.HOLD_TO_TARGET


def test_hazard_param_without_a_horizon_raises():
    with pytest.raises(ValueError):
        action_for("disposition_ratio", 1.2, NEUTRAL_FACTS, {"disposition_ratio": None}, CONFIG)


def test_exit_deficiency_typical_depends_on_lambda():
    active = PmFacts(no_add_rule=False, lambda_active=True, exit_deficiency=0.6)
    assert outcome("exit_deficiency", 0.6, active) == McqAction.ADD
    assert outcome("exit_deficiency", 0.6) == McqAction.LEAVE_ON
    assert outcome("exit_deficiency", 0.3) == McqAction.EXIT_PER_STOP


def test_extrapolation_threshold_at_theta_half():
    assert outcome("extrapolation_theta", 0.5) == McqAction.CHASE_RUN
    assert outcome("extrapolation_theta", 0.49) == McqAction.STAND_ASIDE
    assert outcome("extrapolation_theta", 0.0) == McqAction.SELL_ON_THESIS


def test_overconfidence_buckets():
    assert outcome("overconfidence_coverage", 0.8) == McqAction.SIZE_STANDARD
    assert outcome("overconfidence_coverage", 0.55) == McqAction.SIZE_ONE_AND_HALF
    assert outcome("overconfidence_coverage", 0.4) == McqAction.SIZE_DOUBLE


def test_conviction_uses_the_redraw_rate():
    assert outcome("conviction_size_miscalibration", 0.6) == McqAction.SIZE_TO_RATING
    assert outcome("conviction_size_miscalibration", 0.7) == McqAction.SIZE_OFF_RATING


def test_action_for_returns_only_listed_outcomes():
    grid = [i / 20 for i in range(1, 20)]
    lam = [1.0, 1.1, 1.5, 2.0, 3.0]
    for param, listed in PROBE_ACTIONS.items():
        values = lam if param in ("loss_aversion_lambda", "disposition_ratio") else grid
        facts = [NEUTRAL_FACTS, PmFacts(True, True, 0.5)]
        assert {outcome(param, v, f) for v in values for f in facts} <= set(listed)


def test_typical_probability_rejects_unsupported_params():
    from pm_traitbench.probes.actions import typical_probability

    with pytest.raises(ValueError):
        typical_probability("extrapolation_theta", 0.5, NEUTRAL_FACTS, None, CONFIG)


def test_assemble_collapses_and_fills():
    outcomes = (McqAction.ADD, McqAction.HOLD, McqAction.CUT, McqAction.TRIM_HALF)
    texts = dict(zip(outcomes, ["a", "b", "c", "d"], strict=True))
    sourced = [
        (OptionSource.CURRENT, McqAction.HOLD),
        (OptionSource.PRE_UPDATE, McqAction.HOLD),
        (OptionSource.STATED_PROFILE, McqAction.ADD),
    ]
    opts = assemble_action_options(outcomes, texts, sourced, np.random.default_rng(3))
    assert sorted(opts.sources) == sorted(
        [OptionSource.CURRENT, OptionSource.STATED_PROFILE, OptionSource.NONE, OptionSource.NONE]
    )
    assert sorted(opts.texts) == ["a", "b", "c", "d"]
    assert opts.texts["ABCD".index(opts.answer)] == "b"
    again = assemble_action_options(outcomes, texts, sourced, np.random.default_rng(3))
    assert again == opts
    with pytest.raises(ValueError):
        bad = [(OptionSource.CURRENT, McqAction.HEDGE)]
        assemble_action_options(outcomes, texts, bad, np.random.default_rng(3))


def test_assemble_value_options():
    opts = assemble_value_options(["x", "y", "z"], "x", "y", {"z"}, np.random.default_rng(0))
    assert sorted(opts.sources) == sorted(
        [OptionSource.CURRENT, OptionSource.PRE_UPDATE, OptionSource.THIRD_PARTY]
    )
    assert opts.texts["ABC".index(opts.answer)] == "x"
    with pytest.raises(ValueError):
        assemble_value_options(["x", "y"], "q", None, set(), np.random.default_rng(0))
