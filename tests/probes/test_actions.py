import numpy as np
import pytest

from pm_traitbench.catalogues.loader import PROBE_ACTION_COUNTS
from pm_traitbench.config import Config
from pm_traitbench.enums import OptionSource
from pm_traitbench.probes.actions import (
    NEUTRAL_FACTS,
    PmFacts,
    action_index,
    assemble_action_options,
    assemble_value_options,
    horizons,
    loss_side_outcomes,
)

CONFIG = Config()
HORIZONS = horizons(CONFIG)


def index(param: str, value: float, facts: PmFacts = NEUTRAL_FACTS) -> int:
    return action_index(param, value, facts, HORIZONS, CONFIG)


def test_default_horizons():
    assert horizons(Config()) == {"loss_aversion_lambda": 34, "disposition_ratio": 14}


def test_loss_aversion_typical_at_the_active_floor():
    assert index("loss_aversion_lambda", 1.5) == 0
    assert index("loss_aversion_lambda", 2.0) == 0
    assert index("loss_aversion_lambda", 1.1) == 2


def test_loss_side_outcomes_at_the_neutral_median():
    add, hold, cut = loss_side_outcomes(1.1, NEUTRAL_FACTS, HORIZONS["loss_aversion_lambda"])
    assert add + hold + cut == pytest.approx(1.0)
    assert (add, hold, cut) == pytest.approx((0.15, 0.14, 0.70), abs=0.01)


def test_no_add_rule_scales_the_add_hazard():
    low = PmFacts(no_add_rule=True, lambda_active=True, exit_deficiency=0.06)
    high = PmFacts(no_add_rule=True, lambda_active=True, exit_deficiency=1.0)
    add, hold, cut = loss_side_outcomes(2.0, low, HORIZONS["loss_aversion_lambda"])
    assert (add, hold, cut) == pytest.approx((0.13, 0.34, 0.53), abs=0.01)
    assert index("loss_aversion_lambda", 2.0, low) == 2
    assert index("loss_aversion_lambda", 2.0, high) == 0


def test_disposition_reads_typical_at_the_floor():
    assert index("disposition_ratio", 1.2) == 0
    assert index("disposition_ratio", 1.0) == 1


def test_hazard_param_without_a_horizon_raises():
    with pytest.raises(ValueError):
        action_index("disposition_ratio", 1.2, NEUTRAL_FACTS, {"disposition_ratio": None}, CONFIG)


def test_exit_deficiency_typical_depends_on_lambda():
    active = PmFacts(no_add_rule=False, lambda_active=True, exit_deficiency=0.6)
    assert index("exit_deficiency", 0.6, active) == 1
    assert index("exit_deficiency", 0.6) == 0
    assert index("exit_deficiency", 0.3) == 2


def test_extrapolation_threshold_at_theta_half():
    assert index("extrapolation_theta", 0.5) == 0
    assert index("extrapolation_theta", 0.49) == 1
    assert index("extrapolation_theta", 0.0) == 2


def test_overconfidence_buckets():
    assert index("overconfidence_coverage", 0.8) == 2
    assert index("overconfidence_coverage", 0.55) == 1
    assert index("overconfidence_coverage", 0.4) == 0


def test_conviction_uses_the_redraw_rate():
    assert index("conviction_size_miscalibration", 0.6) == 1
    assert index("conviction_size_miscalibration", 0.7) == 0


def test_action_counts_match_the_bank():
    grid = [i / 20 for i in range(1, 20)]
    lam = [1.0, 1.1, 1.5, 2.0, 3.0]
    for param, count in PROBE_ACTION_COUNTS.items():
        values = lam if param in ("loss_aversion_lambda", "disposition_ratio") else grid
        facts = [NEUTRAL_FACTS, PmFacts(True, True, 0.5)]
        top = max(index(param, v, f) for v in values for f in facts)
        assert top < count


def test_typical_probability_rejects_unsupported_params():
    from pm_traitbench.probes.actions import typical_probability

    with pytest.raises(ValueError):
        typical_probability("extrapolation_theta", 0.5, NEUTRAL_FACTS, None, CONFIG)


def test_assemble_collapses_and_fills():
    actions = ["a", "b", "c", "d"]
    sourced = [
        (OptionSource.CURRENT, 1),
        (OptionSource.PRE_UPDATE, 1),
        (OptionSource.STATED_PROFILE, 0),
    ]
    opts = assemble_action_options(actions, sourced, np.random.default_rng(3))
    assert sorted(opts.sources) == sorted(
        [OptionSource.CURRENT, OptionSource.STATED_PROFILE, OptionSource.NONE, OptionSource.NONE]
    )
    assert sorted(opts.texts) == actions
    assert opts.texts["ABCD".index(opts.answer)] == "b"
    again = assemble_action_options(actions, sourced, np.random.default_rng(3))
    assert again == opts


def test_assemble_value_options():
    opts = assemble_value_options(["x", "y", "z"], "x", "y", {"z"}, np.random.default_rng(0))
    assert sorted(opts.sources) == sorted(
        [OptionSource.CURRENT, OptionSource.PRE_UPDATE, OptionSource.THIRD_PARTY]
    )
    assert opts.texts["ABC".index(opts.answer)] == "x"
    with pytest.raises(ValueError):
        assemble_value_options(["x", "y"], "q", None, set(), np.random.default_rng(0))
