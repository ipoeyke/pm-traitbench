"""Tests for the position-side bias rules and the completed bias-param registry."""

import math

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.engine.biases import BIAS_RULES, _check_registry
from pm_traitbench.engine.biases import anchoring as anchoring_module
from pm_traitbench.engine.biases import disposition as disposition_module
from pm_traitbench.engine.biases import exit_deficiency as exit_deficiency_module
from pm_traitbench.engine.biases import loss_aversion as loss_aversion_module
from pm_traitbench.engine.biases.disposition import draw_sell, sell_hazard
from pm_traitbench.engine.biases.disposition import flag as disposition_flag
from pm_traitbench.engine.biases.exit_deficiency import LATE_ROLL_FLAG, Response, respond
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import PnlState, RuleResponse
from pm_traitbench.rng import stream


def _params(values: dict[str, float], active: set[str]) -> EffectiveParams:
    """An EffectiveParams over all eight BIAS_PARAMS, with only the given overrides."""
    return EffectiveParams(
        values={param: values.get(param, 0.5) for param in BIAS_PARAMS},
        active={param: param in active for param in BIAS_PARAMS},
    )


# --- registry ----------------------------------------------------------------


def test_bias_rules_registers_the_four_position_side_modules() -> None:
    assert BIAS_RULES["loss_aversion_lambda"] is loss_aversion_module
    assert BIAS_RULES["disposition_ratio"] is disposition_module
    assert BIAS_RULES["anchoring_rho"] is anchoring_module
    assert BIAS_RULES["exit_deficiency"] is exit_deficiency_module


def test_bias_rules_keys_equal_bias_params() -> None:
    assert set(BIAS_RULES) == set(BIAS_PARAMS)


def test_check_registry_raises_on_mismatch() -> None:
    with pytest.raises(ValueError):
        _check_registry({"a": object()}, ("a", "b"))


def test_check_registry_passes_on_match() -> None:
    _check_registry({"a": object(), "b": object()}, ("a", "b"))


# --- exit_deficiency.respond ---------------------------------------------------


def test_respond_e_zero_always_acted() -> None:
    params = _params({"exit_deficiency": 0.0}, {"loss_aversion_lambda"})
    rng = stream(1, "exit-def-e0")
    for _ in range(200):
        result = respond(params, at_loss=True, rng=rng)
        assert result == Response(RuleResponse.ACTED, None)


def test_respond_e_one_never_acted() -> None:
    params = _params({"exit_deficiency": 1.0}, {"loss_aversion_lambda"})
    rng = stream(1, "exit-def-e1")
    for _ in range(200):
        result = respond(params, at_loss=True, rng=rng)
        assert result.response != RuleResponse.ACTED


def test_respond_adds_only_when_lambda_active_and_at_loss() -> None:
    params = _params({"exit_deficiency": 1.0}, {"loss_aversion_lambda"})
    rng = stream(1, "exit-def-add")
    for _ in range(200):
        result = respond(params, at_loss=True, rng=rng)
        assert result == Response(RuleResponse.ADDED, "exit_deficiency:added")


def test_respond_acked_no_action_when_not_at_loss() -> None:
    params = _params({"exit_deficiency": 1.0}, {"loss_aversion_lambda"})
    rng = stream(1, "exit-def-noloss")
    for _ in range(200):
        result = respond(params, at_loss=False, rng=rng)
        assert result == Response(RuleResponse.ACKED_NO_ACTION, None)


def test_respond_acked_no_action_when_lambda_inactive() -> None:
    params = _params({"exit_deficiency": 1.0}, set())
    rng = stream(1, "exit-def-inactive")
    for _ in range(200):
        result = respond(params, at_loss=True, rng=rng)
        assert result == Response(RuleResponse.ACKED_NO_ACTION, None)


def test_respond_breach_share_matches_e_over_many_draws() -> None:
    # False-alarm rate: tolerance 0.02 at SE about sqrt(0.06*0.94/2000) ~= 0.0053
    # is about 3.8 SE two-sided, roughly 1 in 6,000.
    params = _params({"exit_deficiency": 0.06}, {"loss_aversion_lambda"})
    rng = stream(1, "exit-def-share")
    breaches = sum(
        respond(params, at_loss=False, rng=rng).response != RuleResponse.ACTED for _ in range(2000)
    )
    share = breaches / 2000
    assert share == pytest.approx(0.06, abs=0.02)


def test_late_roll_flag_constant() -> None:
    assert LATE_ROLL_FLAG == "exit_deficiency:late_roll"


# --- disposition.sell_hazard, draw_sell, flag ----------------------------------


def test_sell_hazard_at_d_one_equals_base_at_zero_progress() -> None:
    config = Config()
    params = _params({"disposition_ratio": 1.0}, {"disposition_ratio"})
    for pnl_state in (PnlState.GAIN, PnlState.LOSS, PnlState.FLAT):
        h = sell_hazard(0.0, pnl_state, params, config)
        assert h == pytest.approx(config.engine.base_hazard)


def test_sell_hazard_gain_to_loss_ratio_equals_d() -> None:
    config = Config()
    params = _params({"disposition_ratio": 3.0}, {"disposition_ratio"})
    h_gain = sell_hazard(0.2, PnlState.GAIN, params, config)
    h_loss = sell_hazard(0.2, PnlState.LOSS, params, config)
    assert h_gain / h_loss == pytest.approx(3.0)


def test_sell_hazard_gain_matches_absolute_formula_at_d_three() -> None:
    # Pins the exact scaling (base * (1 + progress) * sqrt(D)), not just a ratio,
    # so a symmetric error that scales both gain and loss the same way cannot pass.
    config = Config()
    params = _params({"disposition_ratio": 3.0}, {"disposition_ratio"})
    h = sell_hazard(0.2, PnlState.GAIN, params, config)
    expected = config.engine.base_hazard * 1.2 * math.sqrt(3.0)
    assert h == pytest.approx(expected)


def test_sell_hazard_negative_progress_is_clamped_to_base() -> None:
    config = Config()
    params = _params({"disposition_ratio": 3.0}, {"disposition_ratio"})
    h = sell_hazard(-5.0, PnlState.FLAT, params, config)
    assert h == pytest.approx(config.engine.base_hazard)


def test_sell_hazard_flat_is_unchanged_by_d() -> None:
    config = Config()
    params_d1 = _params({"disposition_ratio": 1.0}, {"disposition_ratio"})
    params_d3 = _params({"disposition_ratio": 3.0}, {"disposition_ratio"})
    h1 = sell_hazard(0.4, PnlState.FLAT, params_d1, config)
    h3 = sell_hazard(0.4, PnlState.FLAT, params_d3, config)
    assert h1 == pytest.approx(h3)


def test_sell_hazard_clips_to_one() -> None:
    config = Config()
    params = _params({"disposition_ratio": 100.0}, {"disposition_ratio"})
    h = sell_hazard(100.0, PnlState.GAIN, params, config)
    assert h == pytest.approx(1.0)


def test_draw_sell_zero_hazard_never_sells() -> None:
    rng = stream(1, "draw-sell-zero")
    assert all(not draw_sell(0.0, rng) for _ in range(200))


def test_draw_sell_one_hazard_always_sells() -> None:
    rng = stream(1, "draw-sell-one")
    assert all(draw_sell(1.0, rng) for _ in range(200))


def test_disposition_flag_realise_gain_early() -> None:
    params = _params({"disposition_ratio": 1.5}, {"disposition_ratio"})
    assert disposition_flag(PnlState.GAIN, 0.5, True, params) == "disposition:realise_gain_early"


def test_disposition_flag_hold_loser() -> None:
    params = _params({"disposition_ratio": 1.5}, {"disposition_ratio"})
    assert disposition_flag(PnlState.LOSS, 0.5, False, params) == "disposition:hold_loser"


@pytest.mark.parametrize(
    ("pnl_state", "progress", "sold", "active"),
    [
        (PnlState.GAIN, 1.0, True, True),
        (PnlState.GAIN, 0.5, False, True),
        (PnlState.LOSS, 0.5, True, True),
        (PnlState.FLAT, 0.5, True, True),
        (PnlState.GAIN, 0.5, True, False),
        (PnlState.LOSS, 0.5, False, False),
    ],
)
def test_disposition_flag_none_cases(
    pnl_state: PnlState, progress: float, sold: bool, active: bool
) -> None:
    params = _params({"disposition_ratio": 1.5}, {"disposition_ratio"} if active else set())
    assert disposition_flag(pnl_state, progress, sold, params) is None
