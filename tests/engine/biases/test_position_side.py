"""Tests for the position-side bias rules and the completed bias-param registry."""

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.engine.biases import BIAS_RULES, _check_registry
from pm_traitbench.engine.biases import anchoring as anchoring_module
from pm_traitbench.engine.biases import disposition as disposition_module
from pm_traitbench.engine.biases import exit_deficiency as exit_deficiency_module
from pm_traitbench.engine.biases import loss_aversion as loss_aversion_module
from pm_traitbench.engine.biases.anchoring import evaluate
from pm_traitbench.engine.biases.anchoring import flag as anchoring_flag
from pm_traitbench.engine.biases.disposition import draw_sell, sell_hazard
from pm_traitbench.engine.biases.disposition import flag as disposition_flag
from pm_traitbench.engine.biases.exit_deficiency import LATE_ROLL_FLAG, Response, respond
from pm_traitbench.engine.biases.loss_aversion import LossSideChoice, choose
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import Position
from pm_traitbench.enums import Expression, PnlState, PositionAction, RuleResponse, Side
from pm_traitbench.rng import stream


def _params(values: dict[str, float], active: set[str]) -> EffectiveParams:
    """An EffectiveParams over all eight BIAS_PARAMS, with only the given overrides."""
    return EffectiveParams(
        values={param: values.get(param, 0.5) for param in BIAS_PARAMS},
        active={param: param in active for param in BIAS_PARAMS},
    )


def _position(
    *, side: Side = Side.BUY, bullish_sign: int = 1, target_level: float = 110.0
) -> Position:
    series = Series(
        legs=(LegRef(instrument_id="EQ-0001", tenor=None, coeff=1.0),),
        bullish_sign=bullish_sign,
        unit="pct",
    )
    return Position(
        trade_idea_id="ti_001",
        expression=Expression.OUTRIGHT,
        instrument_id="EQ-0001",
        legs=(),
        series=series,
        side=side,
        entry_t=0,
        entry_level=100.0,
        target_level=target_level,
        stop_level=90.0,
        sd_h_at_entry=1.0,
        forecast=105.0,
        size_pct_book=1.0,
        original_size_pct_book=1.0,
        conviction=1,
        size_rank=1,
        triggers_fired=0,
        consumed_rule_ids=frozenset(),
        run_counters=(),
        size_changed_t=0,
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
    # is about 3.8 SE, well under a 1-in-10,000 chance of a spurious failure.
    params = _params({"exit_deficiency": 0.06}, {"loss_aversion_lambda"})
    rng = stream(1, "exit-def-share")
    breaches = sum(
        respond(params, at_loss=False, rng=rng).response != RuleResponse.ACTED for _ in range(2000)
    )
    share = breaches / 2000
    assert share == pytest.approx(0.06, abs=0.02)


def test_late_roll_flag_constant() -> None:
    assert LATE_ROLL_FLAG == "exit_deficiency:late_roll"


# --- loss_aversion.choose -------------------------------------------------------


def _share(action: PositionAction, choices: list[LossSideChoice]) -> float:
    return sum(1 for c in choices if c.action == action) / len(choices)


def test_choose_raising_lambda_raises_add_or_hold_share_and_lowers_cut_share() -> None:
    # Direction comparison over many draws, no seed hunting.
    config = Config()
    low = _params({"loss_aversion_lambda": 1.0}, {"loss_aversion_lambda"})
    high = _params({"loss_aversion_lambda": 4.0}, {"loss_aversion_lambda"})
    rng_low = stream(1, "loss-aversion-lam-low")
    rng_high = stream(1, "loss-aversion-lam-high")

    choices_low = [choose(-1.0, 0.5, low, config, rng_low, add_allowed=True) for _ in range(2000)]
    choices_high = [
        choose(-1.0, 0.5, high, config, rng_high, add_allowed=True) for _ in range(2000)
    ]

    add_or_hold_low = _share(PositionAction.ADD, choices_low) + _share(
        PositionAction.HOLD, choices_low
    )
    add_or_hold_high = _share(PositionAction.ADD, choices_high) + _share(
        PositionAction.HOLD, choices_high
    )
    cut_low = _share(PositionAction.CUT, choices_low)
    cut_high = _share(PositionAction.CUT, choices_high)

    assert add_or_hold_high > add_or_hold_low
    assert cut_high < cut_low


def test_choose_add_not_allowed_never_returns_add() -> None:
    config = Config()
    params = _params({"loss_aversion_lambda": 4.0}, {"loss_aversion_lambda"})
    rng = stream(1, "loss-aversion-no-add")
    for _ in range(500):
        choice = choose(-1.0, 0.5, params, config, rng, add_allowed=False)
        assert choice.action != PositionAction.ADD


def test_choose_flags_match_action_when_active() -> None:
    config = Config()
    params = _params({"loss_aversion_lambda": 4.0}, {"loss_aversion_lambda"})
    rng = stream(1, "loss-aversion-flags")
    for _ in range(500):
        choice = choose(-1.0, 0.5, params, config, rng, add_allowed=True)
        if choice.action == PositionAction.ADD:
            assert choice.flag == "loss_aversion:add"
        elif choice.action == PositionAction.HOLD:
            assert choice.flag == "loss_aversion:hold"
        else:
            assert choice.flag is None


def test_choose_no_flag_when_inactive() -> None:
    config = Config()
    params = _params({"loss_aversion_lambda": 4.0}, set())
    rng = stream(1, "loss-aversion-inactive")
    for _ in range(500):
        choice = choose(-1.0, 0.5, params, config, rng, add_allowed=True)
        assert choice.flag is None


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


# --- anchoring.evaluate, flag --------------------------------------------------


def test_evaluate_rho_zero_effective_equals_target() -> None:
    params = _params({"anchoring_rho": 0.0}, {"anchoring_rho"})
    pos = _position(side=Side.BUY, bullish_sign=1, target_level=110.0)
    result = evaluate(pos, 100.0, [105.0, 108.0], params)
    assert result.effective_exit_level == pytest.approx(pos.target_level)


def test_evaluate_rho_one_effective_equals_anchor() -> None:
    params = _params({"anchoring_rho": 1.0}, {"anchoring_rho"})
    pos = _position(side=Side.BUY, bullish_sign=1, target_level=110.0)
    result = evaluate(pos, 100.0, [105.0, 108.0], params)
    assert result.effective_exit_level == pytest.approx(result.anchor_level)


def test_evaluate_picks_nearest_candidate_on_target_side_for_buy() -> None:
    params = _params({"anchoring_rho": 1.0}, {"anchoring_rho"})
    pos = _position(side=Side.BUY, bullish_sign=1, target_level=110.0)
    result = evaluate(pos, 100.0, [90.0, 103.0, 120.0], params)
    # Candidates above 100 (the target side for a long): 103 and 120; nearest is 103.
    assert result.anchor_level == pytest.approx(103.0)


def test_evaluate_picks_nearest_candidate_on_target_side_for_sell() -> None:
    params = _params({"anchoring_rho": 1.0}, {"anchoring_rho"})
    pos = _position(side=Side.SELL, bullish_sign=1, target_level=90.0)
    result = evaluate(pos, 95.0, [85.0, 92.0, 70.0], params)
    # Candidates below 95 (the target side for a short): 85, 92, 70; nearest is 92.
    assert result.anchor_level == pytest.approx(92.0)


def test_evaluate_no_candidate_on_target_side_falls_back_to_target() -> None:
    params = _params({"anchoring_rho": 1.0}, {"anchoring_rho"})
    pos = _position(side=Side.SELL, bullish_sign=1, target_level=90.0)
    result = evaluate(pos, 95.0, [96.0, 100.0], params)
    assert result.anchor_level == pytest.approx(pos.target_level)
    assert result.effective_exit_level == pytest.approx(pos.target_level)


def test_evaluate_reached_respects_direction_for_buy() -> None:
    params = _params({"anchoring_rho": 0.0}, {"anchoring_rho"})
    pos = _position(side=Side.BUY, bullish_sign=1, target_level=110.0)
    below = evaluate(pos, 105.0, [], params)
    at_or_above = evaluate(pos, 110.0, [], params)
    assert below.reached is False
    assert at_or_above.reached is True


def test_evaluate_reached_respects_direction_for_sell() -> None:
    params = _params({"anchoring_rho": 0.0}, {"anchoring_rho"})
    pos = _position(side=Side.SELL, bullish_sign=1, target_level=90.0)
    above = evaluate(pos, 95.0, [], params)
    at_or_below = evaluate(pos, 90.0, [], params)
    assert above.reached is False
    assert at_or_below.reached is True


def test_anchoring_flag_active_vs_inactive() -> None:
    active_params = _params({"anchoring_rho": 0.5}, {"anchoring_rho"})
    inactive_params = _params({"anchoring_rho": 0.5}, set())
    assert anchoring_flag(active_params) == "anchoring:exit_at_anchor"
    assert anchoring_flag(inactive_params) is None
