import statistics

from scipy import stats

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.enums import Regime
from pm_traitbench.errors import SamplingError
from pm_traitbench.rng import stream
from pm_traitbench.sampling.biases import BiasDraw, sample_biases

_UNMAPPED = ("overconfidence_coverage", "conviction_size_miscalibration", "exit_deficiency")


def _rngs(i: int):
    return stream(1, "t", i, "activation"), stream(1, "t", i, "biases"), stream(1, "t", i, "regime")


def _draw(config: Config, i: int) -> list[BiasDraw]:
    rng_activation, rng_values, rng_regime = _rngs(i)
    return sample_biases(config, rng_activation, rng_values, rng_regime)


def _sample_many(config: Config, n: int) -> list[list[BiasDraw]]:
    return [_draw(config, i) for i in range(n)]


def test_returns_eight_draws_in_bias_params_order():
    draws = _draw(Config(), 0)
    assert [d.param for d in draws] == list(BIAS_PARAMS)


def test_bias_draw_is_hashable():
    draw = _draw(Config(), 0)[0]
    assert isinstance(hash(draw), int)


def test_multiplier_returns_the_value_for_the_requested_regime():
    draw = _draw(Config(), 0)[0]
    for regime, value in draw.multipliers:
        assert draw.multiplier(regime) == value


def test_multipliers_are_in_regime_enum_order():
    draw = _draw(Config(), 0)[0]
    assert [regime for regime, _ in draw.multipliers] == list(Regime)


def test_fixed_streams_give_equal_draws_across_two_calls():
    assert _draw(Config(), 0) == _draw(Config(), 0)


def test_every_pm_has_at_least_min_active_active_biases():
    config = Config()
    for i in range(500):
        draws = _draw(config, i)
        assert sum(d.active for d in draws) >= config.biases.min_active


def test_activation_failure_raises_sampling_error_naming_p_active_and_min_active():
    config = Config.model_validate({"biases": {"p_active": 0.0, "min_active": 2}})
    try:
        _draw(config, 0)
    except SamplingError as e:
        message = str(e)
        assert "p_active" in message
        assert "min_active" in message
    else:
        raise AssertionError("expected SamplingError")


def test_zero_min_active_and_zero_p_active_gives_all_inactive():
    config = Config.model_validate({"biases": {"p_active": 0.0, "min_active": 0}})
    draws = _draw(config, 0)
    assert all(not d.active for d in draws)
    assert all(d.strength == 0.0 for d in draws)
    assert all(m == 1.0 for d in draws for _, m in d.multipliers)


def test_active_values_respect_floors_and_unit_bounds():
    config = Config()
    floors = {"loss_aversion_lambda": 1.5, "disposition_ratio": 1.2}
    for i in range(500):
        for draw in _draw(config, i):
            if not draw.active:
                continue
            if draw.param in floors:
                assert draw.value >= floors[draw.param]
            else:
                assert 0.0 <= draw.value <= 1.0


def test_multipliers_differ_from_one_only_on_active_mapped_biases_within_bounds():
    config = Config.model_validate({"biases": {"p_regime_cluster": 1.0}})
    for i in range(200):
        for draw in _draw(config, i):
            spec = config.biases.params[draw.param]
            for regime, multiplier in draw.multipliers:
                assert 1.0 <= multiplier <= 1.5
                if multiplier != 1.0:
                    assert draw.active
                    assert spec.cluster_regime == regime


def test_unmapped_biases_never_cluster():
    config = Config.model_validate({"biases": {"p_regime_cluster": 1.0}})
    for i in range(200):
        for draw in _draw(config, i):
            if draw.param in _UNMAPPED:
                assert all(m == 1.0 for _, m in draw.multipliers)


def test_full_cluster_probability_gives_mapped_multiplier_at_least_one_and_sometimes_above():
    config = Config.model_validate({"biases": {"p_regime_cluster": 1.0}})
    any_above_one = False
    for i in range(200):
        for draw in _draw(config, i):
            spec = config.biases.params[draw.param]
            if draw.active and spec.cluster_regime is not None:
                multiplier = draw.multiplier(spec.cluster_regime)
                assert multiplier >= 1.0
                any_above_one = any_above_one or multiplier > 1.0
    assert any_above_one


def test_zero_cluster_probability_gives_all_multipliers_one():
    config = Config.model_validate({"biases": {"p_regime_cluster": 0.0}})
    for i in range(200):
        for draw in _draw(config, i):
            assert all(m == 1.0 for _, m in draw.multipliers)


def test_inactive_draws_follow_the_neutral_marginal_for_the_unmapped_biases():
    # Confirms inactive draws use the neutral distribution, not the active
    # one: an accidental active/neutral swap would move these medians by
    # roughly 0.35-0.40, far outside this band.
    config = Config()
    n = 3000
    pms = _sample_many(config, n)
    for param in ("exit_deficiency", "conviction_size_miscalibration"):
        idx = list(BIAS_PARAMS).index(param)
        inactive_values = [draws[idx].value for draws in pms if not draws[idx].active]
        assert inactive_values
        neutral_median = config.biases.params[param].neutral.median_value()
        median = statistics.median(inactive_values)
        assert abs(median - neutral_median) <= 0.05


def test_statistical_properties_over_many_draws():
    config = Config()
    n = 5000
    pms = _sample_many(config, n)
    by_param = {param: [draws[idx] for draws in pms] for idx, param in enumerate(BIAS_PARAMS)}

    for param in BIAS_PARAMS:
        share_active = sum(d.active for d in by_param[param]) / n
        assert 0.36 <= share_active <= 0.44, (param, share_active)

    active_lambda = [d.value for d in by_param["loss_aversion_lambda"] if d.active]
    assert 1.9 <= statistics.median(active_lambda) <= 2.3

    active_rho = [d.value for d in by_param["anchoring_rho"] if d.active]
    expected_rho_median = float(stats.beta(9, 12).median())
    assert abs(statistics.median(active_rho) - expected_rho_median) <= 0.05

    lambda_vals, disposition_vals = [], []
    overconf_vals, herd_vals = [], []
    for draws in pms:
        by_p = {d.param: d for d in draws}
        if by_p["loss_aversion_lambda"].active and by_p["disposition_ratio"].active:
            lambda_vals.append(by_p["loss_aversion_lambda"].value)
            disposition_vals.append(by_p["disposition_ratio"].value)
        if by_p["overconfidence_coverage"].active and by_p["herding_weight"].active:
            overconf_vals.append(by_p["overconfidence_coverage"].value)
            herd_vals.append(by_p["herding_weight"].value)

    lambda_disposition_corr, _ = stats.spearmanr(lambda_vals, disposition_vals)
    assert lambda_disposition_corr > 0.2

    overconf_herd_corr, _ = stats.spearmanr(overconf_vals, herd_vals)
    assert overconf_herd_corr < -0.1

    active_mapped_count = 0
    cluster_count = 0
    for draws in pms:
        for d in draws:
            spec = config.biases.params[d.param]
            if d.active and spec.cluster_regime is not None:
                active_mapped_count += 1
                if d.multiplier(spec.cluster_regime) != 1.0:
                    cluster_count += 1
    cluster_rate = cluster_count / active_mapped_count
    assert 0.45 <= cluster_rate <= 0.55

    active_overconf = [d for d in by_param["overconfidence_coverage"] if d.active]
    values = [d.value for d in active_overconf]
    strengths = [d.strength for d in active_overconf]
    value_strength_corr, _ = stats.spearmanr(values, strengths)
    assert value_strength_corr < -0.99
