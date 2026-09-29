import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.engine.adapters import adapter_for
from pm_traitbench.enums import AssetClass, StreetView
from pm_traitbench.errors import ProbesError
from pm_traitbench.probes.actions import horizons
from pm_traitbench.probes.situations import MarketEnv, day_index, situation_for

CONFIG = Config()
HORIZONS = horizons(CONFIG)
SUB_STYLES = {
    AssetClass.EQUITIES: "equity_long_short",
    AssetClass.RATES_CREDIT: "long_short_credit",
}


@pytest.fixture
def make_env(fixture_view, neutral_pm):
    def build(asset_class):
        _, _, rules = neutral_pm(asset_class, SUB_STYLES[asset_class])
        adapter = adapter_for(asset_class, SUB_STYLES[asset_class], CONFIG.engine.horizon_days)
        universe = adapter.universe(fixture_view.instruments, rules)
        return MarketEnv(fixture_view, adapter, universe, asset_class), rules

    return build


ASSET_CLASSES = [AssetClass.EQUITIES, AssetClass.RATES_CREDIT]


def rng0():
    return np.random.default_rng(0)


def build_situation(param, env, rules, t=25, seed=0):
    return situation_for(param, env, t, rules, HORIZONS, CONFIG, np.random.default_rng(seed))


@pytest.mark.parametrize("asset_class", ASSET_CLASSES)
def test_loss_aversion_places_current_between_entry_and_stop(make_env, asset_class):
    env, rules = make_env(asset_class)
    sit = build_situation("loss_aversion_lambda", env, rules)
    lv = sit.levels
    assert min(lv["entry"], lv["stop"]) < lv["current"] < max(lv["entry"], lv["stop"])
    depth = (lv["current"] - lv["entry"]) / (lv["stop"] - lv["entry"])
    assert depth == pytest.approx(CONFIG.probes.loss_depth)
    assert (lv["target"] - lv["entry"]) * (lv["stop"] - lv["entry"]) < 0
    assert sit.slots["horizon"] == "34"
    assert {"instrument", "level"} <= sit.slots.keys()


def test_rates_credit_stop_is_above_entry(make_env):
    env, rules = make_env(AssetClass.RATES_CREDIT)
    sit = build_situation("loss_aversion_lambda", env, rules)
    assert sit.levels["stop"] > sit.levels["entry"]
    assert sit.levels["target"] < sit.levels["entry"]


def test_equity_stop_is_below_entry(make_env):
    env, rules = make_env(AssetClass.EQUITIES)
    sit = build_situation("loss_aversion_lambda", env, rules)
    assert sit.levels["stop"] < sit.levels["entry"]


@pytest.mark.parametrize("asset_class", ASSET_CLASSES)
def test_disposition_places_current_partway_to_target(make_env, asset_class):
    env, rules = make_env(asset_class)
    sit = build_situation("disposition_ratio", env, rules)
    lv = sit.levels
    assert min(lv["entry"], lv["target"]) < lv["current"] < max(lv["entry"], lv["target"])
    progress = (lv["current"] - lv["entry"]) / (lv["target"] - lv["entry"])
    assert progress == pytest.approx(CONFIG.probes.disposition_progress)
    assert sit.slots["horizon"] == "14"


@pytest.mark.parametrize("asset_class", ASSET_CLASSES)
def test_anchoring_round_level_lies_between_current_and_target(make_env, asset_class):
    env, rules = make_env(asset_class)
    for seed in range(5):
        sit = build_situation("anchoring_rho", env, rules, seed=seed)
        if sit is None:
            continue
        lv = sit.levels
        assert (
            min(lv["current"], lv["target"]) < lv["round_level"] < max(lv["current"], lv["target"])
        )
        assert (
            min(lv["entry"], lv["round_level"])
            < lv["current"]
            < max(lv["entry"], lv["round_level"])
        )
        return
    pytest.fail("no anchoring situation found")


def test_anchoring_returns_none_without_a_qualifying_round_level(make_env, monkeypatch):
    env, rules = make_env(AssetClass.EQUITIES)
    one = MarketEnv(env.view, env.adapter, env.universe[:1], env.asset_class)
    config = CONFIG.model_copy(
        update={"probes": CONFIG.probes.model_copy(update={"situation_attempts": 1})}
    )
    baseline = situation_for("anchoring_rho", one, 25, rules, HORIZONS, config, rng0())
    entry = baseline.levels["entry"]
    monkeypatch.setattr(type(env.adapter), "round_step", lambda self, series, level: entry)
    assert situation_for("anchoring_rho", one, 25, rules, HORIZONS, config, rng0()) is None


@pytest.mark.parametrize("asset_class", ASSET_CLASSES)
def test_exit_deficiency_stop_is_current(make_env, asset_class):
    env, rules = make_env(asset_class)
    lv = build_situation("exit_deficiency", env, rules).levels
    assert lv["stop"] == lv["current"]
    assert lv["entry"] != lv["current"]


def test_herding_none_when_street_is_neutral_everywhere(make_env):
    env, rules = make_env(AssetClass.EQUITIES)
    neutral_t = next(
        t
        for t in range(len(env.view.dates))
        if all(env.view.street_view(i, t) == StreetView.NEUTRAL for i in env.universe)
    )
    assert build_situation("herding_weight", env, rules, t=neutral_t) is None


def test_herding_own_side_opposes_street(make_env):
    env, rules = make_env(AssetClass.EQUITIES)
    t = next(
        t
        for t in range(len(env.view.dates))
        if any(env.view.street_view(i, t) != StreetView.NEUTRAL for i in env.universe)
    )
    sit = build_situation("herding_weight", env, rules, t=t)
    street = sit.slots["street"]
    assert street in ("overweight", "underweight")
    assert sit.slots["own_side"] == ("sell" if street == "overweight" else "buy")


def test_extrapolation_and_conviction_slots(make_env):
    env, rules = make_env(AssetClass.EQUITIES)
    sit = build_situation("extrapolation_theta", env, rules)
    assert sit.slots["thesis_sd"] == "1.0"
    assert sit.slots["trailing_sd"] == "3.0"
    assert sit.slots["horizon"] == str(CONFIG.engine.horizon_days)
    assert sit.levels == {"current": sit.levels["current"]}
    assert build_situation("conviction_size_miscalibration", env, rules).slots["rating"] == "2"


def test_missing_stop_loss_rule_raises(make_env):
    env, rules = make_env(AssetClass.EQUITIES)
    rules = [r for r in rules if r.param != "stop_loss"]
    with pytest.raises(ProbesError):
        build_situation("loss_aversion_lambda", env, rules)


def test_same_seed_same_instrument(make_env):
    env, rules = make_env(AssetClass.EQUITIES)
    a = build_situation("loss_aversion_lambda", env, rules, seed=3)
    b = build_situation("loss_aversion_lambda", env, rules, seed=3)
    assert a == b


def test_day_index(fixture_view):
    assert day_index(fixture_view, fixture_view.dates[7]) == 7
    with pytest.raises(ProbesError, match="1999"):
        day_index(fixture_view, fixture_view.dates[0].replace(year=1999))
