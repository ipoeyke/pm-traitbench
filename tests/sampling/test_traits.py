from pm_traitbench.enums import Kind, Regime
from pm_traitbench.sampling.biases import BiasDraw
from pm_traitbench.sampling.preferences import PreferenceDraw
from pm_traitbench.sampling.traits import build_traits


def _multipliers() -> dict[Regime, float]:
    return {Regime.RANGE: 1.0, Regime.RISK_OFF: 1.0, Regime.RISK_ON: 1.0}


def _bias(param: str, active: bool, multipliers: dict[Regime, float] | None = None) -> BiasDraw:
    return BiasDraw(
        param=param,
        value=1.5,
        active=active,
        strength=0.4,
        multipliers=multipliers or _multipliers(),
    )


def test_bias_traits_get_ids_t01_through_t08_in_given_order():
    biases = [_bias(f"param_{i}", True) for i in range(8)]
    traits = build_traits("pm_001", biases, [])
    assert [t.trait_id for t in traits] == [f"t_{i:02d}" for i in range(1, 9)]
    assert [t.param for t in traits] == [f"param_{i}" for i in range(8)]
    assert all(t.kind == Kind.BIAS for t in traits)


def test_bias_multipliers_copied_into_matching_columns():
    multipliers = {Regime.RANGE: 1.1, Regime.RISK_OFF: 1.2, Regime.RISK_ON: 1.3}
    traits = build_traits("pm_001", [_bias("loss_aversion_lambda", True, multipliers)], [])
    trait = traits[0]
    assert trait.mult_range == 1.1
    assert trait.mult_risk_off == 1.2
    assert trait.mult_risk_on == 1.3


def test_bias_trait_value_and_active_carried_over_from_the_draw():
    traits = build_traits("pm_001", [_bias("loss_aversion_lambda", False)], [])
    trait = traits[0]
    assert trait.value == 1.5
    assert trait.active is False


def test_preference_traits_start_at_t09_and_are_active_with_no_multipliers():
    biases = [_bias(f"param_{i}", True) for i in range(8)]
    preferences = [PreferenceDraw(param="response_format", value="full prose")]
    traits = build_traits("pm_001", biases, preferences)
    pref = traits[8]
    assert pref.trait_id == "t_09"
    assert pref.kind == Kind.PREFERENCE
    assert pref.param == "response_format"
    assert pref.value == "full prose"
    assert pref.active is True
    assert pref.mult_range is None
    assert pref.mult_risk_off is None
    assert pref.mult_risk_on is None


def test_multiple_preferences_get_sequential_ids_in_given_order():
    preferences = [
        PreferenceDraw(param="response_format", value="full prose"),
        PreferenceDraw(param="challenge_tolerance", value="push back if the data disagrees"),
    ]
    traits = build_traits("pm_001", [], preferences)
    assert [t.trait_id for t in traits] == ["t_01", "t_02"]
    assert [t.param for t in traits] == ["response_format", "challenge_tolerance"]


def test_pm_id_is_carried_onto_every_row():
    biases = [_bias("loss_aversion_lambda", True)]
    preferences = [PreferenceDraw(param="response_format", value="full prose")]
    traits = build_traits("pm_042", biases, preferences)
    assert all(t.pm_id == "pm_042" for t in traits)
