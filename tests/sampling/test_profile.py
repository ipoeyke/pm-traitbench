import pytest

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import Regime, Typicality
from pm_traitbench.errors import SamplingError
from pm_traitbench.rng import stream
from pm_traitbench.sampling.biases import BiasDraw
from pm_traitbench.sampling.profile import sample_self_description


def _multipliers() -> dict[Regime, float]:
    return {regime: 1.0 for regime in Regime}


def _bias(param: str, active: bool, strength: float) -> BiasDraw:
    return BiasDraw(
        param=param, value=1.0, active=active, strength=strength, multipliers=_multipliers()
    )


def _candidates(
    catalogue: Catalogue, param_a: str, param_b: str, typicality: Typicality
) -> set[str]:
    attr = "agree" if typicality == Typicality.TYPICAL else "contradict"
    options_a = getattr(catalogue.self_descriptions[param_a], attr)
    options_b = getattr(catalogue.self_descriptions[param_b], attr)
    return {f"{a}, {b}" for a in options_a for b in options_b}


def test_typical_pm_picks_only_from_agree_phrasings_of_two_strongest(fixture_catalogue: Catalogue):
    biases = [_bias(param, True, 0.9 - 0.01 * i) for i, param in enumerate(BIAS_PARAMS)]
    candidates = _candidates(fixture_catalogue, BIAS_PARAMS[0], BIAS_PARAMS[1], Typicality.TYPICAL)
    for i in range(50):
        rng = stream(1, "t", i, "profile")
        description = sample_self_description(biases, Typicality.TYPICAL, fixture_catalogue, rng)
        assert description in candidates


def test_anti_typical_pm_picks_only_from_contradict_phrasings(fixture_catalogue: Catalogue):
    biases = [_bias(param, True, 0.9 - 0.01 * i) for i, param in enumerate(BIAS_PARAMS)]
    candidates = _candidates(
        fixture_catalogue, BIAS_PARAMS[0], BIAS_PARAMS[1], Typicality.ANTI_TYPICAL
    )
    for i in range(50):
        rng = stream(1, "t", i, "profile")
        description = sample_self_description(
            biases, Typicality.ANTI_TYPICAL, fixture_catalogue, rng
        )
        assert description in candidates


def test_ties_in_strength_break_by_bias_params_order(fixture_catalogue: Catalogue):
    # Biases are fed in reversed BIAS_PARAMS order with equal strengths.
    # Python's sort is stable, so without an explicit BIAS_PARAMS-order
    # tie-break key, the top two would be the last two BIAS_PARAMS entries
    # instead of the first two.
    biases = [_bias(param, True, 0.5) for param in reversed(BIAS_PARAMS)]
    candidates = _candidates(fixture_catalogue, BIAS_PARAMS[0], BIAS_PARAMS[1], Typicality.TYPICAL)
    for i in range(50):
        rng = stream(1, "t", i, "profile")
        description = sample_self_description(biases, Typicality.TYPICAL, fixture_catalogue, rng)
        assert description in candidates


def test_inactive_biases_are_ignored_even_with_high_strength(fixture_catalogue: Catalogue):
    biases = [
        _bias(BIAS_PARAMS[0], False, 0.99),
        _bias(BIAS_PARAMS[1], True, 0.2),
        _bias(BIAS_PARAMS[2], True, 0.1),
    ]
    candidates = _candidates(fixture_catalogue, BIAS_PARAMS[1], BIAS_PARAMS[2], Typicality.TYPICAL)
    for i in range(50):
        rng = stream(1, "t", i, "profile")
        description = sample_self_description(biases, Typicality.TYPICAL, fixture_catalogue, rng)
        assert description in candidates


def test_fewer_than_two_active_biases_raises_sampling_error(fixture_catalogue: Catalogue):
    biases = [_bias(BIAS_PARAMS[0], True, 0.5)]
    rng = stream(1, "t", 0, "profile")
    with pytest.raises(SamplingError):
        sample_self_description(biases, Typicality.TYPICAL, fixture_catalogue, rng)


def test_zero_active_biases_raises_sampling_error(fixture_catalogue: Catalogue):
    biases = [_bias(param, False, 0.5) for param in BIAS_PARAMS]
    rng = stream(1, "t", 0, "profile")
    with pytest.raises(SamplingError):
        sample_self_description(biases, Typicality.TYPICAL, fixture_catalogue, rng)
