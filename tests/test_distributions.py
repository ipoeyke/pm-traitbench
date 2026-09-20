import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from pm_traitbench.distributions import BetaSpec, Distribution, LogNormalSpec

_DISTRIBUTION_ADAPTER = TypeAdapter(Distribution)


@pytest.mark.parametrize(
    "spec",
    [
        LogNormalSpec(median=1.15, sigma=0.10),
        BetaSpec(a=2, b=12),
    ],
)
def test_ppf_at_half_equals_median_value(spec: LogNormalSpec | BetaSpec) -> None:
    assert spec.ppf(0.5) == spec.median_value()


def test_truncated_lognormal_ppf_stays_within_bounds_and_increases() -> None:
    spec = LogNormalSpec(median=2.0, sigma=0.25, lo=1.5, hi=3.0)
    u = np.linspace(0.001, 0.999, 50)
    values = spec.ppf(u)
    assert np.all(values >= 1.5)
    assert np.all(values <= 3.0)
    assert np.all(np.diff(values) > 0)


def test_truncated_beta_ppf_stays_within_bounds_and_increases() -> None:
    spec = BetaSpec(a=9, b=12, lo=0.1, hi=0.9)
    u = np.linspace(0.001, 0.999, 50)
    values = spec.ppf(u)
    assert np.all(values >= 0.1)
    assert np.all(values <= 0.9)
    assert np.all(np.diff(values) > 0)


def test_truncated_lognormal_with_lo_at_median_stays_above_lo() -> None:
    spec = LogNormalSpec(median=2.0, sigma=0.25, lo=2.0)
    assert spec.ppf(0.0001) >= spec.lo


def test_cdf_ppf_round_trip_for_untruncated_lognormal() -> None:
    spec = LogNormalSpec(median=1.1, sigma=0.10)
    u = np.linspace(0.01, 0.99, 50)
    assert np.allclose(spec.cdf(spec.ppf(u)), u)


def test_cdf_ppf_round_trip_for_untruncated_beta() -> None:
    spec = BetaSpec(a=2, b=10)
    u = np.linspace(0.01, 0.99, 50)
    assert np.allclose(spec.cdf(spec.ppf(u)), u)


def test_lo_equal_to_hi_raises() -> None:
    with pytest.raises(ValidationError):
        LogNormalSpec(median=1.0, sigma=0.1, lo=1.0, hi=1.0)


def test_lo_greater_than_hi_raises() -> None:
    with pytest.raises(ValidationError):
        BetaSpec(a=2, b=10, lo=0.9, hi=0.1)


def test_non_positive_median_raises() -> None:
    with pytest.raises(ValidationError):
        LogNormalSpec(median=0.0, sigma=0.1)


def test_non_positive_beta_param_raises() -> None:
    with pytest.raises(ValidationError):
        BetaSpec(a=0.0, b=5)


def test_discriminated_union_parses_beta() -> None:
    result = _DISTRIBUTION_ADAPTER.validate_python({"kind": "beta", "a": 2, "b": 12})
    assert isinstance(result, BetaSpec)
    assert result.a == 2
    assert result.b == 12


def test_discriminated_union_parses_lognormal() -> None:
    result = _DISTRIBUTION_ADAPTER.validate_python(
        {"kind": "lognormal", "median": 1.1, "sigma": 0.1}
    )
    assert isinstance(result, LogNormalSpec)
    assert result.median == 1.1
