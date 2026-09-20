import math

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


def test_degenerate_truncation_window_raises() -> None:
    with pytest.raises(ValidationError):
        LogNormalSpec(median=2.0, sigma=0.25, lo=20.0)


def test_truncated_cdf_below_lo_is_zero() -> None:
    spec = BetaSpec(a=9, b=12, lo=0.1, hi=0.9)
    assert spec.cdf(0.05) == 0.0


def test_truncated_cdf_above_hi_is_one() -> None:
    spec = BetaSpec(a=9, b=12, lo=0.1, hi=0.9)
    assert spec.cdf(0.95) == 1.0


def test_truncated_cdf_inside_bounds_is_strictly_between_zero_and_one() -> None:
    spec = BetaSpec(a=9, b=12, lo=0.1, hi=0.9)
    value = spec.cdf(0.5)
    assert 0.0 < value < 1.0


def test_truncated_cdf_ppf_round_trip() -> None:
    spec = LogNormalSpec(median=2.0, sigma=0.25, lo=1.5, hi=3.0)
    u = np.linspace(0.01, 0.99, 50)
    assert np.allclose(spec.cdf(spec.ppf(u)), u)


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


def test_narrow_window_near_upper_bound_does_not_return_inf() -> None:
    # Window is ~1e-6 wide; the mapped probability at u close to 1 rounds to
    # exactly 1.0 in float64 before the fix, which made scipy's ppf return inf.
    spec = LogNormalSpec(median=2.0, sigma=0.25, lo=6.563363878)
    value = spec.ppf(1 - 1e-12)
    assert math.isfinite(value)


def test_narrow_window_over_an_array_never_returns_inf() -> None:
    spec = LogNormalSpec(median=2.0, sigma=0.25, lo=6.563363878)
    values = spec.ppf(np.array([0.5, 1 - 1e-12, 1 - 1e-10]))
    assert np.all(np.isfinite(values))


def test_non_finite_ppf_result_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = LogNormalSpec(median=1.1, sigma=0.10)
    frozen = spec._frozen()

    class _AlwaysInf:
        def cdf(self, x: float | None) -> float:
            return 0.0 if x is None else frozen.cdf(x)

        def ppf(self, q: np.ndarray) -> np.ndarray:
            return np.full_like(np.asarray(q, dtype=float), np.inf)

    monkeypatch.setattr(spec, "_frozen", lambda: _AlwaysInf())
    with pytest.raises(ValueError, match="non-finite"):
        spec.ppf(0.5)


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
