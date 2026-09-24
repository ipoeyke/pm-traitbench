"""Overconfidence bias: a narrower stated coverage inflates the entry size factor."""

from scipy.stats import norm

from pm_traitbench.engine.params import EffectiveParams


def z_for_coverage(c: float) -> float:
    """The two-sided normal z-score whose central interval covers probability `c`."""
    return float(norm.ppf((1 + c) / 2))


Z_80 = z_for_coverage(0.8)


def size_factor(params: EffectiveParams) -> tuple[float, str | None]:
    """Size multiplier from the stated coverage's z-score against the 80% reference."""
    coverage = params.value("overconfidence_coverage")
    factor = Z_80 / z_for_coverage(coverage)
    active = params.is_active("overconfidence_coverage")
    flag = "overconfidence:oversized" if active and factor > 1 else None
    return factor, flag
