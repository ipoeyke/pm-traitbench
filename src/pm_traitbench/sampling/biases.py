"""Bias sampling: correlated activation, values, strength and regime multipliers.

A Gaussian copula draws all eight biases from a single correlated normal
vector, then maps each coordinate through its own marginal. This keeps the
published cross-bias correlations while each bias retains its own bounded,
skewed distribution.
"""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from scipy import stats

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.enums import Regime
from pm_traitbench.errors import SamplingError


@dataclass(frozen=True)
class BiasDraw:
    """One PM's sampled value for one bias parameter."""

    param: str
    value: float
    active: bool
    strength: float
    multipliers: dict[Regime, float]


def _draw_activation(config: Config, rng: Generator) -> np.ndarray:
    biases = config.biases
    n = len(BIAS_PARAMS)
    for _ in range(biases.max_activation_attempts):
        active = rng.random(n) < biases.p_active
        if active.sum() >= biases.min_active:
            return active
    raise SamplingError(
        f"could not draw at least min_active={biases.min_active} active biases "
        f"with p_active={biases.p_active}"
    )


def sample_biases(
    config: Config,
    rng_activation: Generator,
    rng_values: Generator,
    rng_regime: Generator,
) -> list[BiasDraw]:
    """Sample all eight bias parameters for one PM, in `BIAS_PARAMS` order."""
    biases = config.biases
    n = len(BIAS_PARAMS)
    active = _draw_activation(config, rng_activation)

    correlation = np.array(biases.correlation, dtype=float)
    z = rng_values.multivariate_normal(np.zeros(n), correlation, method="cholesky")
    u = stats.norm.cdf(z)

    draws = []
    for i, param in enumerate(BIAS_PARAMS):
        spec = biases.params[param]
        is_active = bool(active[i])
        dist = spec.active if is_active else spec.neutral
        raw_value = float(dist.ppf(u[i]))

        if is_active:
            q = float(spec.active.cdf(raw_value))
            strength = q if spec.higher_is_stronger else 1.0 - q
        else:
            strength = 0.0

        # Two uniforms are always drawn so the stream stays aligned regardless
        # of the activation pattern.
        cluster_roll, magnitude_roll = rng_regime.random(2)
        multipliers = {regime: 1.0 for regime in Regime}
        if is_active and spec.cluster_regime is not None and cluster_roll < biases.p_regime_cluster:
            multipliers[spec.cluster_regime] = round(
                float(biases.regime_multiplier.ppf(magnitude_roll)), 4
            )

        draws.append(
            BiasDraw(
                param=param,
                value=round(raw_value, 4),
                active=is_active,
                strength=strength,
                multipliers=multipliers,
            )
        )

    return draws
