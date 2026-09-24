"""Overconfidence bias: a narrower stated coverage inflates the entry size factor."""

from pm_traitbench.engine.params import EffectiveParams


def size_factor(params: EffectiveParams) -> tuple[float, str | None]:
    """Size multiplier from the stated coverage's z-score against the 80% reference."""
    # Imported locally: own_signal imports engine.biases.extrapolation at module load,
    # so a module-level import here would form a circular import.
    from pm_traitbench.engine.own_signal import Z_80, z_for_coverage

    coverage = params.value("overconfidence_coverage")
    factor = Z_80 / z_for_coverage(coverage)
    active = params.is_active("overconfidence_coverage")
    flag = "overconfidence:oversized" if active and factor > 1 else None
    return factor, flag
