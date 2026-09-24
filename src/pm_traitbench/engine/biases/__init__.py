"""Bias modules for the behaviour engine, and the param-to-bias registry."""

from collections.abc import Sequence
from types import ModuleType

from pm_traitbench.engine.biases import conviction, extrapolation, herding, overconfidence
from pm_traitbench.engine.constants import BIAS_FLAG_ORDER

# Registers the entry-side bias modules by the param that drives them; the
# position-side modules are added separately, along with a check that the
# keys equal BIAS_PARAMS.
BIAS_RULES: dict[str, ModuleType] = {
    "extrapolation_theta": extrapolation,
    "herding_weight": herding,
    "overconfidence_coverage": overconfidence,
    "conviction_size_miscalibration": conviction,
}


def join_flags(flags: Sequence[str | None]) -> str | None:
    """Join the non-None flags in BIAS_FLAG_ORDER order, or None if there are none."""
    present = [flag for flag in flags if flag is not None]
    if not present:
        return None
    order = {name: i for i, name in enumerate(BIAS_FLAG_ORDER)}
    unknown = len(BIAS_FLAG_ORDER)
    ranked = sorted(present, key=lambda flag: order.get(flag.split(":", 1)[0], unknown))
    return ";".join(ranked)
