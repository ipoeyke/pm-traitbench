"""Bias modules for the behaviour engine, and the param-to-bias registry."""

from collections.abc import Sequence
from types import ModuleType

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.biases import (
    anchoring,
    conviction,
    disposition,
    exit_deficiency,
    extrapolation,
    herding,
    loss_aversion,
    overconfidence,
)
from pm_traitbench.engine.constants import BIAS_FLAG_ORDER

# Registers every bias module by the param that drives it.
BIAS_RULES: dict[str, ModuleType] = {
    "extrapolation_theta": extrapolation,
    "herding_weight": herding,
    "overconfidence_coverage": overconfidence,
    "conviction_size_miscalibration": conviction,
    "loss_aversion_lambda": loss_aversion,
    "disposition_ratio": disposition,
    "anchoring_rho": anchoring,
    "exit_deficiency": exit_deficiency,
}


def _check_registry(rules: dict[str, ModuleType], params: tuple[str, ...]) -> None:
    """Raise if the registry's keys do not exactly match the configured bias params."""
    if set(rules) != set(params):
        raise ValueError(f"BIAS_RULES keys {set(rules)} must equal BIAS_PARAMS {set(params)}")


_check_registry(BIAS_RULES, BIAS_PARAMS)


def join_flags(flags: Sequence[str | None]) -> str | None:
    """Join the non-None flags in BIAS_FLAG_ORDER order, or None if there are none."""
    present = [flag for flag in flags if flag is not None]
    if not present:
        return None
    order = {name: i for i, name in enumerate(BIAS_FLAG_ORDER)}
    unknown = len(BIAS_FLAG_ORDER)
    ranked = sorted(present, key=lambda flag: order.get(flag.split(":", 1)[0], unknown))
    return ";".join(ranked)
