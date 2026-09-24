"""Estimators: each turns one PM's rows into one `Estimate` for one bias parameter."""

from pm_traitbench.gates.gate1.estimate import EstimatorSpec
from pm_traitbench.gates.gate1.estimators import (
    anchoring,
    conviction,
    disposition,
    exit_deficiency,
    extrapolation,
    herding,
    loss_aversion,
    overconfidence,
)

ESTIMATORS: dict[str, EstimatorSpec] = {
    "loss_aversion_lambda": loss_aversion.SPEC,
    "disposition_ratio": disposition.SPEC,
    "anchoring_rho": anchoring.SPEC,
    "extrapolation_theta": extrapolation.SPEC,
    "herding_weight": herding.SPEC,
    "overconfidence_coverage": overconfidence.SPEC,
    "conviction_size_miscalibration": conviction.SPEC,
    "exit_deficiency": exit_deficiency.SPEC,
}
