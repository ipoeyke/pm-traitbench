"""Opportunity-count minimums: below these, an estimator's split is too thin to trust.

Each value is the ceiling of `16 p0 (1 - p0) / (p1 - p0)^2`, the observation
count at which a neutral PM's binomial standard error is a quarter of the gap
between the neutral centre `p0` and the active centre `p1`. Pairs: exit
deficiency 0.06 vs 0.44, loss aversion 0.10 vs 0.40, herding 0.17 vs 0.58,
anchoring 0.20 vs 0.50.
"""

N_MIN: dict[str, tuple[str, int]] = {
    "exit_deficiency": ("triggers_fired", 7),
    "loss_aversion_lambda": ("loss_side_untriggered_days", 16),
    "herding_weight": ("conflict_entries", 14),
    "anchoring_rho": ("exits", 29),
}
