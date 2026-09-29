"""Opportunity-count minimums: below these, an estimator's split is too thin to trust.

Each value is the ceiling of `16 p0 (1 - p0) / (p1 - p0)^2`, the observation
count at which a neutral PM's binomial standard error is a quarter of the gap
between the neutral centre `p0` and the active centre `p1`. Pairs: exit
deficiency 0.06 vs 0.44, loss aversion 0.01 vs 0.10 (the engine's add
hazard at lambda 1.1 and 2.0), herding 0.17 vs 0.58, anchoring 0.20 vs
0.50. Herding's pair was derived for the conflict-follow
rate; its minimum is applied to the agreement statistic's opportunity count
as an approximation, since the two counts differ. Anchoring's pair was
derived for the earlier band statistic and is kept as a rough guide for the
crossing-count statistic that replaced it.
"""

N_MIN: dict[str, int] = {
    "exit_deficiency": 7,
    "loss_aversion_lambda": 20,
    "herding_weight": 14,
    "anchoring_rho": 29,
}
