"""Pool per-PM gate 1 estimates into neutral and active baselines per asset class.

`estimate_all` runs every estimator over every PM, bias parameter and split it
applies to. `aggregate` then pools those per-PM statistics into one cell per
`(seed group, asset class, parameter, split)`: a synthetic pool over every
synthetic seed, each synthetic seed alone, and each real seed alone. A cell
carries the neutral baseline, the active mean and floor, and the rank
correlation and calibration a verdict check later turns into a pass or fail.
"""

from collections import defaultdict
from collections.abc import Collection, Sequence

from pm_traitbench.config import BIAS_PARAMS, Gate1Config
from pm_traitbench.enums import AssetClass, Gate1Split, SeedGroupKind
from pm_traitbench.gates.gate1._cell_stats import (
    CellStats,
    PmEstimate,
    build_cell,
    count_stats,
    pool_shortfall,
)
from pm_traitbench.gates.gate1.estimators import ESTIMATORS
from pm_traitbench.gates.gate1.inputs import PmInputs
from pm_traitbench.gates.gate1.splits import days_for, splits_for

__all__ = ["SYNTHETIC_POOL", "PmEstimate", "estimate_all", "CellStats", "aggregate"]

SYNTHETIC_POOL = "synthetic"


def estimate_all(inputs: Sequence[PmInputs], knobs: Gate1Config) -> list[PmEstimate]:
    """Run every estimator over every PM, for each param's splits.

    `drifted` is whether the param's trait had a drift event; `planted` is the
    trait's sampled value, used as-is for every split. Split rows are
    report-only, so a split after a drift event still compares against the
    trait's sampled value rather than any drifted value.
    """
    results = []
    for pm in inputs:
        for param in BIAS_PARAMS:
            trait = pm.traits[param]
            drifted = bool(pm.drift_dates[param])
            for split in splits_for(pm, param):
                est = ESTIMATORS[param].estimate(pm, days_for(split, pm, param), knobs)
                results.append(
                    PmEstimate(
                        pm_id=pm.pm_id,
                        seed=pm.seed,
                        asset_class=pm.asset_class,
                        is_real_seed=pm.is_real_seed,
                        param=param,
                        split=split,
                        estimate=est,
                        planted=float(trait.value),
                        active=trait.active,
                        drifted=drifted,
                    )
                )
    return results


def aggregate(
    estimates: Sequence[PmEstimate],
    synthetic_seeds: Collection[str],
    real_seeds: Collection[str],
    knobs: Gate1Config,
) -> list[CellStats]:
    """Pool `estimates` into a synthetic-pool, per-synthetic-seed and per-real-seed cell.

    One cell per `(asset_class, param, split)` that has at least one member: for
    `ALL`, a drifted PM's estimate is excluded from every group; other splits
    already hold only the PMs that split applies to. A cell with no member is
    not emitted.
    """
    synthetic_seeds = set(synthetic_seeds)
    real_seeds = set(real_seeds)

    by_key: dict[tuple[AssetClass, str, Gate1Split], list[PmEstimate]] = defaultdict(list)
    for e in estimates:
        by_key[(e.asset_class, e.param, e.split)].append(e)

    shortfall_cache: dict[tuple[AssetClass, str], bool] = {}

    def shortfall_for(asset_class: AssetClass, param: str) -> bool:
        key = (asset_class, param)
        if key not in shortfall_cache:
            all_split_members = [
                e for e in by_key.get((asset_class, param, Gate1Split.ALL), []) if not e.drifted
            ]
            seeds = synthetic_seeds & {e.seed for e in all_split_members}
            shortfall_cache[key] = pool_shortfall(param, seeds, all_split_members)
        return shortfall_cache[key]

    cells: list[CellStats] = []
    for (asset_class, param, split), key_estimates in by_key.items():
        members = (
            [e for e in key_estimates if not e.drifted]
            if split == Gate1Split.ALL
            else key_estimates
        )
        if not members:
            continue
        higher_is_stronger = ESTIMATORS[param].higher_is_stronger

        pool_members = [e for e in members if e.seed in synthetic_seeds]
        if pool_members:
            cells.append(
                build_cell(
                    SYNTHETIC_POOL,
                    SeedGroupKind.SYNTHETIC_POOL,
                    asset_class,
                    param,
                    split,
                    higher_is_stronger,
                    pool_members,
                    knobs,
                    count_p10=None,
                    count_ok=None,
                    count_shortfall=shortfall_for(asset_class, param),
                )
            )

        for seeds, kind in (
            (synthetic_seeds, SeedGroupKind.SYNTHETIC_SEED),
            (real_seeds, SeedGroupKind.REAL_SEED),
        ):
            for seed in sorted(seeds):
                seed_members = [e for e in members if e.seed == seed]
                if not seed_members:
                    continue
                count_p10, count_ok = count_stats(param, split, seed_members)
                cells.append(
                    build_cell(
                        seed,
                        kind,
                        asset_class,
                        param,
                        split,
                        higher_is_stronger,
                        seed_members,
                        knobs,
                        count_p10=count_p10,
                        count_ok=count_ok,
                        count_shortfall=count_ok is False,
                    )
                )

    cells.sort(key=lambda c: (c.seed_group_kind, c.seed_group, c.asset_class, c.param, c.split))
    return cells
