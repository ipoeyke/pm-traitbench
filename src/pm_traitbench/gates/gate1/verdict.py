"""Turn a cell's neutral-versus-active statistics into a pass, fail or insufficient verdict.

`judge` picks the per-PM or population rule by the cell's parameter, checks it
against the cell's `Gate1Config` thresholds, and marks a cell as gate-blocking
only when it is the synthetic pool's cross-class comparison over the full run
(`asset_class` null) for a parameter that is not report-only; a per-asset-class
pooled row is judged the same way but never blocks. `blocking_failures` and
`count_warnings` then summarise a whole run's rows for reporting.
"""

from collections.abc import Sequence

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import Gate1Split, Gate1Test, Gate1Verdict, SeedGroupKind
from pm_traitbench.gates.gate1._cell_stats import CellStats
from pm_traitbench.tables.schema import Gate1CellRow


def judge(stats: CellStats, knobs: Gate1Config) -> Gate1CellRow:
    """Score one cell into a `Gate1CellRow` with its gap, rank, population and verdict checks."""
    direction = 1 if stats.higher_is_stronger else -1
    gap_ok = (
        stats.neutral_sd is not None
        and stats.neutral_mean is not None
        and stats.active_mean is not None
        and (stats.active_mean - stats.neutral_mean) * direction > 0
        and stats.neutral_sd <= knobs.gap_fraction * abs(stats.active_mean - stats.neutral_mean)
    )
    rank_ok = stats.rank_corr is not None and stats.rank_corr >= knobs.min_rank_corr
    pop_ok = (
        stats.pop_z is not None
        and stats.pop_z >= knobs.min_pop_z
        and stats.rank_corr is not None
        and stats.rank_corr > 0
    )

    test = Gate1Test.POPULATION if stats.param in knobs.population_params else Gate1Test.PER_PM

    if stats.n_neutral < knobs.min_pms or stats.n_active < knobs.min_pms:
        verdict = Gate1Verdict.INSUFFICIENT
    elif test == Gate1Test.POPULATION:
        verdict = Gate1Verdict.PASS if pop_ok else Gate1Verdict.FAIL
    elif gap_ok and rank_ok:
        verdict = Gate1Verdict.PASS
    else:
        verdict = Gate1Verdict.FAIL

    blocking = (
        stats.seed_group_kind == SeedGroupKind.SYNTHETIC_POOL
        and stats.split == Gate1Split.ALL
        and stats.asset_class is None
        and stats.param not in knobs.report_only_params
    )

    return Gate1CellRow(
        seed_group=stats.seed_group,
        seed_group_kind=stats.seed_group_kind,
        asset_class=stats.asset_class,
        param=stats.param,
        split=stats.split,
        n_neutral=stats.n_neutral,
        n_active=stats.n_active,
        n_missing=stats.n_missing,
        neutral_mean=stats.neutral_mean,
        neutral_sd=stats.neutral_sd,
        active_mean=stats.active_mean,
        floor=stats.floor,
        active_share_past_floor=stats.active_share_past_floor,
        rank_corr=stats.rank_corr,
        count_p10=stats.count_p10,
        calibration=stats.calibration,
        test=test,
        gap_ok=gap_ok,
        rank_ok=rank_ok,
        pop_z=stats.pop_z,
        pop_ok=pop_ok,
        count_ok=stats.count_ok,
        count_shortfall=stats.count_shortfall,
        verdict=verdict,
        blocking=blocking,
    )


def blocking_failures(rows: Sequence[Gate1CellRow]) -> list[str]:
    """`all/param` for every blocking (cross-class) row whose verdict is not pass, sorted."""
    return sorted(
        f"all/{row.param}" for row in rows if row.blocking and row.verdict != Gate1Verdict.PASS
    )


def count_warnings(rows: Sequence[Gate1CellRow]) -> list[str]:
    """`seed_group/asset_class/param: count_shortfall` for single-seed rows short on counts."""
    return sorted(
        f"{row.seed_group}/{row.asset_class}/{row.param}: count_shortfall"
        for row in rows
        if row.seed_group_kind != SeedGroupKind.SYNTHETIC_POOL and row.count_ok is False
    )
