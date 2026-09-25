"""Cell-level statistics: neutral/active baselines, floor, rank correlation, pop_z and calibration.

Holds `PmEstimate`, `CellStats` and the per-cell helpers `aggregate.py` calls
once per group with that group's member rows; `verdict.py` also imports
`CellStats` to score a cell into a verdict.
"""

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import AssetClass, Gate1Split, SeedGroupKind
from pm_traitbench.gates.gate1.estimate import Estimate
from pm_traitbench.gates.gate1.n_min import N_MIN
from pm_traitbench.tables.schema import Gate1PmRow


@dataclass(frozen=True)
class PmEstimate:
    """One PM's recovered statistic for one bias parameter and split."""

    pm_id: str
    seed: str
    asset_class: AssetClass
    is_real_seed: bool
    param: str
    split: Gate1Split
    estimate: Estimate
    planted: float
    active: bool
    drifted: bool

    def to_row(self) -> Gate1PmRow:
        """This estimate as the row model gate 1 writes to the store."""
        return Gate1PmRow(
            pm_id=self.pm_id,
            param=self.param,
            split=self.split,
            seed=self.seed,
            asset_class=self.asset_class,
            statistic=self.estimate.value,
            n=self.estimate.n,
            planted=self.planted,
            active=self.active,
            drifted=self.drifted,
        )


@dataclass(frozen=True)
class CellStats:
    """A neutral-versus-active comparison for one seed group, param and split.

    `asset_class` is null for the synthetic pool's cross-class row, which
    pools every direct asset class together.
    """

    seed_group: str
    seed_group_kind: SeedGroupKind
    asset_class: AssetClass | None
    param: str
    split: Gate1Split
    higher_is_stronger: bool
    n_neutral: int
    n_active: int
    n_missing: int
    neutral_mean: float | None
    neutral_sd: float | None
    active_mean: float | None
    floor: float | None
    active_share_past_floor: float | None
    rank_corr: float | None
    pop_z: float | None
    count_p10: float | None
    count_ok: bool | None
    count_shortfall: bool
    calibration: float | None


def _mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) >= 2 else None
    return mean, sd


def _floor(
    neutral_mean: float | None, neutral_sd: float | None, higher_is_stronger: bool, floor_se: float
) -> float | None:
    if neutral_mean is None or neutral_sd is None:
        return None
    if higher_is_stronger:
        return neutral_mean + floor_se * neutral_sd
    return neutral_mean - floor_se * neutral_sd


def _active_share_past_floor(
    active_values: Sequence[float], floor: float | None, higher_is_stronger: bool
) -> float | None:
    if not active_values or floor is None:
        return None
    past = sum(1 for v in active_values if (v > floor if higher_is_stronger else v < floor))
    return past / len(active_values)


def _pop_z(
    neutral_mean: float | None,
    neutral_sd: float | None,
    n_neutral: int,
    active_mean: float | None,
    active_sd: float | None,
    n_active: int,
    higher_is_stronger: bool,
) -> float | None:
    """Standard-error-of-the-difference z between the active and neutral means.

    None when either set has fewer than 2 values (no sample sd) or the
    pooled standard error is 0.
    """
    if neutral_mean is None or neutral_sd is None or active_mean is None or active_sd is None:
        return None
    if n_neutral < 2 or n_active < 2:
        return None
    se = math.sqrt(neutral_sd**2 / n_neutral + active_sd**2 / n_active)
    if se == 0:
        return None
    direction = 1 if higher_is_stronger else -1
    return direction * (active_mean - neutral_mean) / se


def _rank_corr(planted: Sequence[float], values: Sequence[float]) -> float | None:
    if len(values) < 3 or len(set(planted)) == 1 or len(set(values)) == 1:
        return None
    rho = spearmanr(planted, values).statistic
    return None if math.isnan(rho) else float(rho)


def _calibration(param: str, pairs: Sequence[tuple[float, float]]) -> float | None:
    if param != "extrapolation_theta" or len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return None
    r = pearsonr(xs, ys).statistic
    return None if math.isnan(r) else float(r)


def build_cell(
    seed_group: str,
    kind: SeedGroupKind,
    asset_class: AssetClass | None,
    param: str,
    split: Gate1Split,
    higher_is_stronger: bool,
    members: Sequence[PmEstimate],
    knobs: Gate1Config,
    count_p10: float | None,
    count_ok: bool | None,
    count_shortfall: bool,
) -> CellStats:
    """Build a `CellStats` from a group's member `PmEstimate` rows."""
    n_missing = sum(1 for e in members if e.estimate.value is None)
    neutral = [e for e in members if not e.active and e.estimate.value is not None]
    active = [e for e in members if e.active and e.estimate.value is not None]
    neutral_values = [e.estimate.value for e in neutral]
    active_values = [e.estimate.value for e in active]

    neutral_mean, neutral_sd = _mean_sd(neutral_values)
    active_mean, active_sd = _mean_sd(active_values)
    floor = _floor(neutral_mean, neutral_sd, higher_is_stronger, knobs.floor_se)
    active_share_past_floor = _active_share_past_floor(active_values, floor, higher_is_stronger)
    pop_z = _pop_z(
        neutral_mean,
        neutral_sd,
        len(neutral_values),
        active_mean,
        active_sd,
        len(active_values),
        higher_is_stronger,
    )

    combined = neutral + active
    rank_corr = _rank_corr([e.planted for e in combined], [e.estimate.value for e in combined])
    pairs = tuple(pair for e in members for pair in e.estimate.pairs)
    calibration = _calibration(param, pairs)

    return CellStats(
        seed_group=seed_group,
        seed_group_kind=kind,
        asset_class=asset_class,
        param=param,
        split=split,
        higher_is_stronger=higher_is_stronger,
        n_neutral=len(neutral),
        n_active=len(active),
        n_missing=n_missing,
        neutral_mean=neutral_mean,
        neutral_sd=neutral_sd,
        active_mean=active_mean,
        floor=floor,
        active_share_past_floor=active_share_past_floor,
        rank_corr=rank_corr,
        pop_z=pop_z,
        count_p10=count_p10,
        count_ok=count_ok,
        count_shortfall=count_shortfall,
        calibration=calibration,
    )


def count_stats(
    param: str, split: Gate1Split, members: Sequence[PmEstimate]
) -> tuple[float | None, bool | None]:
    """`count_p10`/`count_ok` for a single-seed group's own `Estimate.n` values.

    The 10th percentile runs over every member's `Estimate.n`, including
    members whose value is None. `None, None` outside the `ALL` split, for a
    param with no `N_MIN` entry, or when `members` is empty.
    """
    if split != Gate1Split.ALL or param not in N_MIN or not members:
        return None, None
    n_min = N_MIN[param]
    p10 = float(np.percentile([e.estimate.n for e in members], 10))
    return p10, p10 >= n_min


def pool_shortfall(
    param: str, seeds: Collection[str], all_split_members: Sequence[PmEstimate]
) -> bool:
    """Whether any of `seeds`' `ALL`-split count for `param` fell short.

    `all_split_members` is the non-drifted `ALL`-split estimates for this
    `(asset_class, param)`; `seeds` is the synthetic seeds with a member among
    them (a seed whose every PM drifted has none, and is not considered).
    False when `param` has no `N_MIN` entry, since `count_ok` is never
    computed for it.
    """
    if param not in N_MIN:
        return False
    return any(
        count_stats(param, Gate1Split.ALL, [e for e in all_split_members if e.seed == seed])[1]
        is False
        for seed in seeds
    )
