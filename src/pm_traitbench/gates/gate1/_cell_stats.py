"""Cell-level statistics: neutral/active baselines, floor, rank correlation and calibration.

Private to `aggregate.py`, which owns the grouping (seed pool, per-seed, per
split) and calls `build_cell` once per group with that group's member rows.
Kept separate so `aggregate.py` stays focused on the grouping itself.
"""

import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import AssetClass, Gate1Split, SeedGroupKind
from pm_traitbench.gates.gate1.n_min import N_MIN


@dataclass(frozen=True)
class CellStats:
    """A neutral-versus-active comparison for one seed group, param and split."""

    seed_group: str
    seed_group_kind: SeedGroupKind
    asset_class: AssetClass
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
    asset_class: AssetClass,
    param: str,
    split: Gate1Split,
    higher_is_stronger: bool,
    members: Sequence,
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
    active_mean, _ = _mean_sd(active_values)
    floor = _floor(neutral_mean, neutral_sd, higher_is_stronger, knobs.floor_se)
    active_share_past_floor = _active_share_past_floor(active_values, floor, higher_is_stronger)

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
        count_p10=count_p10,
        count_ok=count_ok,
        count_shortfall=count_shortfall,
        calibration=calibration,
    )


def pm_ids_by_seed_asset(estimates: Sequence) -> dict[tuple[str, AssetClass], set[str]]:
    """Unique PM ids per `(seed, asset_class)`, over every estimate regardless of param or split."""
    by_seed_asset: dict[tuple[str, AssetClass], set[str]] = defaultdict(set)
    for e in estimates:
        by_seed_asset[(e.seed, e.asset_class)].add(e.pm_id)
    return by_seed_asset


def seed_counts(
    seed: str,
    asset_class: AssetClass,
    param: str,
    split: Gate1Split,
    pm_ids: Mapping[tuple[str, AssetClass], set[str]],
    engine_counts: Mapping[str, Mapping[str, int]],
) -> tuple[float | None, bool | None]:
    """`count_p10`/`count_ok` for `seed`'s PM set at `asset_class`.

    `None, None` outside `ALL` split or for a param with no `N_MIN` entry.
    """
    if split != Gate1Split.ALL or param not in N_MIN:
        return None, None
    count_name, n_min = N_MIN[param]
    seed_pms = sorted(pm_ids.get((seed, asset_class), set()))
    counts = [engine_counts[pm][count_name] for pm in seed_pms]
    p10 = float(np.percentile(counts, 10))
    return p10, p10 >= n_min


def pool_shortfall(
    asset_class: AssetClass,
    param: str,
    synthetic_seeds: Collection[str],
    pm_ids: Mapping[tuple[str, AssetClass], set[str]],
    engine_counts: Mapping[str, Mapping[str, int]],
) -> bool:
    """Whether any synthetic seed's `ALL`-split count for `(asset_class, param)` fell short.

    False when `param` has no `N_MIN` entry, since `count_ok` is never computed
    for it.
    """
    if param not in N_MIN:
        return False
    return any(
        seed_counts(seed, asset_class, param, Gate1Split.ALL, pm_ids, engine_counts)[1] is False
        for seed in synthetic_seeds
        if (seed, asset_class) in pm_ids
    )
