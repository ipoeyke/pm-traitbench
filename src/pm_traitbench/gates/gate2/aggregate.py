"""Pools gate 2's per-PM rows into cells: exact tests, verdicts and blocking checks."""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import hypergeom

from pm_traitbench.config import BIAS_PARAMS, Gate2Config
from pm_traitbench.enums import Gate2Slice, Gate2Verdict, Kind, Ownership, SignalMode, Valence
from pm_traitbench.tables.schema import Gate2CellRow, Gate2PmRow, Gate2SignalRow, Gate2TraitRow

POOLED_PREFERENCES_ID = "all/preferences"


def fisher_upper_p(a: int, n_active: int, n_inactive: int, present_total: int) -> float:
    """One-sided probability of at least `a` of `present_total` predicted-present PMs being
    active, under random assignment from `n_active` active and `n_inactive` inactive PMs.
    """
    return float(hypergeom.sf(a - 1, n_active + n_inactive, n_active, present_total))


def poisson_binomial_upper_p(chances: Sequence[float], hits: int) -> float:
    """P(X >= hits) for X a sum of independent Bernoulli(chance_i), by dynamic programme
    over the count distribution.
    """
    dist = np.zeros(len(chances) + 1)
    dist[0] = 1.0
    for p in chances:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    # Float accumulation can push the hits=0 tail a hair past 1; clamp to a valid probability.
    return min(1.0, max(0.0, float(dist[hits:].sum())))


def blocking_id(row: Gate2CellRow) -> str:
    """`all/{param}` for a bias row, `all/preferences` for the null-param pooled row.

    Only meaningful for `all`-slice rows; the caller is expected to filter to those.
    """
    if row.param is None:
        return POOLED_PREFERENCES_ID
    return f"{row.slice_value}/{row.param}"


def blocking_failures(rows: Sequence[Gate2CellRow]) -> list[str]:
    """Sorted ids of blocking rows whose verdict is fail. Insufficient rows never block:
    a class below `min_class` says nothing about narration recovery either way.
    """
    failing = (row for row in rows if row.blocking and row.verdict == Gate2Verdict.FAIL)
    return sorted(blocking_id(row) for row in failing)


def insufficient_blocking(rows: Sequence[Gate2CellRow]) -> list[str]:
    """Sorted ids of blocking rows whose verdict is insufficient."""
    insufficient = (
        row for row in rows if row.blocking and row.verdict == Gate2Verdict.INSUFFICIENT
    )
    return sorted(blocking_id(row) for row in insufficient)


def _confusion(actives: Sequence[bool], predicteds: Sequence[bool]) -> tuple[int, int, int, int]:
    """`(n_active, n_inactive, tp, fp)` for two aligned boolean sequences."""
    n_active = sum(actives)
    n_inactive = len(actives) - n_active
    tp = sum(a and p for a, p in zip(actives, predicteds, strict=True))
    fp = sum((not a) and p for a, p in zip(actives, predicteds, strict=True))
    return n_active, n_inactive, tp, fp


def _confusion_cell(
    slice_: Gate2Slice,
    slice_value: str,
    param: str | None,
    actives: Sequence[bool],
    predicteds: Sequence[bool],
    config: Gate2Config,
    blocking: bool,
) -> Gate2CellRow:
    """A cell from a 2x2 active-versus-predicted-active table, tested by Fisher's exact test."""
    n_active, n_inactive, tp, fp = _confusion(actives, predicteds)
    n = n_active + n_inactive
    tn = n_inactive - fp
    n_positive = tp + tn
    rate = None if n_active == 0 or n_inactive == 0 else (tp / n_active + tn / n_inactive) / 2
    if min(n_active, n_inactive) < config.min_class:
        p, verdict = None, Gate2Verdict.INSUFFICIENT
    else:
        p = fisher_upper_p(tp, n_active, n_inactive, tp + fp)
        verdict = Gate2Verdict.PASS if p < config.alpha else Gate2Verdict.FAIL
    return Gate2CellRow(
        slice=slice_,
        slice_value=slice_value,
        param=param,
        n=n,
        n_positive=n_positive,
        rate=rate,
        chance=0.5,
        p=p,
        verdict=verdict,
        blocking=blocking,
    )


def _pooled_preference_cell(
    slice_: Gate2Slice,
    slice_value: str,
    param: str | None,
    held_rows: Sequence[Gate2TraitRow],
    k_by_param: Mapping[str, int],
    config: Gate2Config,
    blocking: bool,
) -> Gate2CellRow:
    """A cell over held preference pairs, tested by the pooled Poisson-binomial tail."""
    n = len(held_rows)
    n_positive = sum(row.correct for row in held_rows)
    rate = n_positive / n if n else None
    chances = [1 / k_by_param[row.param] for row in held_rows]
    chance = sum(chances) / n if n else None
    if n < config.min_class:
        p, verdict = None, Gate2Verdict.INSUFFICIENT
    else:
        p = poisson_binomial_upper_p(chances, n_positive)
        verdict = Gate2Verdict.PASS if p < config.alpha else Gate2Verdict.FAIL
    return Gate2CellRow(
        slice=slice_,
        slice_value=slice_value,
        param=param,
        n=n,
        n_positive=n_positive,
        rate=rate,
        chance=chance,
        p=p,
        verdict=verdict,
        blocking=blocking,
    )


def _share_cell(
    slice_: Gate2Slice, slice_value: str, param: str | None, n: int, n_positive: int
) -> Gate2CellRow:
    """A report-only cell: a bare recovery rate, no test, never blocking."""
    rate = n_positive / n if n else None
    return Gate2CellRow(
        slice=slice_,
        slice_value=slice_value,
        param=param,
        n=n,
        n_positive=n_positive,
        rate=rate,
        chance=None,
        p=None,
        verdict=None,
        blocking=False,
    )


def _bias_param_rows(
    bias_rows: Sequence[Gate2TraitRow],
    slice_: Gate2Slice,
    slice_value: str,
    config: Gate2Config,
    blocking: bool,
) -> list[Gate2CellRow]:
    """One `_confusion_cell` per bias parameter, over `bias_rows` restricted to that param."""
    rows = []
    for param in BIAS_PARAMS:
        param_rows = [row for row in bias_rows if row.param == param]
        actives = [row.truth_active for row in param_rows]
        predicteds = [row.predicted_active for row in param_rows]
        cell = _confusion_cell(slice_, slice_value, param, actives, predicteds, config, blocking)
        rows.append(cell)
    return rows


def _mode_rows(signal_rows: Sequence[Gate2SignalRow]) -> list[Gate2CellRow]:
    """Report-only recovery rates by signal mode, and by mode and param, for self-confirmed
    signals not predating a drift update.
    """
    eligible = [
        signal
        for signal in signal_rows
        if signal.ownership == Ownership.SELF
        and signal.valence == Valence.CONFIRM
        and not signal.pre_update
    ]
    rows: list[Gate2CellRow] = []
    for mode in SignalMode:
        mode_signals = [signal for signal in eligible if signal.mode == mode]
        rows.append(
            _share_cell(
                Gate2Slice.MODE,
                mode.value,
                None,
                len(mode_signals),
                sum(signal.recovered for signal in mode_signals),
            )
        )
        for param in sorted({signal.param for signal in mode_signals}):
            param_signals = [signal for signal in mode_signals if signal.param == param]
            rows.append(
                _share_cell(
                    Gate2Slice.MODE,
                    mode.value,
                    param,
                    len(param_signals),
                    sum(signal.recovered for signal in param_signals),
                )
            )
    return rows


def _pm_slice_rows(
    slice_: Gate2Slice,
    cell_value: Mapping[str, str],
    trait_rows: Sequence[Gate2TraitRow],
    k_by_param: Mapping[str, int],
    config: Gate2Config,
) -> list[Gate2CellRow]:
    """The all-bias rows and the pooled preference row, restricted to each cell's own PMs."""
    rows: list[Gate2CellRow] = []
    for value in sorted(set(cell_value.values())):
        pm_ids = {pm_id for pm_id, pm_value in cell_value.items() if pm_value == value}
        subset = [row for row in trait_rows if row.pm_id in pm_ids]
        bias_subset = [row for row in subset if row.kind == Kind.BIAS]
        rows.extend(_bias_param_rows(bias_subset, slice_, value, config, blocking=False))
        held = [
            row for row in subset if row.kind == Kind.PREFERENCE and row.truth_value is not None
        ]
        pooled = _pooled_preference_cell(
            slice_, value, None, held, k_by_param, config, blocking=False
        )
        rows.append(pooled)
    return rows


def _sort_key(row: Gate2CellRow) -> tuple:
    """`(slice, slice_value, param)` with a null param sorted first."""
    return (row.slice.value, row.slice_value, row.param is not None, row.param or "")


def build_cells(
    trait_rows: Sequence[Gate2TraitRow],
    signal_rows: Sequence[Gate2SignalRow],
    pm_rows: Sequence[Gate2PmRow],
    k_by_param: Mapping[str, int],
    config: Gate2Config,
) -> list[Gate2CellRow]:
    """Pool `trait_rows`, `signal_rows` and `pm_rows` into the gate's full cell table."""
    bias_rows = [row for row in trait_rows if row.kind == Kind.BIAS]
    pref_rows = [row for row in trait_rows if row.kind == Kind.PREFERENCE]
    held_rows = [row for row in pref_rows if row.truth_value is not None]
    pref_params = sorted({row.param for row in pref_rows})

    rows: list[Gate2CellRow] = []
    rows.extend(_bias_param_rows(bias_rows, Gate2Slice.ALL, "all", config, blocking=True))
    pooled = _pooled_preference_cell(
        Gate2Slice.ALL, "all", None, held_rows, k_by_param, config, True
    )
    rows.append(pooled)
    for param in pref_params:
        param_held = [row for row in held_rows if row.param == param]
        param_pooled = _pooled_preference_cell(
            Gate2Slice.ALL, "all", param, param_held, k_by_param, config, False
        )
        rows.append(param_pooled)
        param_rows = [row for row in pref_rows if row.param == param]
        actives = [row.truth_value is not None for row in param_rows]
        predicteds = [row.predicted_value is not None for row in param_rows]
        held_cell = _confusion_cell(
            Gate2Slice.HELD, "all", param, actives, predicteds, config, False
        )
        rows.append(held_cell)

    bias_correct = sum(row.correct for row in bias_rows)
    held_correct = sum(row.correct for row in held_rows)
    rows.append(_share_cell(Gate2Slice.KIND, "bias", None, len(bias_rows), bias_correct))
    rows.append(_share_cell(Gate2Slice.KIND, "preference", None, len(held_rows), held_correct))

    rows.extend(_mode_rows(signal_rows))

    for slice_, cell_value in (
        (Gate2Slice.ASSET_CLASS, {pm.pm_id: pm.asset_class.value for pm in pm_rows}),
        (Gate2Slice.TYPICALITY, {pm.pm_id: pm.typicality.value for pm in pm_rows}),
        (Gate2Slice.DRIFT, {pm.pm_id: pm.drift.value for pm in pm_rows}),
    ):
        rows.extend(_pm_slice_rows(slice_, cell_value, trait_rows, k_by_param, config))

    return sorted(rows, key=_sort_key)
