"""Tests for gate 1's estimate run and its pooling into neutral/active cells."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from scipy.stats import pearsonr

from pm_traitbench.config import BIAS_PARAMS, Gate1Config
from pm_traitbench.enums import AssetClass, Gate1Split, SeedGroupKind
from pm_traitbench.gates.gate1.aggregate import PmEstimate, aggregate, estimate_all
from pm_traitbench.gates.gate1.estimate import Estimate
from pm_traitbench.gates.gate1.splits import splits_for

EQ = AssetClass.EQUITIES


def _with_drift(inputs, param: str, dates: tuple[date, ...]):
    return replace(inputs, drift_dates={**inputs.drift_dates, param: dates})


def _pm(
    pm_id: str,
    seed: str,
    param: str,
    split: Gate1Split,
    value: float | None,
    n: int,
    planted: float,
    active: bool,
    *,
    asset_class: AssetClass = EQ,
    is_real_seed: bool = False,
    drifted: bool = False,
    pairs: tuple[tuple[float, float], ...] = (),
) -> PmEstimate:
    return PmEstimate(
        pm_id=pm_id,
        seed=seed,
        asset_class=asset_class,
        is_real_seed=is_real_seed,
        param=param,
        split=split,
        estimate=Estimate(value=value, n=n, pairs=pairs),
        planted=planted,
        active=active,
        drifted=drifted,
    )


# --- estimate_all -----------------------------------------------------------


def test_estimate_all_emits_one_row_per_param_and_split_with_right_drifted_flags(
    make_inputs,
) -> None:
    inputs = _with_drift(make_inputs(), "loss_aversion_lambda", (date(2026, 1, 20),))
    rows = estimate_all([inputs], Gate1Config())

    expected_n_rows = sum(len(splits_for(inputs, p)) for p in BIAS_PARAMS)
    assert len(rows) == expected_n_rows

    drift_rows = [r for r in rows if r.param == "loss_aversion_lambda"]
    assert {r.split for r in drift_rows} == set(splits_for(inputs, "loss_aversion_lambda"))
    assert all(r.drifted for r in drift_rows)
    assert all(not r.drifted for r in rows if r.param != "loss_aversion_lambda")

    for row in rows:
        assert row.pm_id == inputs.pm_id
        assert row.seed == inputs.seed
        assert row.asset_class == inputs.asset_class
        assert row.is_real_seed == inputs.is_real_seed
        assert row.planted == inputs.traits[row.param].value
        assert row.active == inputs.traits[row.param].active


def test_pm_estimate_to_row_matches_its_fields() -> None:
    row = _pm("pm_001", "seed_a", "exit_deficiency", Gate1Split.ALL, 0.3, 7, 0.4, True).to_row()
    assert row.pm_id == "pm_001"
    assert row.seed == "seed_a"
    assert row.asset_class == EQ
    assert row.param == "exit_deficiency"
    assert row.split == Gate1Split.ALL
    assert row.statistic == 0.3
    assert row.n == 7
    assert row.planted == 0.4
    assert row.active is True
    assert row.drifted is False


# --- aggregate: grouping -----------------------------------------------------


def test_pooled_group_mixes_synthetic_seeds_and_real_seed_never_joins_it() -> None:
    param = "disposition_ratio"
    estimates = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.1, 10, 0.3, False),
        _pm("pm_002", "seed_b", param, Gate1Split.ALL, 0.2, 10, 0.3, False),
        _pm("pm_003", "seed_r", param, Gate1Split.ALL, 0.15, 10, 0.3, False, is_real_seed=True),
    ]
    cells = aggregate(estimates, {}, {"seed_a", "seed_b"}, {"seed_r"}, Gate1Config())

    pool = next(c for c in cells if c.seed_group_kind == SeedGroupKind.SYNTHETIC_POOL)
    assert pool.seed_group == "synthetic"
    assert pool.n_neutral == 2

    seed_a = next(c for c in cells if c.seed_group == "seed_a")
    seed_b = next(c for c in cells if c.seed_group == "seed_b")
    assert seed_a.n_neutral == 1
    assert seed_b.n_neutral == 1

    real = next(c for c in cells if c.seed_group_kind == SeedGroupKind.REAL_SEED)
    assert real.seed_group == "seed_r"


def test_drifted_pm_excluded_from_all_but_present_in_before() -> None:
    param = "disposition_ratio"
    estimates = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.1, 10, 0.3, False, drifted=True),
        _pm("pm_001", "seed_a", param, Gate1Split.BEFORE, 0.1, 10, 0.3, False, drifted=True),
        _pm("pm_002", "seed_a", param, Gate1Split.ALL, 0.2, 10, 0.3, False),
    ]
    cells = aggregate(estimates, {}, {"seed_a"}, set(), Gate1Config())

    all_cell = next(c for c in cells if c.split == Gate1Split.ALL)
    assert all_cell.n_neutral == 1  # only pm_002, pm_001 is drifted

    before_cell = next(c for c in cells if c.split == Gate1Split.BEFORE)
    assert before_cell.n_neutral == 1  # pm_001's before row is present


def test_no_member_group_is_not_emitted() -> None:
    estimates = [_pm("pm_001", "seed_a", "disposition_ratio", Gate1Split.ALL, 0.1, 10, 0.3, False)]
    cells = aggregate(estimates, {}, {"seed_a", "seed_b"}, {"seed_r"}, Gate1Config())
    assert not any(c.seed_group in ("seed_b", "seed_r") for c in cells)


# --- aggregate: neutral/active stats -----------------------------------------


def test_none_values_dropped_and_counted_in_n_missing() -> None:
    estimates = [
        _pm("pm_001", "seed_a", "disposition_ratio", Gate1Split.ALL, None, 0, 0.3, False),
        _pm("pm_002", "seed_a", "disposition_ratio", Gate1Split.ALL, 1.2, 10, 0.3, False),
    ]
    cells = aggregate(estimates, {}, {"seed_a"}, set(), Gate1Config())
    cell = next(c for c in cells if c.seed_group == "seed_a")
    assert cell.n_missing == 1
    assert cell.n_neutral == 1


def test_floor_and_active_share_for_a_lower_is_stronger_param() -> None:
    param = "overconfidence_coverage"  # higher_is_stronger is False
    neutral_values = [0.80, 0.82, 0.78, 0.80, 0.81]
    estimates = [
        _pm(f"pm_{i:03d}", "seed_a", param, Gate1Split.ALL, v, 10, 0.3, False)
        for i, v in enumerate(neutral_values, start=1)
    ] + [
        _pm("pm_010", "seed_a", param, Gate1Split.ALL, 0.50, 10, 0.9, True),  # below floor
        _pm("pm_011", "seed_a", param, Gate1Split.ALL, 0.95, 10, 0.9, True),  # above floor
    ]
    cells = aggregate(estimates, {}, {"seed_a"}, set(), Gate1Config())
    cell = next(c for c in cells if c.seed_group == "seed_a")

    assert cell.floor < cell.neutral_mean
    expected_floor = cell.neutral_mean - Gate1Config().floor_se * cell.neutral_sd
    assert cell.floor == pytest.approx(expected_floor)
    assert cell.active_share_past_floor == 0.5


def test_rank_corr_has_no_sign_adjustment_for_a_lower_is_stronger_param() -> None:
    param = "overconfidence_coverage"  # higher_is_stronger is False
    estimates = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.9, 10, 0.2, False),
        _pm("pm_002", "seed_a", param, Gate1Split.ALL, 0.6, 10, 0.5, True),
        _pm("pm_003", "seed_a", param, Gate1Split.ALL, 0.3, 10, 0.8, True),
    ]
    cells = aggregate(estimates, {}, {"seed_a"}, set(), Gate1Config())
    cell = next(c for c in cells if c.seed_group == "seed_a")
    # value falls as planted rises: raw (unflipped) spearman correlation is negative.
    assert cell.rank_corr == pytest.approx(-1.0)


def test_rank_corr_is_positive_when_coverage_moves_with_its_planted_value() -> None:
    param = "overconfidence_coverage"  # higher_is_stronger is False
    # a PM with lower planted overconfidence realises lower coverage too: the
    # statistic moves with the planted value, so no sign flip gives a positive rho.
    estimates = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.45, 10, 0.5, False),
        _pm("pm_002", "seed_a", param, Gate1Split.ALL, 0.65, 10, 0.7, True),
        _pm("pm_003", "seed_a", param, Gate1Split.ALL, 0.85, 10, 0.9, True),
    ]
    cells = aggregate(estimates, {}, {"seed_a"}, set(), Gate1Config())
    cell = next(c for c in cells if c.seed_group == "seed_a")
    assert cell.rank_corr == pytest.approx(1.0)


def test_rank_corr_none_below_three_pms_and_for_a_constant_statistic() -> None:
    param = "disposition_ratio"
    two_pm = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.9, 10, 0.2, False),
        _pm("pm_002", "seed_a", param, Gate1Split.ALL, 0.6, 10, 0.5, True),
    ]
    cells = aggregate(two_pm, {}, {"seed_a"}, set(), Gate1Config())
    assert next(c for c in cells if c.seed_group == "seed_a").rank_corr is None

    constant_stat = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.5, 10, 0.2, False),
        _pm("pm_002", "seed_a", param, Gate1Split.ALL, 0.5, 10, 0.5, True),
        _pm("pm_003", "seed_a", param, Gate1Split.ALL, 0.5, 10, 0.8, True),
    ]
    cells = aggregate(constant_stat, {}, {"seed_a"}, set(), Gate1Config())
    assert next(c for c in cells if c.seed_group == "seed_a").rank_corr is None


# --- aggregate: counts --------------------------------------------------------


def test_count_p10_matches_percentile_and_shortfall_propagates_to_pooled_row() -> None:
    param = "exit_deficiency"  # N_MIN: ("triggers_fired", 7)
    estimates = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.1, 10, 0.3, False),
        _pm("pm_002", "seed_a", param, Gate1Split.ALL, 0.2, 10, 0.3, False),
        _pm("pm_003", "seed_b", param, Gate1Split.ALL, 0.15, 10, 0.3, False),
        _pm("pm_004", "seed_b", param, Gate1Split.ALL, 0.25, 10, 0.3, False),
    ]
    engine_counts = {
        "pm_001": {"triggers_fired": 5},
        "pm_002": {"triggers_fired": 20},
        "pm_003": {"triggers_fired": 50},
        "pm_004": {"triggers_fired": 60},
    }
    cells = aggregate(estimates, engine_counts, {"seed_a", "seed_b"}, set(), Gate1Config())

    seed_a = next(c for c in cells if c.seed_group == "seed_a")
    seed_b = next(c for c in cells if c.seed_group == "seed_b")
    assert seed_a.count_p10 == pytest.approx(float(np.percentile([5, 20], 10)))
    assert seed_a.count_ok is False  # below n_min of 7
    assert seed_a.count_shortfall is True
    assert seed_b.count_ok is True
    assert seed_b.count_shortfall is False

    pool = next(c for c in cells if c.seed_group_kind == SeedGroupKind.SYNTHETIC_POOL)
    assert pool.count_p10 is None
    assert pool.count_ok is None
    assert pool.count_shortfall is True  # propagated from seed_a's shortfall


def test_count_ok_none_for_a_param_with_no_n_min() -> None:
    estimates = [_pm("pm_001", "seed_a", "disposition_ratio", Gate1Split.ALL, 1.0, 5, 0.3, False)]
    cells = aggregate(estimates, {"pm_001": {}}, {"seed_a"}, set(), Gate1Config())
    cell = cells[0]
    assert cell.count_p10 is None
    assert cell.count_ok is None
    assert cell.count_shortfall is False


def test_count_ok_none_for_a_non_all_split() -> None:
    param = "exit_deficiency"
    estimates = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.1, 10, 0.3, False, drifted=True),
        _pm("pm_001", "seed_a", param, Gate1Split.BEFORE, 0.1, 10, 0.3, False, drifted=True),
    ]
    engine_counts = {"pm_001": {"triggers_fired": 20}}
    cells = aggregate(estimates, engine_counts, {"seed_a"}, set(), Gate1Config())
    before_cell = next(c for c in cells if c.split == Gate1Split.BEFORE)
    assert before_cell.count_p10 is None
    assert before_cell.count_ok is None


def test_count_p10_none_when_no_pm_is_present_in_engine_counts() -> None:
    param = "exit_deficiency"
    estimates = [_pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.1, 10, 0.3, False)]
    cells = aggregate(estimates, {}, {"seed_a"}, set(), Gate1Config())
    cell = next(c for c in cells if c.seed_group == "seed_a")
    assert cell.count_p10 is None
    assert cell.count_ok is None
    assert cell.count_shortfall is False


def test_pool_shortfall_ignores_a_seed_whose_every_pm_drifted() -> None:
    param = "exit_deficiency"  # N_MIN: ("triggers_fired", 7)
    estimates = [
        # seed_a's only PM drifted, so it has no ALL-split row for this param.
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.1, 10, 0.3, False, drifted=True),
        _pm("pm_001", "seed_a", param, Gate1Split.BEFORE, 0.1, 10, 0.3, False, drifted=True),
        _pm("pm_002", "seed_b", param, Gate1Split.ALL, 0.2, 10, 0.3, False),
    ]
    engine_counts = {
        "pm_001": {"triggers_fired": 1},  # would fail n_min if seed_a were considered
        "pm_002": {"triggers_fired": 20},
    }
    cells = aggregate(estimates, engine_counts, {"seed_a", "seed_b"}, set(), Gate1Config())

    assert not any(c.seed_group == "seed_a" and c.split == Gate1Split.ALL for c in cells)

    pool = next(
        c
        for c in cells
        if c.seed_group_kind == SeedGroupKind.SYNTHETIC_POOL and c.split == Gate1Split.ALL
    )
    assert pool.count_shortfall is False


# --- aggregate: calibration ----------------------------------------------------


def test_calibration_matches_pearsonr_on_pooled_pairs_and_is_none_for_other_params() -> None:
    param = "extrapolation_theta"
    pairs_a = ((1.0, 0.5), (1.0, 0.6))
    pairs_b = ((-1.0, -0.4),)
    estimates = [
        _pm("pm_001", "seed_a", param, Gate1Split.ALL, 0.3, 2, 0.3, False, pairs=pairs_a),
        _pm("pm_002", "seed_a", param, Gate1Split.ALL, 0.4, 1, 0.3, False, pairs=pairs_b),
    ]
    cells = aggregate(estimates, {}, {"seed_a"}, set(), Gate1Config())
    cell = next(c for c in cells if c.seed_group == "seed_a")

    all_pairs = pairs_a + pairs_b
    xs = [p[0] for p in all_pairs]
    ys = [p[1] for p in all_pairs]
    assert cell.calibration == pytest.approx(pearsonr(xs, ys).statistic)

    other_param = "disposition_ratio"
    other = [
        _pm("pm_001", "seed_a", other_param, Gate1Split.ALL, 0.3, 2, 0.3, False, pairs=pairs_a),
    ]
    other_cells = aggregate(other, {}, {"seed_a"}, set(), Gate1Config())
    assert next(c for c in other_cells if c.seed_group == "seed_a").calibration is None
