"""Tests for gate 1 verdict rules: pass, fail or insufficient per cell."""

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.enums import AssetClass, Gate1Split, Gate1Test, Gate1Verdict, SeedGroupKind
from pm_traitbench.gates.gate1._cell_stats import CellStats
from pm_traitbench.gates.gate1.verdict import blocking_failures, count_warnings, judge

KNOBS = Config().gate1
EQ = AssetClass.EQUITIES


def _cell(
    *,
    seed_group: str = "synthetic",
    seed_group_kind: SeedGroupKind = SeedGroupKind.SYNTHETIC_POOL,
    asset_class: AssetClass | None = None,  # None: the pooled cross-class row, blocking-eligible
    param: str = "loss_aversion_lambda",  # per-PM by default; not in population_params
    split: Gate1Split = Gate1Split.ALL,
    higher_is_stronger: bool = True,
    n_neutral: int = 20,
    n_active: int = 10,
    n_missing: int = 0,
    neutral_mean: float | None = 0.1,
    neutral_sd: float | None = 0.03,
    active_mean: float | None = 0.5,
    floor: float | None = None,
    active_share_past_floor: float | None = None,
    rank_corr: float | None = 1.0,
    active_rank_corr: float | None = 1.0,
    pop_z: float | None = None,
    count_p10: float | None = None,
    count_ok: bool | None = None,
    count_shortfall: bool = False,
    calibration: float | None = None,
) -> CellStats:
    return CellStats(
        seed_group=seed_group,
        seed_group_kind=seed_group_kind,
        asset_class=asset_class,
        param=param,
        split=split,
        higher_is_stronger=higher_is_stronger,
        n_neutral=n_neutral,
        n_active=n_active,
        n_missing=n_missing,
        neutral_mean=neutral_mean,
        neutral_sd=neutral_sd,
        active_mean=active_mean,
        floor=floor,
        active_share_past_floor=active_share_past_floor,
        rank_corr=rank_corr,
        active_rank_corr=active_rank_corr,
        pop_z=pop_z,
        count_p10=count_p10,
        count_ok=count_ok,
        count_shortfall=count_shortfall,
        calibration=calibration,
    )


# --- judge: gap_ok / verdict --------------------------------------------------


def test_clean_separation_passes_and_is_blocking_on_pooled_all_row() -> None:
    row = judge(_cell(), KNOBS)
    assert row.gap_ok is True
    assert row.rank_ok is True
    assert row.verdict == Gate1Verdict.PASS
    assert row.blocking is True


def test_active_mean_less_than_two_neutral_sds_from_neutral_mean_fails() -> None:
    # gap = 0.5 - 0.1 = 0.4 (default); shrink the gap so neutral_sd*2 > gap.
    cell = _cell(active_mean=0.14, neutral_sd=0.03)  # gap 0.04, sd 0.03: 0.03 > 0.5*0.04
    row = judge(cell, KNOBS)
    assert row.gap_ok is False
    assert row.verdict == Gate1Verdict.FAIL


def test_active_mean_wrong_side_of_neutral_mean_with_large_gap_fails_gap() -> None:
    cell = _cell(higher_is_stronger=True, neutral_mean=0.5, active_mean=0.1, neutral_sd=0.03)
    row = judge(cell, KNOBS)
    assert row.gap_ok is False
    assert row.verdict == Gate1Verdict.FAIL


def test_lower_is_stronger_passes_with_active_mean_below_neutral_mean() -> None:
    cell = _cell(higher_is_stronger=False, neutral_mean=0.5, active_mean=0.1, neutral_sd=0.03)
    row = judge(cell, KNOBS)
    assert row.gap_ok is True
    assert row.verdict == Gate1Verdict.PASS


def test_lower_is_stronger_wrong_side_of_neutral_mean_with_large_gap_fails_gap() -> None:
    cell = _cell(higher_is_stronger=False, neutral_mean=0.1, active_mean=0.5, neutral_sd=0.03)
    row = judge(cell, KNOBS)
    assert row.gap_ok is False
    assert row.verdict == Gate1Verdict.FAIL


def test_rank_corr_below_threshold_fails() -> None:
    row = judge(_cell(active_rank_corr=0.3), KNOBS)
    assert row.rank_ok is False
    assert row.verdict == Gate1Verdict.FAIL


def test_rank_corr_none_fails() -> None:
    row = judge(_cell(active_rank_corr=None), KNOBS)
    assert row.rank_ok is False
    assert row.verdict == Gate1Verdict.FAIL


def test_rank_ok_reads_active_rank_corr_not_the_combined_one() -> None:
    # Combined high, active low: bimodal values separate the groups without
    # ordering within active, so this must fail on the active-only measure.
    high_combined = judge(_cell(rank_corr=0.9, active_rank_corr=0.3), KNOBS)
    assert high_combined.rank_ok is False
    assert high_combined.verdict == Gate1Verdict.FAIL

    # Combined low, active high: the check passes on the active-only measure
    # even though the combined one (report-only) would not have passed it.
    low_combined = judge(_cell(rank_corr=0.3, active_rank_corr=0.9), KNOBS)
    assert low_combined.rank_ok is True
    assert low_combined.verdict == Gate1Verdict.PASS


def test_four_neutral_pms_is_insufficient_regardless_of_the_tests() -> None:
    row = judge(_cell(n_neutral=4), KNOBS)
    assert row.verdict == Gate1Verdict.INSUFFICIENT

    # even with a failing gap and rank corr, insufficient still wins.
    row = judge(_cell(n_neutral=4, active_rank_corr=None, active_mean=0.11), KNOBS)
    assert row.verdict == Gate1Verdict.INSUFFICIENT


def test_four_active_pms_is_insufficient() -> None:
    row = judge(_cell(n_active=4), KNOBS)
    assert row.verdict == Gate1Verdict.INSUFFICIENT


def test_threshold_boundaries_all_pass() -> None:
    # exactly representable floats: neutral_sd == gap_fraction * gap (0.5 == 0.5 * 1.0),
    # active_rank_corr == min_rank_corr (0.4 == 0.4), n_neutral == n_active == min_pms (5).
    assert KNOBS.gap_fraction == 0.5
    assert KNOBS.min_rank_corr == 0.4
    assert KNOBS.min_pms == 5
    cell = _cell(
        neutral_mean=0.0,
        neutral_sd=0.5,
        active_mean=1.0,
        active_rank_corr=0.4,
        n_neutral=5,
        n_active=5,
    )
    row = judge(cell, KNOBS)
    assert row.gap_ok is True
    assert row.rank_ok is True
    assert row.verdict == Gate1Verdict.PASS


# --- judge: population rule -----------------------------------------------------


def test_population_param_passes_on_pop_z_and_positive_rank_corr_despite_failing_gap() -> None:
    # gap 0.04, sd 0.03: gap_ok is False, but herding_weight is population-tested.
    cell = _cell(
        param="herding_weight", active_mean=0.14, neutral_sd=0.03, pop_z=3.5, active_rank_corr=0.2
    )
    row = judge(cell, KNOBS)
    assert row.test == Gate1Test.POPULATION
    assert row.gap_ok is False
    assert row.pop_ok is True
    assert row.verdict == Gate1Verdict.PASS


def test_population_param_fails_below_min_pop_z() -> None:
    cell = _cell(param="herding_weight", pop_z=2.5, active_rank_corr=0.2)
    row = judge(cell, KNOBS)
    assert row.pop_ok is False
    assert row.verdict == Gate1Verdict.FAIL


def test_population_rule_ignores_the_rank_correlation() -> None:
    # A population-tested parameter's per-PM estimates are too noisy to order,
    # so a negative or missing active rank correlation does not fail it.
    negative = judge(_cell(param="herding_weight", pop_z=3.5, active_rank_corr=-0.1), KNOBS)
    missing = judge(_cell(param="herding_weight", pop_z=3.5, active_rank_corr=None), KNOBS)
    assert negative.pop_ok is True and negative.verdict == Gate1Verdict.PASS
    assert missing.pop_ok is True and missing.verdict == Gate1Verdict.PASS


def test_per_pm_param_never_uses_the_population_rule() -> None:
    cell = _cell(param="loss_aversion_lambda", pop_z=10.0, active_rank_corr=0.9, active_mean=0.14)
    row = judge(cell, KNOBS)
    assert row.test == Gate1Test.PER_PM
    assert row.verdict == Gate1Verdict.FAIL  # gap_ok is False; pop_z is ignored


# --- judge: report-only never blocks --------------------------------------------


def test_report_only_param_pooled_all_row_never_blocks_whatever_its_verdict() -> None:
    passing = judge(_cell(param="disposition_ratio", pop_z=5.0, active_rank_corr=0.9), KNOBS)
    failing = judge(_cell(param="disposition_ratio", pop_z=1.0, active_rank_corr=0.9), KNOBS)
    assert passing.verdict == Gate1Verdict.PASS
    assert failing.verdict == Gate1Verdict.FAIL
    assert passing.blocking is False
    assert failing.blocking is False
    # disposition_ratio is report-only, so neither row names it as a failure.
    assert all("disposition_ratio" not in f for f in blocking_failures([passing, failing], KNOBS))


# --- judge: blocking -----------------------------------------------------------


def test_real_seed_row_is_not_blocking() -> None:
    row = judge(_cell(seed_group_kind=SeedGroupKind.REAL_SEED, seed_group="seed_r"), KNOBS)
    assert row.blocking is False


def test_single_synthetic_seed_row_is_not_blocking() -> None:
    row = judge(_cell(seed_group_kind=SeedGroupKind.SYNTHETIC_SEED, seed_group="seed_a"), KNOBS)
    assert row.blocking is False


def test_regime_split_on_the_pool_is_not_blocking() -> None:
    row = judge(_cell(split=Gate1Split.REGIME_RANGE), KNOBS)
    assert row.blocking is False


def test_per_class_pooled_all_row_never_blocks_even_when_the_cross_class_row_would() -> None:
    cross_class = judge(_cell(asset_class=None), KNOBS)
    per_class = judge(_cell(asset_class=EQ), KNOBS)
    assert cross_class.blocking is True
    assert per_class.blocking is False


# --- judge: field copy and count_shortfall ------------------------------------


def test_count_shortfall_true_with_passing_tests_stays_pass() -> None:
    cell = _cell(count_ok=False, count_shortfall=True)
    row = judge(cell, KNOBS)
    assert row.verdict == Gate1Verdict.PASS
    assert row.count_shortfall is True


def test_every_other_field_copies_from_stats() -> None:
    cell = _cell(
        n_missing=2,
        floor=0.2,
        active_share_past_floor=0.7,
        rank_corr=0.9,
        active_rank_corr=0.5,
        pop_z=1.7,
        count_p10=12.0,
        count_ok=True,
        calibration=0.4,
    )
    row = judge(cell, KNOBS)
    assert row.seed_group == cell.seed_group
    assert row.seed_group_kind == cell.seed_group_kind
    assert row.asset_class == cell.asset_class
    assert row.param == cell.param
    assert row.split == cell.split
    assert row.n_neutral == cell.n_neutral
    assert row.n_active == cell.n_active
    assert row.n_missing == cell.n_missing
    assert row.neutral_mean == cell.neutral_mean
    assert row.neutral_sd == cell.neutral_sd
    assert row.active_mean == cell.active_mean
    assert row.floor == cell.floor
    assert row.active_share_past_floor == cell.active_share_past_floor
    assert row.rank_corr == cell.rank_corr
    assert row.active_rank_corr == cell.active_rank_corr
    assert row.pop_z == cell.pop_z
    assert row.count_p10 == cell.count_p10
    assert row.count_ok == cell.count_ok
    assert row.count_shortfall == cell.count_shortfall
    assert row.calibration == cell.calibration


# --- blocking_failures / count_warnings ---------------------------------------


def test_blocking_failures_lists_only_blocking_non_pass_rows_sorted() -> None:
    rows = [
        # cross-class, blocking, fail
        judge(_cell(param="loss_aversion_lambda", active_rank_corr=0.3), KNOBS),
        judge(_cell(param="exit_deficiency"), KNOBS),  # cross-class, blocking, pass
        judge(_cell(param="extrapolation_theta"), KNOBS),  # cross-class, blocking, pass
        judge(_cell(param="overconfidence_coverage"), KNOBS),  # cross-class, blocking, pass
        judge(
            _cell(
                param="herding_weight",
                seed_group_kind=SeedGroupKind.SYNTHETIC_SEED,
                seed_group="seed_a",
                active_rank_corr=0.3,
            ),
            KNOBS,
        ),  # population, not blocking (single seed), fail
        judge(
            _cell(param="conviction_size_miscalibration", asset_class=None, pop_z=1.0),
            KNOBS,
        ),  # population, blocking, fail (pop_z below the minimum)
        judge(
            _cell(
                param="conviction_size_miscalibration",
                asset_class=AssetClass.COMMODITIES,
                pop_z=1.0,
            ),
            KNOBS,
        ),  # per-class pooled row, never blocking even though it fails
    ]
    # Every non-report-only param has a blocking row here, so nothing is missing.
    assert blocking_failures(rows, KNOBS) == [
        "all/conviction_size_miscalibration",
        "all/loss_aversion_lambda",
    ]


def test_blocking_failures_lists_missing_non_report_only_params_without_a_blocking_row() -> None:
    # Four of the five non-report-only params get a passing blocking row; the
    # fifth, conviction_size_miscalibration, has none and is reported missing.
    rows = [
        judge(_cell(param="loss_aversion_lambda"), KNOBS),
        judge(_cell(param="extrapolation_theta"), KNOBS),
        judge(_cell(param="overconfidence_coverage"), KNOBS),
        judge(_cell(param="exit_deficiency"), KNOBS),
    ]
    assert blocking_failures(rows, KNOBS) == ["all/conviction_size_miscalibration: missing"]


def test_blocking_failures_on_an_empty_row_list_lists_every_non_report_only_param_missing() -> None:
    # A report-only param (herding_weight, disposition_ratio, anchoring_rho)
    # never gets a missing entry, whatever rows exist for it.
    non_report_only = sorted(set(BIAS_PARAMS) - set(KNOBS.report_only_params))
    assert blocking_failures([], KNOBS) == [f"all/{param}: missing" for param in non_report_only]


def test_count_warnings_lists_only_single_seed_rows_with_count_ok_false_sorted() -> None:
    rows = [
        judge(
            _cell(
                seed_group_kind=SeedGroupKind.SYNTHETIC_SEED,
                seed_group="seed_b",
                asset_class=EQ,
                param="anchoring_rho",
                count_ok=False,
            ),
            KNOBS,
        ),
        judge(
            _cell(
                seed_group_kind=SeedGroupKind.REAL_SEED,
                seed_group="seed_r",
                asset_class=EQ,
                param="anchoring_rho",
                count_ok=False,
            ),
            KNOBS,
        ),
        judge(
            _cell(
                seed_group_kind=SeedGroupKind.SYNTHETIC_POOL, param="anchoring_rho", count_ok=False
            ),
            KNOBS,
        ),  # pooled row: never a count warning even if count_ok is False.
        judge(
            _cell(
                seed_group_kind=SeedGroupKind.SYNTHETIC_SEED,
                seed_group="seed_a",
                param="anchoring_rho",
                count_ok=True,
            ),
            KNOBS,
        ),
    ]
    assert count_warnings(rows) == [
        "seed_b/equities/anchoring_rho: count_shortfall",
        "seed_r/equities/anchoring_rho: count_shortfall",
    ]
