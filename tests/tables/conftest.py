"""Shared Gate 1 row fixtures for the table tests in this package."""

import pytest

from pm_traitbench.enums import AssetClass, Gate1Split, Gate1Test, Gate1Verdict, SeedGroupKind
from pm_traitbench.tables.schema import Gate1CellRow, Gate1PmRow


@pytest.fixture
def gate1_pm_row() -> Gate1PmRow:
    return Gate1PmRow(
        pm_id="pm_001",
        param="loss_aversion_lambda",
        split=Gate1Split.ALL,
        seed="A",
        asset_class=AssetClass.EQUITIES,
        statistic=None,
        n=12,
        planted=1.8,
        active=True,
        drifted=False,
    )


@pytest.fixture
def gate1_cell_row() -> Gate1CellRow:
    return Gate1CellRow(
        seed_group="pool",
        seed_group_kind=SeedGroupKind.SYNTHETIC_POOL,
        asset_class=AssetClass.EQUITIES,
        param="loss_aversion_lambda",
        split=Gate1Split.ALL,
        n_neutral=20,
        n_active=10,
        n_missing=1,
        neutral_mean=1.1,
        neutral_sd=None,
        active_mean=1.9,
        floor=None,
        active_share_past_floor=0.7,
        rank_corr=None,
        count_p10=8.0,
        calibration=None,
        test=Gate1Test.PER_PM,
        gap_ok=True,
        rank_ok=False,
        pop_z=None,
        pop_ok=False,
        count_ok=None,
        count_shortfall=False,
        verdict=Gate1Verdict.PASS,
        blocking=True,
    )


@pytest.fixture
def gate1_cell_row_pooled() -> Gate1CellRow:
    """A cross-class pooled row: `asset_class` null, pooling every direct asset class."""
    return Gate1CellRow(
        seed_group="synthetic",
        seed_group_kind=SeedGroupKind.SYNTHETIC_POOL,
        asset_class=None,
        param="loss_aversion_lambda",
        split=Gate1Split.ALL,
        n_neutral=36,
        n_active=18,
        n_missing=2,
        neutral_mean=1.1,
        neutral_sd=0.1,
        active_mean=1.9,
        floor=1.3,
        active_share_past_floor=0.8,
        rank_corr=0.6,
        count_p10=None,
        calibration=None,
        test=Gate1Test.PER_PM,
        gap_ok=True,
        rank_ok=True,
        pop_z=None,
        pop_ok=False,
        count_ok=None,
        count_shortfall=False,
        verdict=Gate1Verdict.PASS,
        blocking=True,
    )


@pytest.fixture
def gate1_cell_row_population() -> Gate1CellRow:
    """A population-tested cell, to round-trip `test`, `pop_z` and `pop_ok` at non-default
    values."""
    return Gate1CellRow(
        seed_group="pool",
        seed_group_kind=SeedGroupKind.SYNTHETIC_POOL,
        asset_class=AssetClass.EQUITIES,
        param="herding_weight",
        split=Gate1Split.ALL,
        n_neutral=20,
        n_active=10,
        n_missing=1,
        neutral_mean=0.2,
        neutral_sd=0.05,
        active_mean=0.6,
        floor=None,
        active_share_past_floor=0.9,
        rank_corr=0.3,
        count_p10=None,
        calibration=None,
        test=Gate1Test.POPULATION,
        gap_ok=False,
        rank_ok=False,
        pop_z=4.2,
        pop_ok=True,
        count_ok=None,
        count_shortfall=False,
        verdict=Gate1Verdict.PASS,
        blocking=True,
    )
