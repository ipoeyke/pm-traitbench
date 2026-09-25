"""Tests for the behaviour engine's fixed constants."""

from pm_traitbench.engine.constants import (
    CONVICTION_CUTS,
    DV01_PER_MILLION,
    ENTRY_THRESHOLD,
    LOSS_ADD_CAP,
    LOSS_ADD_SLOPE,
    LOSS_CUT_HAZARD,
    RISK_STEPS,
    SIZE_HEADROOM,
)
from pm_traitbench.enums import SOVEREIGN_TENORS
from pm_traitbench.market.constants import COMMODITIES, CONTRACT_MULTIPLIER


def test_contract_multiplier_covers_every_commodity_code() -> None:
    codes = {spec.code for specs in COMMODITIES.values() for spec in specs}
    assert set(CONTRACT_MULTIPLIER) == codes
    assert all(value > 0 for value in CONTRACT_MULTIPLIER.values())


def test_dv01_per_million_covers_sovereign_tenors() -> None:
    assert set(DV01_PER_MILLION) == set(SOVEREIGN_TENORS)


def test_conviction_cuts_ascending_starting_at_entry_threshold() -> None:
    assert CONVICTION_CUTS[0] == ENTRY_THRESHOLD
    assert list(CONVICTION_CUTS) == sorted(CONVICTION_CUTS)
    assert len(set(CONVICTION_CUTS)) == len(CONVICTION_CUTS)


def test_risk_steps_ascending_within_unit_interval() -> None:
    assert list(RISK_STEPS) == sorted(RISK_STEPS)
    assert all(0 < step <= 1 for step in RISK_STEPS)


def test_loss_aversion_hazard_constants() -> None:
    assert LOSS_CUT_HAZARD == 0.05
    assert LOSS_ADD_SLOPE == 0.1
    assert LOSS_ADD_CAP == 0.5


def test_size_headroom_constant() -> None:
    assert SIZE_HEADROOM == 2.5
