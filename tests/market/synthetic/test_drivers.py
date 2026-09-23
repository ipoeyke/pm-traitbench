"""Tests for market driver shocks: drawing, combining and vol scaling."""

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import CommodityGroup, Regime
from pm_traitbench.market.regimes import constant_path
from pm_traitbench.market.synthetic.drivers import daily_vol, draw_shocks, seed_driver


def test_draw_shocks_is_deterministic_for_the_same_root_seed() -> None:
    a = draw_shocks(1, 20)
    b = draw_shocks(1, 20)
    assert np.array_equal(a.common, b.common)
    for group in CommodityGroup:
        assert np.array_equal(a.groups[group], b.groups[group])


def test_draw_shocks_differs_for_a_different_root_seed() -> None:
    a = draw_shocks(1, 20)
    b = draw_shocks(2, 20)
    assert not np.array_equal(a.common, b.common)


def test_draw_shocks_draws_every_commodity_group_at_the_requested_length() -> None:
    shocks = draw_shocks(1, 20)
    assert set(shocks.groups) == set(CommodityGroup)
    assert shocks.common.shape == (20,)
    for vector in shocks.groups.values():
        assert vector.shape == (20,)


def test_group_vectors_are_pairwise_different_and_differ_from_common() -> None:
    shocks = draw_shocks(1, 50)
    vectors = [shocks.common, *shocks.groups.values()]
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            assert not np.array_equal(vectors[i], vectors[j])


def test_seed_driver_adds_driver_mean_and_macro_exactly() -> None:
    config = Config()
    path = constant_path(Regime.RISK_ON, 5, config)
    shocks = draw_shocks(1, 5)
    macro = np.array([0.1, -0.2, 0.0, 0.05, -0.05])
    z = seed_driver(shocks, path, macro)
    assert np.array_equal(z, shocks.common + path.driver_mean + macro)


def test_daily_vol_scales_annual_vol_by_the_regime_vol_multiplier() -> None:
    config = Config()
    path = constant_path(Regime.RISK_OFF, 4, config)
    vol = daily_vol(0.2, path)
    assert np.allclose(vol, 0.2 * path.vol_multiplier / np.sqrt(252))
