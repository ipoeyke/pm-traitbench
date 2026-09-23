"""Tests for the synthetic market's realised-moment check: check_market end to
end, round-level crossing counts, and model-implied moments.
"""

import numpy as np
import pytest

from pm_traitbench.config import Config, RegimeParams
from pm_traitbench.enums import Family, InstrumentKind, Regime
from pm_traitbench.errors import MarketCheckError
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.synthetic.build import build_seed
from pm_traitbench.market.synthetic.check import (
    check_market,
    implied_moments,
    round_level_test_count,
)
from pm_traitbench.market.synthetic.drivers import draw_shocks
from pm_traitbench.market.synthetic.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import Instrument


def _rigged_config(config: Config, **regime_overrides: dict) -> Config:
    regimes = config.market.regimes.model_copy(update=regime_overrides)
    market = config.market.model_copy(update={"regimes": regimes})
    return config.model_copy(update={"market": market})


def _config_with_rates_driver_corr(config: Config, driver_corr: float) -> Config:
    rates = config.market.families.rates.model_copy(update={"driver_corr": driver_corr})
    families = config.market.families.model_copy(update={"rates": rates})
    market = config.market.model_copy(update={"families": families})
    return config.model_copy(update={"market": market})


@pytest.mark.parametrize("seed", ["A", "B", "C"])
def test_check_market_passes_at_default_config(universe, markets, seed: str) -> None:
    config, instruments, _, _ = universe
    report = check_market(markets[seed], instruments, config)
    failed = [m for m in report.metrics if not m.passed]
    assert not failed, failed
    assert all(m.passed for m in report.metrics)


@pytest.mark.parametrize("root_seed", [25, 30])
def test_check_market_passes_at_default_config_for_late_roots(root_seed: int) -> None:
    # The rates check must use each curve's own level vol, not one shared vol.
    config = Config.model_validate({"seed": {"root": root_seed}})
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = tuple(build_universe(config, stream(config.seed.root, "market", "universe")))
    shocks = draw_shocks(config.seed.root, axis.n_days)
    market = build_seed(config, "C", instruments, shocks, axis)
    report = check_market(market, instruments, config)
    assert all(m.passed for m in report.metrics)


@pytest.mark.parametrize(("root_seed", "seed"), [(61, "A"), (61, "B"), (61, "C"), (96, "A")])
def test_check_market_passes_at_default_config_for_fat_tailed_corr_roots(
    root_seed: int, seed: str
) -> None:
    # The correlation band must widen near +-1 (Fisher z), not stay a fixed
    # normal-theory band that fat-tailed noise can breach at these roots.
    config = Config.model_validate({"seed": {"root": root_seed}})
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = tuple(build_universe(config, stream(config.seed.root, "market", "universe")))
    shocks = draw_shocks(config.seed.root, axis.n_days)
    market = build_seed(config, seed, instruments, shocks, axis)
    report = check_market(market, instruments, config)
    assert all(m.passed for m in report.metrics)


def test_rigged_vol_multiplier_raises_naming_the_regime_and_metric(universe, markets) -> None:
    config, instruments, _, _ = universe
    rigged = _rigged_config(
        config, vol_multiplier={**config.market.regimes.vol_multiplier, Regime.RISK_OFF: 3.0}
    )
    with pytest.raises(MarketCheckError) as excinfo:
        check_market(markets["A"], instruments, rigged)
    message = str(excinfo.value)
    assert "seed A" in message
    assert "regime risk_off" in message
    assert "metric vol" in message


def test_rigged_rates_driver_corr_sign_raises_naming_family_and_metric(universe, markets) -> None:
    config, instruments, _, _ = universe
    assert config.market.families.rates.driver_corr == pytest.approx(0.3)
    rigged = _config_with_rates_driver_corr(config, -0.3)
    with pytest.raises(MarketCheckError) as excinfo:
        check_market(markets["A"], instruments, rigged)
    message = str(excinfo.value)
    assert "family rates" in message
    assert "metric corr" in message


class TestRoundLevelTestCount:
    """Unit tests for the per-instrument round-level crossing counter."""

    def test_crossing_between_two_outside_closes_is_counted(self) -> None:
        # step=1, band=0.02: bands are [level-0.02*level, level+0.02*level].
        # 9.4 -> 10.6 both sit outside any band but straddle the level at 10.
        series = np.array([9.4, 10.6])
        assert round_level_test_count(series, 1.0, 0.02) == 1

    def test_entry_into_a_band_is_counted(self) -> None:
        # 9.4 is outside every band; 10.0 sits exactly on the level 10 band.
        series = np.array([9.4, 10.0])
        assert round_level_test_count(series, 1.0, 0.02) == 1

    def test_staying_inside_a_band_is_not_counted_twice(self) -> None:
        # Enter the band at t=1, then stay inside it at t=2: one test, not two.
        series = np.array([9.4, 10.0, 10.01])
        assert round_level_test_count(series, 1.0, 0.02) == 1

    def test_move_entirely_between_two_levels_is_not_counted(self) -> None:
        # 9.3 -> 9.7 stay strictly between the level-9 and level-10 bands.
        series = np.array([9.3, 9.7])
        assert round_level_test_count(series, 1.0, 0.02) == 0

    def test_a_nearest_level_of_zero_is_replaced_by_one_step(self) -> None:
        # Both closes round to a raw nearest level of 0 (a yield near the floor):
        # the level used is the step (1.0), not 0, so the band [0.3, 1.7] can
        # still contain a close; a zero-width band at 0 never could.
        series = np.array([0.1, 0.4])
        assert round_level_test_count(series, 1.0, 0.7) == 1

    def test_log_grid_path_counts_a_crossing(self) -> None:
        # step=None uses the price-dependent grid: log_grid_step(78) is 5.0, and
        # 75 is a grid level strictly between the two outside-band closes.
        series = np.array([72.0, 78.0])
        assert round_level_test_count(series, None, 0.02) == 1


def test_implied_moments_equities_matches_the_hand_formula() -> None:
    config = Config()
    equities = [
        Instrument(
            instrument_id="EQ-0001",
            family=Family.EQUITIES,
            kind=InstrumentKind.EQUITY,
            name="Equity 0001",
            currency="USD",
            sector="sector_01",
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=0.8,
            expiry_rule=None,
        ),
        Instrument(
            instrument_id="EQ-0002",
            family=Family.EQUITIES,
            kind=InstrumentKind.EQUITY,
            name="Equity 0002",
            currency="USD",
            sector="sector_01",
            rating_band=None,
            commodity_group=None,
            duration_years=None,
            beta=1.2,
            expiry_rule=None,
        ),
    ]
    params = RegimeParams(driver_mean=0.0, vol_multiplier=1.5, mean_reversion_kappa=0.0)

    vol, rho = implied_moments(Family.EQUITIES, params, equities, config)

    cfg = config.market.families.equity
    mean_beta = (0.8 + 1.2) / 2
    loading = mean_beta * cfg.market_vol
    independent = cfg.idio_vol**2 / 2
    expected_vol = params.vol_multiplier * (loading**2 + independent) ** 0.5
    expected_rho = loading / (loading**2 + independent) ** 0.5

    assert vol == pytest.approx(expected_vol)
    assert rho == pytest.approx(expected_rho)
