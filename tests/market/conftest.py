"""Fixtures shared by the shared-module and synthetic-module check tests."""

import pytest

from pm_traitbench.config import Config
from pm_traitbench.market.axis import SimAxis, build_axis
from pm_traitbench.market.seed import SeedMarket
from pm_traitbench.market.synthetic.build import build_seed
from pm_traitbench.market.synthetic.drivers import DriverShocks, draw_shocks
from pm_traitbench.market.synthetic.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import Instrument


@pytest.fixture(scope="module")
def universe() -> tuple[Config, tuple[Instrument, ...], SimAxis, DriverShocks]:
    """The default config, full default universe, axis and driver shocks, shared across seeds."""
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = tuple(build_universe(config, stream(config.seed.root, "market", "universe")))
    shocks = draw_shocks(config.seed.root, axis.n_days)
    return config, instruments, axis, shocks


@pytest.fixture(scope="module")
def markets(universe) -> dict[str, SeedMarket]:
    """Every default seed, generated once for the whole module."""
    config, instruments, axis, shocks = universe
    return {
        seed: build_seed(config, seed, instruments, shocks, axis) for seed in config.market.seeds
    }


@pytest.fixture
def build_axis_and_universe():
    """Factory: build one config's axis and instrument universe for a given root seed."""

    def _build(config: Config, root: int) -> tuple[SimAxis, list[Instrument]]:
        axis = build_axis(config.timeline(), config.market.burn_in_days)
        instruments = build_universe(config, stream(root, "market", "universe"))
        return axis, instruments

    return _build
