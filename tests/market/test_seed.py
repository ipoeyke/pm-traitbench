"""Tests for the seed market dataclasses and row conversion."""

import dataclasses

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import Regime
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.consensus import ConsensusResult
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.regimes import constant_path
from pm_traitbench.market.seed import SeedMarket, to_rows


def _market(config: Config) -> SeedMarket:
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    path = constant_path(Regime.RANGE, axis.n_days, config)
    output = ProcessOutput(prices={"EQ-0001": 100.0 + np.arange(axis.n_days, dtype=float)})
    return SeedMarket(
        seed="A",
        axis=axis,
        schedule=[],
        path=path,
        z=np.zeros(axis.n_days),
        output=output,
        calendar=[],
        drawn_events={},
        consensus=ConsensusResult(
            rows=[], flips=[], drawn_flips={}, street_score={}, positioning_pct={}
        ),
    )


def test_fills_defaults_to_empty_and_to_rows_ignores_it() -> None:
    market = _market(Config())
    assert market.fills == {}

    filled = dataclasses.replace(market, fills={"EQ-0001": 3})
    assert filled.fills == {"EQ-0001": 3}

    rows = to_rows(filled)
    assert rows.prices  # a real price series makes the equality below meaningful
    assert rows == to_rows(market)
