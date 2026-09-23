"""Tests for the shared realised-check building blocks: family indices, the
check report and calendar-count comparisons.
"""

import dataclasses
import json

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import EventType, Family
from pm_traitbench.errors import MarketCheckError
from pm_traitbench.market.axis import SimAxis, build_axis
from pm_traitbench.market.check import CheckReport, family_indices
from pm_traitbench.market.seed import SeedMarket
from pm_traitbench.market.synthetic.build import build_seed
from pm_traitbench.market.synthetic.check import check_market
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


def test_dropped_positioning_report_row_raises_naming_the_count_metric(universe, markets) -> None:
    config, instruments, _, _ = universe
    market = markets["A"]
    dropped_index = next(
        i for i, row in enumerate(market.calendar) if row.event == EventType.POSITIONING_REPORT
    )
    calendar = list(market.calendar)
    del calendar[dropped_index]
    modified = dataclasses.replace(market, calendar=calendar)

    with pytest.raises(MarketCheckError) as excinfo:
        check_market(modified, instruments, config)
    assert "count:positioning_report" in str(excinfo.value)


def test_duplicated_and_missing_expiry_with_same_total_raises(universe, markets) -> None:
    # A duplicated expiry date leaves another month uncovered while the row
    # count stays the same, so the check must compare exact dates, not counts.
    config, instruments, _, _ = universe
    market = markets["A"]
    commodity = next(i for i in instruments if i.family == Family.COMMODITIES)
    calendar = list(market.calendar)
    expiry_indices = [
        i
        for i, row in enumerate(calendar)
        if row.event == EventType.CONTRACT_EXPIRY and row.instrument_id == commodity.instrument_id
    ]
    assert len(expiry_indices) >= 2
    first_date = calendar[expiry_indices[0]].date
    calendar[expiry_indices[1]] = calendar[expiry_indices[1]].model_copy(
        update={"date": first_date}
    )
    modified = dataclasses.replace(market, calendar=calendar)

    with pytest.raises(MarketCheckError) as excinfo:
        check_market(modified, instruments, config)
    assert f"count:contract_expiry:{commodity.instrument_id}" in str(excinfo.value)


def test_duplicated_and_missing_positioning_report_with_same_total_raises(
    universe, markets
) -> None:
    # Same total positioning_report row count, but one Friday's report is
    # duplicated onto another Friday's date, leaving that other Friday uncovered.
    config, instruments, _, _ = universe
    market = markets["A"]
    calendar = list(market.calendar)
    report_indices = [
        i for i, row in enumerate(calendar) if row.event == EventType.POSITIONING_REPORT
    ]
    assert len(report_indices) >= 2
    first_date = calendar[report_indices[0]].date
    calendar[report_indices[1]] = calendar[report_indices[1]].model_copy(
        update={"date": first_date}
    )
    modified = dataclasses.replace(market, calendar=calendar)

    with pytest.raises(MarketCheckError) as excinfo:
        check_market(modified, instruments, config)
    assert "count:positioning_report" in str(excinfo.value)


def test_to_dict_round_trips_through_json(universe, markets) -> None:
    config, instruments, _, _ = universe
    report = check_market(markets["A"], instruments, config)
    payload = json.dumps(report.to_dict())
    restored = json.loads(payload)
    assert restored["seed"] == "A"
    assert len(restored["metrics"]) == len(report.metrics)
    count_metrics = [m for m in restored["metrics"] if m["metric"].startswith("count:")]
    assert all(m["regime"] is None for m in count_metrics)


def test_family_indices_covers_every_family(universe, markets) -> None:
    _, instruments, _, _ = universe
    indices = family_indices(markets["A"], instruments)
    assert set(indices) == set(Family)
    for series in indices.values():
        assert np.isnan(series[0])
        assert not np.isnan(series[1:]).any()


def test_check_report_is_a_frozen_dataclass(universe, markets) -> None:
    config, instruments, _, _ = universe
    report = check_market(markets["A"], instruments, config)
    assert isinstance(report, CheckReport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.seed = "Z"
