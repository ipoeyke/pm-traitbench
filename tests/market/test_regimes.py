"""Tests for the regime schedule: spans, lookup and per-day regime paths."""

from datetime import date, timedelta

import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import Regime
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.regimes import RegimeLookup, build_schedule, constant_path, regime_path

_DEFAULT_DATES = (
    (date(2026, 1, 5), date(2026, 4, 24)),
    (date(2026, 4, 27), date(2026, 7, 31)),
    (date(2026, 8, 3), date(2027, 1, 1)),
)


def test_seed_a_spans_match_default_calendar() -> None:
    config = Config()
    spans = build_schedule(config, "A", config.timeline())
    assert [(s.regime, s.date_start, s.date_end) for s in spans] == [
        (Regime.RANGE, date(2026, 1, 5), date(2026, 4, 24)),
        (Regime.RISK_OFF, date(2026, 4, 27), date(2026, 7, 31)),
        (Regime.RISK_ON, date(2026, 8, 3), date(2027, 1, 1)),
    ]


@pytest.mark.parametrize("seed", ["B", "C"])
def test_seed_b_and_c_spans_follow_their_regime_order_on_the_same_dates(seed: str) -> None:
    config = Config()
    order = config.market.seeds[seed]
    spans = build_schedule(config, seed, config.timeline())
    assert [(s.regime, s.date_start, s.date_end) for s in spans] == [
        (regime, start, end) for regime, (start, end) in zip(order, _DEFAULT_DATES, strict=True)
    ]


def test_all_spans_carry_the_requested_seed_string() -> None:
    config = Config()
    spans = build_schedule(config, "B", config.timeline())
    assert all(span.seed == "B" for span in spans)


def test_spans_tile_the_horizon_with_no_gap() -> None:
    config = Config()
    timeline = config.timeline()
    spans = build_schedule(config, "A", timeline)
    for prev, curr in zip(spans, spans[1:], strict=False):
        assert (curr.date_start - prev.date_end).days == 3
    assert spans[0].date_start == timeline.week_start(1)
    assert spans[-1].date_end == timeline.week_start(timeline.n_weeks) + timedelta(days=4)


def test_lookup_maps_burn_in_dates_to_the_seeds_first_regime() -> None:
    config = Config()
    timeline = config.timeline()
    axis = build_axis(timeline, config.market.burn_in_days)
    spans = build_schedule(config, "A", timeline)
    lookup = RegimeLookup(spans, burn_in_regime=config.market.seeds["A"][0])
    for day in axis.dates[: axis.n_burn]:
        assert lookup.regime(day) == Regime.RANGE


def test_lookup_matches_span_regime_including_span_endpoints() -> None:
    config = Config()
    spans = build_schedule(config, "A", config.timeline())
    lookup = RegimeLookup(spans, burn_in_regime=Regime.RANGE)
    assert lookup.regime(date(2026, 1, 5)) == Regime.RANGE
    assert lookup.regime(date(2026, 4, 24)) == Regime.RANGE
    assert lookup.regime(date(2026, 4, 27)) == Regime.RISK_OFF
    assert lookup.regime(date(2026, 7, 31)) == Regime.RISK_OFF
    assert lookup.regime(date(2026, 8, 3)) == Regime.RISK_ON
    assert lookup.regime(date(2027, 1, 1)) == Regime.RISK_ON


def test_lookup_raises_for_a_date_after_the_last_span() -> None:
    config = Config()
    spans = build_schedule(config, "A", config.timeline())
    lookup = RegimeLookup(spans, burn_in_regime=Regime.RANGE)
    with pytest.raises(ValueError):
        lookup.regime(date(2027, 1, 4))


def test_regime_path_arrays_match_params_on_sampled_days() -> None:
    config = Config()
    timeline = config.timeline()
    axis = build_axis(timeline, config.market.burn_in_days)
    spans = build_schedule(config, "A", timeline)
    lookup = RegimeLookup(spans, burn_in_regime=config.market.seeds["A"][0])
    path = regime_path(axis, lookup, config)

    assert len(path.regimes) == axis.n_days
    for i in (0, axis.n_burn, axis.n_days // 2, axis.n_days - 1):
        regime = lookup.regime(axis.dates[i])
        params = config.market.regimes.params(regime)
        assert path.regimes[i] == regime
        assert path.driver_mean[i] == params.driver_mean
        assert path.vol_multiplier[i] == params.vol_multiplier
        assert path.kappa[i] == params.mean_reversion_kappa


def test_constant_path_holds_one_regime_for_every_day() -> None:
    config = Config()
    path = constant_path(Regime.RISK_OFF, 10, config)
    params = config.market.regimes.params(Regime.RISK_OFF)
    assert path.regimes == (Regime.RISK_OFF,) * 10
    assert (path.driver_mean == params.driver_mean).all()
    assert (path.vol_multiplier == params.vol_multiplier).all()
    assert (path.kappa == params.mean_reversion_kappa).all()
