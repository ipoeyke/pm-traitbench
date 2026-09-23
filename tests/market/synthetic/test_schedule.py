"""Tests for the regime schedule: spans built per market seed."""

from datetime import date, timedelta

import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import Regime
from pm_traitbench.market.synthetic.schedule import build_schedule

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
