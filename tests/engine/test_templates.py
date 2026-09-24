"""Tests for catalogue-driven idea names, theses, outcomes and signpost text."""

import numpy as np

from pm_traitbench.engine.series import LegRef
from pm_traitbench.engine.templates import (
    idea_name,
    render_outcome,
    render_signpost_text,
    render_thesis,
)
from pm_traitbench.enums import AssetClass, Expression, Tenor


def test_idea_name_outright(fixture_view) -> None:
    legs = (LegRef("EQ-0001", None, 1.0),)
    assert (
        idea_name(fixture_view, legs, Expression.OUTRIGHT)
        == fixture_view.instruments["EQ-0001"].name
    )


def test_idea_name_pair(fixture_view) -> None:
    legs = (LegRef("EQ-0001", None, 1.0), LegRef("EQ-0002", None, -1.0))
    name = idea_name(fixture_view, legs, Expression.PAIR)
    first = fixture_view.instruments["EQ-0001"].name
    second = fixture_view.instruments["EQ-0002"].name
    assert name == f"{first} versus {second}"


def test_idea_name_curve_orders_short_tenor_first(fixture_view) -> None:
    legs = (LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0))
    assert idea_name(fixture_view, legs, Expression.CURVE) == "2Y versus 10Y"


def test_idea_name_calendar_spread_orders_front_month_first(fixture_view) -> None:
    legs = (LegRef("CM-CRD", Tenor.M5, -1.0), LegRef("CM-CRD", Tenor.M1, 1.0))
    assert idea_name(fixture_view, legs, Expression.CALENDAR_SPREAD) == "M1 versus M5"


def test_render_thesis_contains_entry_and_target(catalogue) -> None:
    rng = np.random.default_rng(0)
    text = render_thesis(
        catalogue,
        AssetClass.EQUITIES,
        Expression.PAIR,
        side="long",
        name="A versus B",
        entry=12.34,
        target=15.67,
        move=1.2,
        unit="pct",
        horizon=20,
        rng=rng,
    )
    assert "12.34" in text
    assert "15.67" in text


def test_render_thesis_rounds_entry_and_target_for_bp_series(catalogue) -> None:
    rng = np.random.default_rng(0)
    text = render_thesis(
        catalogue,
        AssetClass.RATES_CREDIT,
        Expression.CURVE,
        side="steepener",
        name="2Y versus 10Y",
        entry=123.456789,
        target=145.987654321,
        move=5.4,
        unit="bp",
        horizon=20,
        rng=rng,
    )
    assert "123.5" in text
    assert "146.0" in text
    assert "123.456789" not in text
    assert "145.987654" not in text


def test_render_outcome_win_contains_signed_pnl(catalogue) -> None:
    rng = np.random.default_rng(0)
    text = render_outcome(catalogue, kind="win", pnl=3.456, unit="pct", closer="target", rng=rng)
    assert "+3.5" in text


def test_render_outcome_loss_contains_signed_pnl(catalogue) -> None:
    rng = np.random.default_rng(0)
    text = render_outcome(catalogue, kind="loss", pnl=-2.1, unit="bp", closer="stop", rng=rng)
    assert "-2.1" in text


def test_render_signpost_text_level_contains_window(catalogue) -> None:
    rng = np.random.default_rng(0)
    text = render_signpost_text(
        catalogue, AssetClass.EQUITIES, "level", level=12.3, unit="pct", window=4, rng=rng
    )
    # Every equities 'level' template places the window right before "sessions".
    assert "4 sessions" in text


def test_render_signpost_text_event_contains_event(catalogue) -> None:
    rng = np.random.default_rng(0)
    text = render_signpost_text(
        catalogue, AssetClass.RATES_CREDIT, "event", event="cb_meeting", rng=rng
    )
    assert "cb meeting" in text
