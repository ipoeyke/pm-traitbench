"""Tests for catalogue-driven idea names, theses, outcomes and signpost text."""

import math

import numpy as np

from pm_traitbench.engine.series import LegRef
from pm_traitbench.engine.templates import (
    idea_name,
    level_text,
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


def test_idea_name_curve_names_the_currency_and_orders_short_tenor_first(fixture_view) -> None:
    legs = (LegRef("RT-USD", Tenor.Y10, 1.0), LegRef("RT-USD", Tenor.Y2, -1.0))
    assert idea_name(fixture_view, legs, Expression.CURVE) == "USD 2Y versus 10Y"


def test_idea_name_rates_outright_names_the_currency_and_tenor(fixture_view) -> None:
    legs = (LegRef("RT-USD", Tenor.Y10, 1.0),)
    assert idea_name(fixture_view, legs, Expression.OUTRIGHT) == "USD 10Y"


def test_idea_name_calendar_spread_names_the_commodity_front_month_first(fixture_view) -> None:
    legs = (LegRef("CM-CRD", Tenor.M5, -1.0), LegRef("CM-CRD", Tenor.M1, 1.0))
    assert idea_name(fixture_view, legs, Expression.CALENDAR_SPREAD) == "Crude M1 versus M5"


def test_level_text_quotes_a_price_or_a_fixed_precision_series_level() -> None:
    assert level_text(100.0 * math.log(55.0), "pct", price_quoted=True) == "55.00"
    assert level_text(420.06, "bp", price_quoted=False) == "420.1bp"
    assert level_text(-4.5, "pct", price_quoted=False) == "-4.50%"


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
        price_quoted=False,
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
        price_quoted=False,
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


def test_render_thesis_keeps_trailing_zeros_on_a_quoted_price(catalogue) -> None:
    for seed in range(3):
        text = render_thesis(
            catalogue,
            AssetClass.COMMODITIES,
            Expression.OUTRIGHT,
            side="long",
            name="silver",
            entry=100.0 * math.log(31.2),
            target=100.0 * math.log(37.0),
            move=5.0,
            unit="pct",
            price_quoted=True,
            horizon=20,
            rng=np.random.default_rng(seed),
        )
        assert "31.20" in text
        assert "37.00" in text
        assert "%" not in text.replace("+5.00%", "")
