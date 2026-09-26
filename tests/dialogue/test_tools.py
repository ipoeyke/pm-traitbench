"""Tests for the advisor's read-only market lookup tools."""

from datetime import date

import pytest

from pm_traitbench.dialogue.tools import TOOL_DEFINITIONS, MarketLookup, run_tool
from pm_traitbench.enums import AdvisorTool
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import CurvePoint
from tests.engine.conftest import fixture_market  # noqa: F401


def test_tool_definitions_are_strict_and_cover_every_advisor_tool():
    assert [definition["name"] for definition in TOOL_DEFINITIONS] == [t.value for t in AdvisorTool]
    for definition in TOOL_DEFINITIONS:
        assert definition["strict"] is True
        schema = definition["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_get_quote_returns_the_row_on_today(market_lookup: MarketLookup, fixture_market: dict):
    today = fixture_market["dates"][5]
    expected = next(
        p for p in fixture_market["prices"] if p.instrument_id == "EQ-0001" and p.date == today
    )

    outcome = run_tool(market_lookup, "get_quote", {"instrument": "EQ-0001"}, today)

    assert outcome.is_error is False
    assert outcome.result == {
        "instrument_id": "EQ-0001",
        "name": "Equity 0001",
        "date": today.isoformat(),
        "price": expected.price,
        "spread_bp": None,
    }


def test_get_quote_resolves_by_name_case_insensitively(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_quote", {"instrument": "equity 0001"}, today)

    assert outcome.is_error is False
    assert outcome.result["instrument_id"] == "EQ-0001"


def test_get_quote_returns_spread_for_credit_issuer(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][3]
    expected = next(
        p for p in fixture_market["prices"] if p.instrument_id == "CR-IG-001" and p.date == today
    )

    outcome = run_tool(market_lookup, "get_quote", {"instrument": "CR-IG-001"}, today)

    assert outcome.is_error is False
    assert outcome.result["spread_bp"] == expected.spread_bp


def test_get_history_never_returns_rows_after_today(
    market_lookup: MarketLookup, fixture_market: dict
):
    dates = fixture_market["dates"]
    today = dates[10]

    outcome = run_tool(market_lookup, "get_history", {"instrument": "EQ-0001", "n_days": 5}, today)

    assert outcome.is_error is False
    points = outcome.result["points"]
    assert len(points) == 5
    assert [p["date"] for p in points] == [d.isoformat() for d in dates[6:11]]
    assert all(date.fromisoformat(p["date"]) <= today for p in points)


def test_get_history_field_is_spread_bp_for_credit_issuer(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][10]

    outcome = run_tool(
        market_lookup, "get_history", {"instrument": "CR-IG-001", "n_days": 3}, today
    )

    assert outcome.is_error is False
    assert outcome.result["field"] == "spread_bp"


def test_get_curve_returns_tenors_in_order(market_lookup: MarketLookup, fixture_market: dict):
    today = fixture_market["dates"][7]

    outcome = run_tool(market_lookup, "get_curve", {"instrument": "RT-USD"}, today)

    assert outcome.is_error is False
    assert list(outcome.result["levels"].keys()) == ["2Y", "5Y", "10Y", "30Y"]


def test_get_curve_on_instrument_with_no_curve_is_an_error(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_curve", {"instrument": "EQ-0001"}, today)

    assert outcome.is_error is True
    assert "curve" in outcome.result["error"]


def test_get_consensus_with_no_rows_is_an_error(market_lookup: MarketLookup, fixture_market: dict):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_consensus", {"instrument": "RT-USD"}, today)

    assert outcome.is_error is True
    assert "consensus" in outcome.result["error"]


def test_get_calendar_hides_surprise_on_future_rows(
    market_lookup: MarketLookup, fixture_market: dict
):
    dates = fixture_market["dates"]
    today = dates[25]

    outcome = run_tool(
        market_lookup,
        "get_calendar",
        {"instrument": "EQ-0001", "days_back": 15, "days_forward": 15},
        today,
    )

    assert outcome.is_error is False
    events = {(e["date"], e["event"]): e["surprise"] for e in outcome.result["events"]}
    assert events[(dates[10].isoformat(), "earnings")] == 0.2
    assert events[(dates[40].isoformat(), "earnings")] is None
    assert events[(dates[25].isoformat(), "macro_print")] == 0.4


def test_no_row_on_or_before_today_is_an_error(market_lookup: MarketLookup):
    outcome = run_tool(market_lookup, "get_quote", {"instrument": "EQ-0001"}, date(2020, 1, 1))

    assert outcome.is_error is True
    assert "no row" in outcome.result["error"]


def test_unknown_instrument_is_an_error_listing_close_names(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_quote", {"instrument": "Equty 0001"}, today)

    assert outcome.is_error is True
    assert "Equity 0001" in outcome.result["error"]


def test_out_of_range_window_is_an_error(market_lookup: MarketLookup, fixture_market: dict):
    today = fixture_market["dates"][30]

    outcome = run_tool(
        market_lookup,
        "get_calendar",
        {"instrument": "EQ-0001", "days_back": 21, "days_forward": 0},
        today,
    )

    assert outcome.is_error is True


def test_n_days_out_of_range_is_an_error(market_lookup: MarketLookup, fixture_market: dict):
    today = fixture_market["dates"][30]

    outcome = run_tool(market_lookup, "get_history", {"instrument": "EQ-0001", "n_days": 61}, today)

    assert outcome.is_error is True


def test_unknown_tool_name_is_an_error(market_lookup: MarketLookup, fixture_market: dict):
    outcome = run_tool(market_lookup, "get_weather", {}, fixture_market["dates"][0])

    assert outcome.is_error is True


def test_build_raises_dialogue_error_for_curve_id_with_no_instrument(fixture_market: dict):
    stray_curve = CurvePoint(
        seed="T", date=fixture_market["dates"][0], curve_id="NOPE", tenor="2Y", level=1.0
    )

    with pytest.raises(DialogueError, match="NOPE"):
        MarketLookup.build(
            seed="T",
            instruments=fixture_market["instruments"],
            prices=fixture_market["prices"],
            curves=[*fixture_market["curves"], stray_curve],
            consensus=fixture_market["consensus"],
            calendar=fixture_market["calendar"],
        )
