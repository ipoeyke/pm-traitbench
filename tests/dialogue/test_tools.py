"""Tests for the advisor's read-only market lookup tools."""

from datetime import date

import pytest

from pm_traitbench.dialogue.tools import TOOL_DEFINITIONS, MarketLookup, run_tool
from pm_traitbench.enums import AdvisorTool, EventType, Family, InstrumentKind
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import CalendarEvent, CurvePoint, Instrument, Price

_HOLIDAY = date(2026, 1, 10)  # a Saturday: not a trading day in the mini lookup below


def _mini_calendar_lookup() -> MarketLookup:
    """A hand-built lookup: 6 trading days (Jan 5-9 and Jan 12 2026, skipping the Jan 10-11
    weekend) with one earnings event on each trading day, plus one more dated on the Jan 10
    holiday itself, for exact calendar-window boundary tests.
    """
    instrument = Instrument(
        instrument_id="EQ-TST",
        family=Family.EQUITIES,
        kind=InstrumentKind.EQUITY,
        name="Test Equity",
        currency="USD",
        sector="sector_01",
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=1.0,
        expiry_rule=None,
    )
    trading_days = [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
        date(2026, 1, 12),
    ]
    prices = [
        Price(seed="T", date=day, instrument_id="EQ-TST", price=100.0, spread_bp=None)
        for day in trading_days
    ]
    event_days = [*trading_days, _HOLIDAY]
    calendar = [
        CalendarEvent(
            seed="T",
            date=day,
            instrument_id="EQ-TST",
            event=EventType.EARNINGS,
            surprise=0.1,
            affected="equities",
        )
        for day in event_days
    ]
    return MarketLookup.build(
        seed="T",
        instruments=[instrument],
        prices=prices,
        curves=[],
        consensus=[],
        calendar=calendar,
    )


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


def test_get_quote_with_none_instrument_is_an_error(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_quote", {"instrument": None}, today)

    assert outcome.is_error is True


def test_get_quote_with_integer_instrument_is_an_error(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_quote", {"instrument": 5}, today)

    assert outcome.is_error is True


def test_get_quote_missing_instrument_key_is_an_error(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_quote", {}, today)

    assert outcome.is_error is True


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


def test_get_history_with_bool_n_days_is_an_error(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][30]

    outcome = run_tool(
        market_lookup, "get_history", {"instrument": "EQ-0001", "n_days": True}, today
    )

    assert outcome.is_error is True


@pytest.mark.parametrize("n_days", [0, 61])
def test_history_bounds_reject_out_of_range_values(
    market_lookup: MarketLookup, fixture_market: dict, n_days: int
):
    today = fixture_market["dates"][30]

    outcome = run_tool(
        market_lookup, "get_history", {"instrument": "EQ-0001", "n_days": n_days}, today
    )

    assert outcome.is_error is True


@pytest.mark.parametrize("n_days", [1, 60])
def test_history_bounds_accept_inclusive_edges(
    market_lookup: MarketLookup, fixture_market: dict, n_days: int
):
    today = fixture_market["dates"][30]

    outcome = run_tool(
        market_lookup, "get_history", {"instrument": "EQ-0001", "n_days": n_days}, today
    )

    assert outcome.is_error is False


def test_get_curve_returns_tenors_in_order(market_lookup: MarketLookup, fixture_market: dict):
    today = fixture_market["dates"][7]

    outcome = run_tool(market_lookup, "get_curve", {"instrument": "RT-USD"}, today)

    assert outcome.is_error is False
    assert list(outcome.result["levels"].keys()) == ["2Y", "5Y", "10Y", "30Y"]


def test_get_curve_names_its_field_as_level(market_lookup: MarketLookup, fixture_market: dict):
    """The curves table column is `level`; the advisor is told to use a tool's field name, so
    the result must name it explicitly rather than leave the advisor to guess from `levels`.
    """
    today = fixture_market["dates"][7]

    outcome = run_tool(market_lookup, "get_curve", {"instrument": "RT-USD"}, today)

    assert outcome.result["field"] == "level"


def test_get_curve_on_instrument_with_no_curve_is_an_error(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_curve", {"instrument": "EQ-0001"}, today)

    assert outcome.is_error is True
    assert "curve" in outcome.result["error"]


def test_get_curve_never_returns_a_date_after_today(
    market_lookup: MarketLookup, fixture_market: dict
):
    dates = fixture_market["dates"]
    today = dates[7]

    outcome = run_tool(market_lookup, "get_curve", {"instrument": "RT-USD"}, today)

    assert outcome.is_error is False
    result_date = date.fromisoformat(outcome.result["date"])
    assert result_date == today
    assert result_date != dates[-1]


def test_get_consensus_with_no_rows_is_an_error(market_lookup: MarketLookup, fixture_market: dict):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_consensus", {"instrument": "RT-USD"}, today)

    assert outcome.is_error is True
    assert "consensus" in outcome.result["error"]


def test_get_consensus_never_returns_a_date_after_today(
    market_lookup: MarketLookup, fixture_market: dict
):
    dates = fixture_market["dates"]
    today = dates[7]

    outcome = run_tool(market_lookup, "get_consensus", {"instrument": "EQ-0001"}, today)

    assert outcome.is_error is False
    result_date = date.fromisoformat(outcome.result["date"])
    assert result_date == today
    assert result_date != dates[-1]


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


def test_get_calendar_with_string_days_back_is_an_error(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][30]

    outcome = run_tool(
        market_lookup,
        "get_calendar",
        {"instrument": "EQ-0001", "days_back": "5", "days_forward": 0},
        today,
    )

    assert outcome.is_error is True


@pytest.mark.parametrize("days_back,days_forward", [(-1, 0), (0, -1), (21, 0), (0, 21)])
def test_calendar_window_bounds_reject_out_of_range_values(
    market_lookup: MarketLookup, fixture_market: dict, days_back: int, days_forward: int
):
    today = fixture_market["dates"][30]

    outcome = run_tool(
        market_lookup,
        "get_calendar",
        {"instrument": "EQ-0001", "days_back": days_back, "days_forward": days_forward},
        today,
    )

    assert outcome.is_error is True


@pytest.mark.parametrize("days_back,days_forward", [(0, 0), (20, 20)])
def test_calendar_window_bounds_accept_inclusive_edges(
    market_lookup: MarketLookup, fixture_market: dict, days_back: int, days_forward: int
):
    today = fixture_market["dates"][30]

    outcome = run_tool(
        market_lookup,
        "get_calendar",
        {"instrument": "EQ-0001", "days_back": days_back, "days_forward": days_forward},
        today,
    )

    assert outcome.is_error is False


def test_get_calendar_on_a_holiday_today_does_not_widen_the_back_window():
    lookup = _mini_calendar_lookup()

    outcome = run_tool(
        lookup,
        "get_calendar",
        {"instrument": "EQ-TST", "days_back": 1, "days_forward": 0},
        _HOLIDAY,
    )

    assert outcome.is_error is False
    # Jan 9 (1 session back) and the holiday itself, never Jan 8 (2 sessions back).
    assert [e["date"] for e in outcome.result["events"]] == [
        date(2026, 1, 9).isoformat(),
        _HOLIDAY.isoformat(),
    ]


def test_get_calendar_on_a_holiday_today_does_not_drop_same_day_rows():
    lookup = _mini_calendar_lookup()

    outcome = run_tool(
        lookup,
        "get_calendar",
        {"instrument": "EQ-TST", "days_back": 0, "days_forward": 1},
        _HOLIDAY,
    )

    assert outcome.is_error is False
    # The holiday itself and Jan 12 (1 session forward), never Jan 9 (0 sessions back).
    assert [e["date"] for e in outcome.result["events"]] == [
        _HOLIDAY.isoformat(),
        date(2026, 1, 12).isoformat(),
    ]


def test_get_calendar_on_a_holiday_today_with_zero_window_only_returns_the_same_day_row():
    lookup = _mini_calendar_lookup()

    outcome = run_tool(
        lookup,
        "get_calendar",
        {"instrument": "EQ-TST", "days_back": 0, "days_forward": 0},
        _HOLIDAY,
    )

    assert outcome.is_error is False
    assert [e["date"] for e in outcome.result["events"]] == [_HOLIDAY.isoformat()]


def test_get_calendar_today_before_the_first_date_only_looks_forward():
    lookup = _mini_calendar_lookup()
    before_start = date(2026, 1, 1)

    empty = run_tool(
        lookup,
        "get_calendar",
        {"instrument": "EQ-TST", "days_back": 3, "days_forward": 0},
        before_start,
    )
    assert empty.is_error is False
    assert empty.result["events"] == []

    forward = run_tool(
        lookup,
        "get_calendar",
        {"instrument": "EQ-TST", "days_back": 0, "days_forward": 1},
        before_start,
    )
    assert forward.is_error is False
    assert [e["date"] for e in forward.result["events"]] == [date(2026, 1, 5).isoformat()]


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


def test_unknown_instrument_close_to_an_id_names_it(market_lookup: MarketLookup, fixture_market):
    today = fixture_market["dates"][0]

    outcome = run_tool(market_lookup, "get_quote", {"instrument": "EQ-001"}, today)

    assert outcome.is_error is True
    assert "EQ-0001" in outcome.result["error"]


def test_unknown_instrument_with_nothing_close_lists_the_sessions_instruments(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]
    session_names = {"EQ-0001": "Equity 0001", "CR-IG-001": "Issuer IG 001"}

    outcome = run_tool(market_lookup, "get_quote", {"instrument": "SPX"}, today, session_names)

    assert outcome.is_error is True
    assert outcome.result["error"] == (
        "unknown instrument 'SPX'; the PM's instruments: "
        "Equity 0001 (EQ-0001), Issuer IG 001 (CR-IG-001)"
    )


def test_unknown_instrument_lists_close_names_and_the_sessions_instruments(
    market_lookup: MarketLookup, fixture_market: dict
):
    today = fixture_market["dates"][0]
    session_names = {"CR-IG-001": "Issuer IG 001"}

    outcome = run_tool(market_lookup, "get_quote", {"instrument": "EQ-001"}, today, session_names)

    assert outcome.is_error is True
    assert "closest names: " in outcome.result["error"]
    assert "EQ-0001" in outcome.result["error"]
    assert outcome.result["error"].endswith("; the PM's instruments: Issuer IG 001 (CR-IG-001)")


@pytest.mark.parametrize("query", ["usd", "UST", "treasuries", "usd curve", "USD sovereign"])
def test_unknown_instrument_suggests_the_curve_its_currency_names(
    market_lookup: MarketLookup, fixture_market: dict, query: str
):
    today = fixture_market["dates"][0]
    curves = [c for c in market_lookup.instruments.values() if c.family == Family.RATES]
    usd = next(c for c in curves if c.currency == "USD")

    outcome = run_tool(market_lookup, "get_curve", {"instrument": query}, today)

    assert outcome.is_error is True
    suggested = outcome.result["error"].split("closest names: ")[1].split("; the PM")[0]
    assert suggested.split(", ")[0] == f"{usd.name} ({usd.instrument_id})"


def test_unknown_tool_name_is_an_error(market_lookup: MarketLookup, fixture_market: dict):
    outcome = run_tool(market_lookup, "get_weather", {}, fixture_market["dates"][0])

    assert outcome.is_error is True
    assert "get_weather" in outcome.result["error"]


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


def test_build_raises_dialogue_error_for_curve_id_naming_the_wrong_kind(fixture_market: dict):
    stray_curve = CurvePoint(
        seed="T", date=fixture_market["dates"][0], curve_id="EQ-0001", tenor="2Y", level=1.0
    )

    with pytest.raises(DialogueError, match="EQ-0001"):
        MarketLookup.build(
            seed="T",
            instruments=fixture_market["instruments"],
            prices=fixture_market["prices"],
            curves=[*fixture_market["curves"], stray_curve],
            consensus=fixture_market["consensus"],
            calendar=fixture_market["calendar"],
        )


def test_lookup_instruments_are_in_sorted_instrument_id_order(market_lookup: MarketLookup):
    ids = list(market_lookup.instruments)

    assert ids == sorted(ids)
