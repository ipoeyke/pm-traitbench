"""Tests for the ledger consistency layer: trade-mention checks and level warnings."""

from datetime import timedelta

from pm_traitbench.dialogue.validate.ledger import check_trades, count_level_warnings
from pm_traitbench.enums import AdvisorTool, Side, Tenor
from pm_traitbench.tables.schema import ToolCall, canonical_json
from tests.dialogue.validate.fixtures import (
    advisor_turn,
    level_mention,
    log_of,
    pm_turn,
    skeleton_of,
    trade_mention,
)
from tests.gates.fixtures import DEFAULT_DATE as DATE
from tests.gates.fixtures import DEFAULT_IDEA_ID as IDEA
from tests.gates.fixtures import PM_ID, ledger_row


def _log(skeleton, *turns):
    return log_of(skeleton.session_id, skeleton.pm_id, turns)


def test_exact_mention_of_the_days_trade_passes():
    row = ledger_row()
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))
    mention = trade_mention(row)
    log = _log(skeleton, pm_turn("bought it", mentions=(mention,)), advisor_turn("noted"))

    assert check_trades(log, skeleton, (row,), size_tolerance=0.05) == ()


def test_size_within_tolerance_passes_and_beyond_fails():
    row = ledger_row()
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))

    within = trade_mention(row, size=104.0)
    log = _log(skeleton, pm_turn("bought it", mentions=(within,)), advisor_turn("noted"))
    assert check_trades(log, skeleton, (row,), size_tolerance=0.05) == ()

    beyond = trade_mention(row, size=106.0)
    log = _log(skeleton, pm_turn("bought it", mentions=(beyond,)), advisor_turn("noted"))
    assert check_trades(log, skeleton, (row,), size_tolerance=0.05) == (
        "trade not in ledger: ti_001 EQ-0001 - buy 106.0",
        "trade not mentioned: ti_001 EQ-0001 - buy 100.0",
    )


def test_wrong_side_fails():
    row = ledger_row()
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))
    mention = trade_mention(row).model_copy(update={"side": Side.SELL})
    log = _log(skeleton, pm_turn("sold it", mentions=(mention,)), advisor_turn("noted"))

    assert check_trades(log, skeleton, (row,), size_tolerance=0.05) == (
        "trade not in ledger: ti_001 EQ-0001 - sell 100.0",
        "trade not mentioned: ti_001 EQ-0001 - buy 100.0",
    )


def test_wrong_tenor_fails():
    row = ledger_row(instrument_id="CM-CRD", tenor=Tenor.M1)
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))
    mention = trade_mention(row).model_copy(update={"tenor": Tenor.M2})
    log = _log(skeleton, pm_turn("bought M2", mentions=(mention,)), advisor_turn("noted"))

    assert check_trades(log, skeleton, (row,), size_tolerance=0.05) == (
        "trade not in ledger: ti_001 CM-CRD M2 buy 100.0",
        "trade not mentioned: ti_001 CM-CRD M1 buy 100.0",
    )


def test_earlier_row_of_a_session_idea_may_be_mentioned():
    earlier = DATE - timedelta(days=3)
    row = ledger_row(date=earlier)
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))
    mention = trade_mention(row)
    log = _log(skeleton, pm_turn("recall that buy", mentions=(mention,)), advisor_turn("noted"))

    assert check_trades(log, skeleton, (row,), size_tolerance=0.05) == ()


def test_row_dated_after_the_session_is_not_matchable():
    later = DATE + timedelta(days=3)
    row = ledger_row(date=later)
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))
    mention = trade_mention(row)
    log = _log(skeleton, pm_turn("bought it", mentions=(mention,)), advisor_turn("noted"))

    assert check_trades(log, skeleton, (row,), size_tolerance=0.05) == (
        "trade not in ledger: ti_001 EQ-0001 - buy 100.0",
    )


def test_every_same_day_row_must_be_mentioned():
    row_a = ledger_row()
    row_b = ledger_row(instrument_id="EQ-0002", size=50.0)
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))
    mention = trade_mention(row_a)
    log = _log(skeleton, pm_turn("bought EQ-0001", mentions=(mention,)), advisor_turn("noted"))

    assert check_trades(log, skeleton, (row_a, row_b), size_tolerance=0.05) == (
        "trade not mentioned: ti_001 EQ-0002 - buy 50.0",
    )


def test_same_day_row_of_an_idea_not_in_the_skeleton_is_not_required():
    other_idea = "ti_002"
    row = ledger_row(trade_idea_id=other_idea)
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))
    log = _log(skeleton, pm_turn("no mention"), advisor_turn("noted"))

    assert check_trades(log, skeleton, (row,), size_tolerance=0.05) == ()


def test_reasons_are_sorted_and_unique():
    row = ledger_row()
    skeleton = skeleton_of(PM_ID, DATE, (IDEA,))
    mention = trade_mention(row).model_copy(update={"side": Side.SELL})
    log = _log(
        skeleton,
        pm_turn("sold it", mentions=(mention,)),
        advisor_turn("noted"),
        pm_turn("sold it again", mentions=(mention,)),
        advisor_turn("noted again"),
    )

    reasons = check_trades(log, skeleton, (row,), size_tolerance=0.05)
    assert reasons == (
        "trade not in ledger: ti_001 EQ-0001 - sell 100.0",
        "trade not mentioned: ti_001 EQ-0001 - buy 100.0",
    )

    # Two not-in-ledger mentions in reverse-sorted order prove the tuple is sorted,
    # not in insertion order.
    later_idea = trade_mention(ledger_row(trade_idea_id="ti_002", size=10.0))
    earlier_idea = trade_mention(ledger_row(trade_idea_id="ti_001", size=10.0))
    reverse_skeleton = skeleton_of(PM_ID, DATE, ())
    reverse_log = _log(
        reverse_skeleton,
        pm_turn("later idea first", mentions=(later_idea,)),
        advisor_turn("noted"),
        pm_turn("earlier idea second", mentions=(earlier_idea,)),
        advisor_turn("noted again"),
    )
    reverse_reasons = check_trades(reverse_log, reverse_skeleton, (), size_tolerance=0.05)
    assert reverse_reasons == (
        "trade not in ledger: ti_001 EQ-0001 - buy 10.0",
        "trade not in ledger: ti_002 EQ-0001 - buy 10.0",
    )


def _quote_call(price: float) -> ToolCall:
    return ToolCall(
        name=AdvisorTool.GET_QUOTE,
        input_json=canonical_json({"instrument": "EQ-0001"}),
        result_json=canonical_json(
            {
                "instrument_id": "EQ-0001",
                "name": "Equity 0001",
                "date": DATE.isoformat(),
                "price": price,
                "spread_bp": None,
            }
        ),
        is_error=False,
    )


def test_advisor_level_confirmed_by_tool_result_value(market_lookup):
    skeleton = skeleton_of(PM_ID, DATE, ())
    call = _quote_call(101.37)

    confirmed = level_mention("EQ-0001", "price", 101.4)
    log = _log(
        skeleton, pm_turn("what's the price?"), advisor_turn("about 101.4", (confirmed,), (call,))
    )
    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 0

    off = level_mention("EQ-0001", "price", 110)
    log = _log(skeleton, pm_turn("what's the price?"), advisor_turn("110", (off,), (call,)))
    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 1


def test_advisor_level_ignores_error_tool_results(market_lookup):
    skeleton = skeleton_of(PM_ID, DATE, ())
    error_call = ToolCall(
        name=AdvisorTool.GET_QUOTE,
        input_json=canonical_json({"instrument": "EQ-0001"}),
        result_json=canonical_json({"error": "unknown instrument", "price": 101.37}),
        is_error=True,
    )
    mention = level_mention("EQ-0001", "price", 101.37)
    log = _log(
        skeleton, pm_turn("what's the price?"), advisor_turn("no data", (mention,), (error_call,))
    )

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 1


def test_advisor_level_ignores_bools_in_the_number_walk(market_lookup):
    skeleton = skeleton_of(PM_ID, DATE, ())
    call = ToolCall(
        name=AdvisorTool.GET_QUOTE,
        input_json=canonical_json({"instrument": "EQ-0001"}),
        result_json=canonical_json({"instrument_id": "EQ-0001", "active": True}),
        is_error=False,
    )
    mention = level_mention("EQ-0001", "price", 1.0)
    log = _log(skeleton, pm_turn("what's the price?"), advisor_turn("no data", (mention,), (call,)))

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 1


def test_advisor_level_finds_values_nested_in_curve_levels_and_history_points(market_lookup):
    skeleton = skeleton_of(PM_ID, DATE, ())
    curve_call = ToolCall(
        name=AdvisorTool.GET_CURVE,
        input_json=canonical_json({"instrument": "RT-USD"}),
        result_json=canonical_json(
            {
                "instrument_id": "RT-USD",
                "name": "US Treasury Curve",
                "date": DATE.isoformat(),
                "field": "level",
                "levels": {"2Y": 3.51, "5Y": 3.82},
            }
        ),
        is_error=False,
    )
    history_call = ToolCall(
        name=AdvisorTool.GET_HISTORY,
        input_json=canonical_json({"instrument": "EQ-0001", "n_days": 5}),
        result_json=canonical_json(
            {
                "instrument_id": "EQ-0001",
                "name": "Equity 0001",
                "field": "price",
                "points": [{"date": DATE.isoformat(), "value": 101.37}],
            }
        ),
        is_error=False,
    )
    curve_mention = level_mention("RT-USD", "level", 3.82, tenor=Tenor.Y5)
    history_mention = level_mention("EQ-0001", "price", 101.37)
    log = _log(
        skeleton,
        pm_turn("levels?"),
        advisor_turn(
            "here they are",
            (curve_mention, history_mention),
            (curve_call, history_call),
        ),
    )

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 0


def test_pm_level_confirmed_against_market_price(market_lookup, fixture_market):
    instrument = sorted(fixture_market["instruments"], key=lambda i: i.instrument_id)[0]
    row = market_lookup.latest_price(instrument.instrument_id, DATE)
    mention = level_mention(instrument.instrument_id, "price", row.price)
    skeleton = skeleton_of(PM_ID, DATE, ())
    log = _log(skeleton, pm_turn("mark", mentions=(mention,)), advisor_turn("ok"))

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 0


def test_pm_level_unknown_field_is_a_warning(market_lookup, fixture_market):
    instrument = sorted(fixture_market["instruments"], key=lambda i: i.instrument_id)[0]
    mention = level_mention(instrument.instrument_id, "bogus_field", 123.0)
    skeleton = skeleton_of(PM_ID, DATE, ())
    log = _log(skeleton, pm_turn("mark", mentions=(mention,)), advisor_turn("ok"))

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 1


def test_pm_level_confirmed_against_curve_tenor(market_lookup):
    _, levels = market_lookup.curve_on_or_before("RT-USD", DATE)
    mention = level_mention("RT-USD", "level", levels[Tenor.Y5], tenor=Tenor.Y5)
    skeleton = skeleton_of(PM_ID, DATE, ())
    log = _log(skeleton, pm_turn("curve", mentions=(mention,)), advisor_turn("ok"))

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 0


def test_pm_level_with_no_tenor_is_a_warning(market_lookup):
    mention = level_mention("RT-USD", "level", 3.8)
    skeleton = skeleton_of(PM_ID, DATE, ())
    log = _log(skeleton, pm_turn("curve", mentions=(mention,)), advisor_turn("ok"))

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 1


def test_pm_level_confirmed_against_spread_bp(market_lookup):
    row = market_lookup.latest_price("CR-IG-001", DATE)
    mention = level_mention("CR-IG-001", "spread_bp", row.spread_bp)
    skeleton = skeleton_of(PM_ID, DATE, ())
    log = _log(skeleton, pm_turn("spread", mentions=(mention,)), advisor_turn("ok"))

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 0


def test_pm_level_confirmed_against_street_score(market_lookup):
    row = market_lookup.latest_consensus("EQ-0001", DATE)
    mention = level_mention("EQ-0001", "street_score", row.street_score)
    skeleton = skeleton_of(PM_ID, DATE, ())
    log = _log(skeleton, pm_turn("consensus", mentions=(mention,)), advisor_turn("ok"))

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 0


def test_pm_level_missing_market_row_is_a_warning(market_lookup):
    mention = level_mention("EQ-9999", "price", 100.0)
    skeleton = skeleton_of(PM_ID, DATE, ())
    log = _log(skeleton, pm_turn("mark", mentions=(mention,)), advisor_turn("ok"))

    assert count_level_warnings(log, skeleton, market_lookup, level_tolerance=0.01) == 1
