"""Tests for the shared series-level renderer."""

import math

import pytest

from pm_traitbench.enums import AssetClass, Expression
from pm_traitbench.levels import idea_level_text, is_price_quoted, level_text


def test_level_text_quotes_a_price_or_a_fixed_precision_series_level() -> None:
    assert level_text(100.0 * math.log(55.0), "pct", price_quoted=True) == "55.00"
    assert level_text(420.06, "bp", price_quoted=False) == "420.1bp"
    assert level_text(-4.5, "pct", price_quoted=False) == "-4.50%"


@pytest.mark.parametrize(
    ("asset_class", "expression", "expected"),
    [
        (AssetClass.EQUITIES, Expression.OUTRIGHT, True),
        (AssetClass.COMMODITIES, Expression.OUTRIGHT, True),
        (AssetClass.EQUITIES, Expression.PAIR, False),
        (AssetClass.COMMODITIES, Expression.CALENDAR_SPREAD, False),
        (AssetClass.RATES_CREDIT, Expression.OUTRIGHT, False),
    ],
)
def test_only_equity_and_commodity_outrights_are_price_quoted(
    asset_class: AssetClass, expression: Expression, expected: bool
) -> None:
    assert is_price_quoted(asset_class, expression) is expected


def test_idea_level_text_undoes_the_log_price_for_a_round_level() -> None:
    log_7500 = 100.0 * math.log(7500.0)

    assert idea_level_text(log_7500, AssetClass.COMMODITIES, Expression.OUTRIGHT) == "7500.00"
    assert idea_level_text(log_7500, AssetClass.COMMODITIES, Expression.CALENDAR_SPREAD) == (
        f"{log_7500:.2f}%"
    )
    assert idea_level_text(425.0, AssetClass.RATES_CREDIT, Expression.CURVE) == "425.0bp"
