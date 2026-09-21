import pytest

from pm_traitbench.numeric import parse_number


@pytest.mark.parametrize("text", ["1_0", " 5 ", "nan", "inf", "-inf", "Infinity", "1e999"])
def test_ambiguous_or_non_finite_text_is_not_numeric(text: str) -> None:
    assert parse_number(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("-15", -15.0), ("2.5", 2.5), ("1e3", 1000.0), (".5", 0.5)],
)
def test_plain_decimal_and_scientific_notation_is_numeric(text: str, expected: float) -> None:
    assert parse_number(text) == expected
