from importlib import resources
from pathlib import Path

import pytest
import yaml

from pm_traitbench.catalogues.loader import Catalogue
from pm_traitbench.enums import CheckKind, FormatOutcome
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.checks import (
    CHECKED_PARAMS,
    load_check_map,
    parse_routine_answer,
    run_check,
)

_JUDGE_KEY = ("hedging_language", "flag uncertainty once, then commit to a view")


def _packaged_map() -> dict:
    text = (
        resources.files("pm_traitbench.harness").joinpath("checks.yaml").read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "checks.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_packaged_map_covers_catalogue(catalogue: Catalogue) -> None:
    mapping = load_check_map(catalogue)
    for entry in catalogue.preferences:
        if entry.param in CHECKED_PARAMS:
            for value in entry.values:
                assert (entry.param, value) in mapping
    assert mapping[_JUDGE_KEY] is CheckKind.JUDGE


def test_map_missing_catalogue_value_raises(catalogue: Catalogue, tmp_path: Path) -> None:
    data = _packaged_map()
    del data["length_on_routine_questions"]["one sentence"]
    with pytest.raises(HarnessError, match=r"length_on_routine_questions.*one sentence"):
        load_check_map(catalogue, _write(tmp_path, data))


def test_map_missing_param_raises(catalogue: Catalogue, tmp_path: Path) -> None:
    data = _packaged_map()
    del data["number_language"]
    with pytest.raises(HarnessError, match="number_language"):
        load_check_map(catalogue, _write(tmp_path, data))


def test_map_with_unknown_value_raises(catalogue: Catalogue, tmp_path: Path) -> None:
    data = _packaged_map()
    data["response_format"]["a haiku"] = "bullets"
    with pytest.raises(HarnessError, match="a haiku"):
        load_check_map(catalogue, _write(tmp_path, data))


def test_map_with_unknown_kind_raises(catalogue: Catalogue, tmp_path: Path) -> None:
    data = _packaged_map()
    data["response_format"]["short bullets"] = "nonsense"
    with pytest.raises(HarnessError, match="nonsense"):
        load_check_map(catalogue, _write(tmp_path, data))


_WORDS_400 = " ".join(["word"] * 400)
_WORDS_401 = " ".join(["word"] * 401)

_PASSING = [
    (CheckKind.BULLETS, "- one\n- two"),
    (CheckKind.PROSE_PARAGRAPH, "Rates rose.\nCredit held."),
    (CheckKind.TABLE, "| a | b |\n|---|---|\n| 1 | 2 |"),
    (CheckKind.HEADERS, "# One\ntext\n## Two\ntext"),
    (CheckKind.UNITS_BP, "up 12bp"),
    (CheckKind.UNITS_PERCENT, "up 0.3%"),
    (CheckKind.UNITS_BOTH, "up 12bp, 0.3%"),
    (CheckKind.ONE_SENTENCE, "Yields rose 4.25 today."),
    (CheckKind.TWO_TO_THREE_SENTENCES, "Yields rose. Credit held."),
    (CheckKind.SHORT_PAGE, _WORDS_400),
    (CheckKind.NO_HEDGES, "This will rally."),
    (CheckKind.CONFIDENCE_LEVEL, "High confidence: buy."),
]
_FAILING = [
    (CheckKind.BULLETS, "- one\nplain line\n- two"),
    (CheckKind.PROSE_PARAGRAPH, "Rates rose.\n\nCredit held."),
    (CheckKind.TABLE, "| a | b |\n| 1 | 2 |"),
    (CheckKind.HEADERS, "# One\ntext"),
    (CheckKind.UNITS_BP, "up 12bp, 0.3%"),
    (CheckKind.UNITS_PERCENT, "up 12bp, 0.3%"),
    (CheckKind.UNITS_BOTH, "up 12bp"),
    (CheckKind.ONE_SENTENCE, "Yields rose. Credit held."),
    (CheckKind.TWO_TO_THREE_SENTENCES, "One. Two. Three. Four."),
    (CheckKind.SHORT_PAGE, _WORDS_401),
    (CheckKind.NO_HEDGES, "This could rally."),
    (CheckKind.CONFIDENCE_LEVEL, "Buy it."),
]


@pytest.mark.parametrize(("kind", "reply"), _PASSING)
def test_check_passes(kind: CheckKind, reply: str) -> None:
    assert run_check(kind, reply, 400) is FormatOutcome.PASS


@pytest.mark.parametrize(("kind", "reply"), _FAILING)
def test_check_fails(kind: CheckKind, reply: str) -> None:
    assert run_check(kind, reply, 400) is FormatOutcome.FAIL


@pytest.mark.parametrize(
    "kind", [CheckKind.UNITS_BP, CheckKind.UNITS_PERCENT, CheckKind.UNITS_BOTH]
)
def test_units_not_applicable_without_units(kind: CheckKind) -> None:
    assert run_check(kind, "Rates were flat.", 400) is FormatOutcome.NOT_APPLICABLE


def test_judge_kind_raises() -> None:
    with pytest.raises(ValueError, match="judge"):
        run_check(CheckKind.JUDGE, "anything", 400)


def test_parse_routine_answer() -> None:
    answer = (
        "format: response_format=short bullets; "
        "hedging_language=flag uncertainty once, then commit to a view; intrusion: none"
    )
    assert parse_routine_answer(answer) == (
        ("response_format", "short bullets"),
        ("hedging_language", "flag uncertainty once, then commit to a view"),
    )
    assert parse_routine_answer("format: none; intrusion: none") == ()
    with pytest.raises(HarnessError):
        parse_routine_answer("garbage")


_EXTRA = [
    (CheckKind.UNITS_BP, "up 12 bps", FormatOutcome.PASS),
    (CheckKind.UNITS_BP, "up 12 basis points", FormatOutcome.PASS),
    (CheckKind.UNITS_PERCENT, "up 0.3 percent", FormatOutcome.PASS),
    (CheckKind.UNITS_PERCENT, "up 0.3 per cent", FormatOutcome.PASS),
    (CheckKind.UNITS_BP, "abp is a ticker", FormatOutcome.NOT_APPLICABLE),
    (CheckKind.BULLETS, "1. x\n2) y", FormatOutcome.PASS),
    (CheckKind.TWO_TO_THREE_SENTENCES, "Only one.", FormatOutcome.FAIL),
    (CheckKind.TWO_TO_THREE_SENTENCES, "One. Two. Three.", FormatOutcome.PASS),
    (CheckKind.CONFIDENCE_LEVEL, "80% confidence on the long", FormatOutcome.PASS),
    (CheckKind.CONFIDENCE_LEVEL, "Conviction: medium.", FormatOutcome.PASS),
    (CheckKind.CONFIDENCE_LEVEL, "confidence level of 70%", FormatOutcome.PASS),
    (CheckKind.CONFIDENCE_LEVEL, "Yields up 0.3%.", FormatOutcome.FAIL),
    (CheckKind.CONFIDENCE_LEVEL, "My confidence is lower now.", FormatOutcome.FAIL),
    (CheckKind.CONFIDENCE_LEVEL, "Conviction is highly dependent on data.", FormatOutcome.FAIL),
    (CheckKind.NO_HEDGES, "In May yields rose.", FormatOutcome.PASS),
    (CheckKind.NO_HEDGES, "Yields may rise.", FormatOutcome.FAIL),
]


@pytest.mark.parametrize(("kind", "reply", "expected"), _EXTRA)
def test_check_edge_cases(kind: CheckKind, reply: str, expected: FormatOutcome) -> None:
    assert run_check(kind, reply, 400) is expected


def test_map_with_extra_param_raises(catalogue: Catalogue, tmp_path: Path) -> None:
    data = _packaged_map()
    data["register"] = {"formal": "bullets"}
    with pytest.raises(HarnessError, match="register"):
        load_check_map(catalogue, _write(tmp_path, data))


@pytest.mark.parametrize(
    "answer",
    [
        "format: response_format; intrusion: none",
        "format: a=b=c; intrusion: none",
        "format: a=b;c; intrusion: none",
        "format: =b; intrusion: none",
    ],
)
def test_parse_routine_answer_rejects_bad_items(answer: str) -> None:
    with pytest.raises(HarnessError):
        parse_routine_answer(answer)
