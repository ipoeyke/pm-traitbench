"""Tests for the stage progress lines."""

import pytest

from pm_traitbench.dialogue.client import UsageTotals
from pm_traitbench.dialogue.progress import Progress, format_count, format_elapsed, format_line


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1k"),
        (182_400, "182k"),
        (999_499, "999k"),
        (999_500, "1.0M"),
        (5_000_000, "5.0M"),
    ],
)
def test_format_count(n: int, expected: str) -> None:
    assert format_count(n) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(21, "0m21s"), (3_599, "59m59s"), (3_720, "1h02m")],
)
def test_format_elapsed(seconds: float, expected: str) -> None:
    assert format_elapsed(seconds) == expected


def test_format_line_without_budget() -> None:
    totals = UsageTotals(calls=4, cache_hits=0, input_tokens=18_000, output_tokens=2_000)

    line = format_line("dialogue", "sessions", 1, 96, 0, totals, None, 21)

    assert (
        line
        == "[dialogue] 1/96 sessions (0 failed) | 4 calls, 0 cached | 18k in, 2k out tokens | 0m21s"
    )


def test_format_line_with_budget() -> None:
    totals = UsageTotals(calls=9, cache_hits=1, input_tokens=37_000, output_tokens=5_000)

    line = format_line("dialogue", "sessions", 2, 96, 1, totals, 5_000_000, 40)

    assert line == (
        "[dialogue] 2/96 sessions (1 failed) | 9 calls, 1 cached "
        "| 37k in, 5k out tokens (42k/5.0M budget) | 0m40s"
    )


def test_progress_prints_start_line_then_one_line_per_finish(
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [100.0]
    totals = UsageTotals()
    progress = Progress("gate2", "units", 2, totals, None, clock=lambda: now[0])

    progress.start(8)
    now[0] = 121.0
    totals.calls = 3
    progress.finish(failed=False)
    now[0] = 130.0
    progress.finish(failed=True)

    assert capsys.readouterr().err.splitlines() == [
        "[gate2] 2 units, concurrency 8",
        "[gate2] 1/2 units (0 failed) | 3 calls, 0 cached | 0 in, 0 out tokens | 0m21s",
        "[gate2] 2/2 units (1 failed) | 3 calls, 0 cached | 0 in, 0 out tokens | 0m30s",
    ]
