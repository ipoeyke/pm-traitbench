"""Tests for the gate 1 opportunity-count cross-checks."""

import pytest

from pm_traitbench.engine.step import _OPPORTUNITY_KEYS
from pm_traitbench.errors import Gate1Error
from pm_traitbench.gates.gate1.checks import check_counts


def _counts(**overrides) -> dict[str, int]:
    counts = {key: 0 for key in _OPPORTUNITY_KEYS}
    counts.update(overrides)
    return counts


def test_matching_counts_pass(make_inputs):
    inputs = make_inputs(engine_counts=_counts())
    check_counts(inputs)


@pytest.mark.parametrize(
    "count_name",
    [
        "loss_side_untriggered_days",
        "entries_after_run",
        "ideas",
        "sell_day_position_days",
        "conflict_entries",
    ],
)
def test_each_mismatch_raises_gate1_error_naming_the_count(make_inputs, count_name):
    inputs = make_inputs(engine_counts=_counts(**{count_name: 1}))
    with pytest.raises(Gate1Error, match=count_name) as excinfo:
        check_counts(inputs)
    message = str(excinfo.value)
    assert inputs.pm_id in message
    assert "recomputed 0" in message
    assert "engine counted 1" in message
