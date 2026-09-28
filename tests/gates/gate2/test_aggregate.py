"""Tests for gate 2's aggregation, exact tests and verdicts."""

from itertools import product

import pytest

from pm_traitbench.config import BIAS_PARAMS, Gate2Config
from pm_traitbench.enums import (
    AssetClass,
    DriftStatus,
    Gate2Slice,
    Gate2Verdict,
    Kind,
    Ownership,
    SignalMode,
    Typicality,
    Valence,
)
from pm_traitbench.gates.gate2.aggregate import (
    POOLED_PREFERENCES_ID,
    blocking_failures,
    blocking_id,
    build_cells,
    fisher_upper_p,
    insufficient_blocking,
    poisson_binomial_upper_p,
)
from pm_traitbench.tables.schema import Gate2PmRow, Gate2SignalRow, Gate2TraitRow

BIAS_PARAM = BIAS_PARAMS[0]


def _bias_row(pm_id: str, truth_active: bool, predicted_active: bool, param: str = BIAS_PARAM):
    """A `Gate2TraitRow` for a bias, correct exactly when truth and prediction agree."""
    return Gate2TraitRow(
        pm_id=pm_id,
        param=param,
        trait_id="t_01",
        kind=Kind.BIAS,
        truth_active=truth_active,
        truth_value=None,
        predicted_active=predicted_active,
        predicted_value=None,
        correct=truth_active == predicted_active,
        cited_session_ids=(),
        false_attribution_ids=(),
    )


def _pref_row(pm_id: str, param: str, truth_value: str | None, predicted_value: str | None):
    """A `Gate2TraitRow` for a preference, correct exactly when the values match."""
    return Gate2TraitRow(
        pm_id=pm_id,
        param=param,
        trait_id=None,
        kind=Kind.PREFERENCE,
        truth_active=None,
        truth_value=truth_value,
        predicted_active=None,
        predicted_value=predicted_value,
        correct=truth_value == predicted_value,
        cited_session_ids=(),
        false_attribution_ids=(),
    )


def _signal_row(
    pm_id: str,
    param: str,
    mode: SignalMode,
    kind: Kind = Kind.BIAS,
    valence: Valence = Valence.CONFIRM,
    ownership: Ownership = Ownership.SELF,
    pre_update: bool = False,
    recovered: bool = False,
    signal_id: str = "sg_001",
):
    """A `Gate2SignalRow`; `recovered` implies `cited`."""
    return Gate2SignalRow(
        pm_id=pm_id,
        signal_id=signal_id,
        session_id=f"s_{pm_id.replace('_', '')}_2026-01-05_a",
        trait_id="t_01",
        param=param,
        kind=kind,
        mode=mode,
        valence=valence,
        ownership=ownership,
        pre_update=pre_update,
        cited=recovered,
        recovered=recovered,
        classified=False,
        kind_predicted=None,
        kind_ok=None,
    )


def _pm_row(pm_id: str, asset_class: AssetClass = AssetClass.EQUITIES):
    """A `Gate2PmRow` with placeholder counts, for slice-membership tests."""
    return Gate2PmRow(
        pm_id=pm_id,
        asset_class=asset_class,
        typicality=Typicality.TYPICAL,
        drift=DriftStatus.STATIC,
        seed="seed_a",
        sessions=1,
        context_chars=100,
        ngram_containment=0.0,
        biases_correct=0,
        preferences_held=0,
        preferences_correct=0,
        stated_signals=0,
        stated_kind_ok=0,
    )


def _row(cells, slice_: Gate2Slice, slice_value: str, param: str | None) -> object:
    matches = [
        c for c in cells if c.slice == slice_ and c.slice_value == slice_value and c.param == param
    ]
    assert len(matches) == 1, f"expected exactly one row for {slice_}/{slice_value}/{param}"
    return matches[0]


def test_fisher_upper_p_matches_hand_values():
    assert fisher_upper_p(3, 3, 9, 3) == pytest.approx(1 / 220, abs=1e-12)
    assert fisher_upper_p(2, 3, 9, 2) == pytest.approx(10 / 220, abs=1e-12)
    assert fisher_upper_p(1, 3, 9, 4) > 0.5


def test_poisson_binomial_matches_brute_force_on_six_pairs():
    chances = [0.5, 0.25, 0.25, 1 / 3, 0.25, 0.5]
    n = len(chances)
    brute = [0.0] * (n + 1)
    for outcome in product([0, 1], repeat=n):
        weight = 1.0
        for bit, p in zip(outcome, chances, strict=True):
            weight *= p if bit else (1 - p)
        brute[sum(outcome)] += weight
    for hits in range(n + 1):
        expected = sum(brute[hits:])
        assert poisson_binomial_upper_p(chances, hits) == pytest.approx(expected, abs=1e-12)
    assert poisson_binomial_upper_p(chances, 0) == pytest.approx(1.0, abs=1e-12)
    assert poisson_binomial_upper_p(chances, n + 1) == pytest.approx(0.0, abs=1e-12)


def test_build_cells_bias_row_verdicts():
    config = Gate2Config()
    pass_rows = [_bias_row(f"pm_{i:03d}", True, True) for i in range(3)] + [
        _bias_row(f"pm_{i:03d}", False, False) for i in range(3, 12)
    ]
    cells = build_cells(pass_rows, [], [], {}, config)
    row = _row(cells, Gate2Slice.ALL, "all", BIAS_PARAM)
    assert row.verdict == Gate2Verdict.PASS
    assert row.rate == pytest.approx(1.0)
    assert row.blocking is True
    assert row.p == pytest.approx(1 / 220, abs=1e-12)

    insufficient_rows = [_bias_row(f"pm_{i:03d}", True, True) for i in range(3)] + [
        _bias_row("pm_100", False, False)
    ]
    cells = build_cells(insufficient_rows, [], [], {}, config)
    row = _row(cells, Gate2Slice.ALL, "all", BIAS_PARAM)
    assert row.verdict == Gate2Verdict.INSUFFICIENT
    assert row.p is None

    fail_rows = (
        [_bias_row("pm_000", True, True)]
        + [_bias_row(f"pm_{i:03d}", True, False) for i in range(1, 3)]
        + [_bias_row(f"pm_{i:03d}", False, True) for i in range(3, 7)]
        + [_bias_row(f"pm_{i:03d}", False, False) for i in range(7, 12)]
    )
    cells = build_cells(fail_rows, [], [], {}, config)
    row = _row(cells, Gate2Slice.ALL, "all", BIAS_PARAM)
    assert row.verdict == Gate2Verdict.FAIL


def test_build_cells_pooled_preference_row_and_per_param_rows():
    config = Gate2Config()
    k_by_param = {"response_format": 2, "register": 3}
    rows = [
        _pref_row("pm_001", "response_format", "short bullets", "short bullets"),
        _pref_row("pm_002", "response_format", "short bullets", "a table with columns"),
        _pref_row("pm_003", "response_format", None, None),
        _pref_row("pm_004", "response_format", None, "short bullets"),
        _pref_row("pm_005", "register", "blunt", "blunt"),
        _pref_row("pm_006", "register", "formal", "formal"),
    ]
    cells = build_cells(rows, [], [], k_by_param, config)

    pooled = _row(cells, Gate2Slice.ALL, "all", None)
    assert pooled.n == 4
    assert pooled.n_positive == 3
    assert pooled.rate == pytest.approx(0.75)
    assert pooled.chance == pytest.approx((1 / 2 + 1 / 2 + 1 / 3 + 1 / 3) / 4)
    assert pooled.blocking is True
    assert pooled.p == pytest.approx(poisson_binomial_upper_p([0.5, 0.5, 1 / 3, 1 / 3], 3))

    per_param = _row(cells, Gate2Slice.ALL, "all", "response_format")
    assert per_param.n == 2
    assert per_param.n_positive == 1
    assert per_param.chance == pytest.approx(0.5)
    assert per_param.blocking is False

    held = _row(cells, Gate2Slice.HELD, "all", "response_format")
    assert held.n == 4
    assert held.n_positive == 3
    assert held.blocking is False
    assert held.p is not None

    kind_pref = _row(cells, Gate2Slice.KIND, "preference", None)
    assert kind_pref.n == 4
    assert kind_pref.n_positive == 3
    assert kind_pref.p is None
    assert kind_pref.blocking is False


def test_mode_rows_exclude_distractors_and_pre_update_signals():
    config = Gate2Config()
    signals = [
        _signal_row("pm_001", "loss_aversion_lambda", SignalMode.STATED, recovered=True),
        _signal_row("pm_001", "loss_aversion_lambda", SignalMode.STATED, recovered=True),
        _signal_row("pm_001", "loss_aversion_lambda", SignalMode.STATED, recovered=False),
        _signal_row(
            "pm_001",
            "loss_aversion_lambda",
            SignalMode.STATED,
            valence=Valence.RETRACTED,
            recovered=False,
        ),
        _signal_row(
            "pm_001",
            "loss_aversion_lambda",
            SignalMode.STATED,
            ownership=Ownership.COLLEAGUE,
            recovered=False,
        ),
        _signal_row(
            "pm_001",
            "response_format",
            SignalMode.STATED,
            kind=Kind.PREFERENCE,
            pre_update=True,
            recovered=False,
        ),
    ]
    cells = build_cells([], signals, [], {}, config)
    row = _row(cells, Gate2Slice.MODE, SignalMode.STATED.value, None)
    assert row.n == 3
    assert row.n_positive == 2
    assert row.rate == pytest.approx(2 / 3)
    assert row.p is None
    assert row.chance is None
    assert row.verdict is None

    param_row = _row(cells, Gate2Slice.MODE, SignalMode.STATED.value, "loss_aversion_lambda")
    assert param_row.n == 3
    assert param_row.n_positive == 2


def test_slice_rows_restrict_to_cell_pms_and_never_block():
    config = Gate2Config()
    trait_rows = [
        _bias_row("pm_001", True, True),
        _bias_row("pm_002", True, True),
        _bias_row("pm_003", True, True),
        _bias_row("pm_004", False, False),
        _bias_row("pm_005", False, True),
    ]
    pm_rows = [
        _pm_row("pm_001", AssetClass.EQUITIES),
        _pm_row("pm_002", AssetClass.EQUITIES),
        _pm_row("pm_003", AssetClass.EQUITIES),
        _pm_row("pm_004", AssetClass.RATES_CREDIT),
        _pm_row("pm_005", AssetClass.RATES_CREDIT),
    ]
    cells = build_cells(trait_rows, [], pm_rows, {}, config)

    equities_row = _row(cells, Gate2Slice.ASSET_CLASS, AssetClass.EQUITIES.value, BIAS_PARAM)
    assert equities_row.n == 3
    assert equities_row.n_positive == 3
    assert equities_row.blocking is False

    rates_row = _row(cells, Gate2Slice.ASSET_CLASS, AssetClass.RATES_CREDIT.value, BIAS_PARAM)
    assert rates_row.n == 2
    assert rates_row.n_positive == 1
    assert rates_row.blocking is False

    for cell in cells:
        if cell.slice != Gate2Slice.ALL:
            assert cell.blocking is False


def test_blocking_ids():
    config = Gate2Config()
    trait_rows = [_bias_row(f"pm_{i:03d}", True, True) for i in range(3)]
    cells = build_cells(trait_rows, [], [], {}, config)
    bias_cell = _row(cells, Gate2Slice.ALL, "all", BIAS_PARAM)
    assert blocking_id(bias_cell) == f"all/{BIAS_PARAM}"

    pooled_cell = _row(cells, Gate2Slice.ALL, "all", None)
    assert blocking_id(pooled_cell) == POOLED_PREFERENCES_ID == "all/preferences"

    insufficient_cells = build_cells([_bias_row("pm_001", True, True)], [], [], {}, config)
    failures = blocking_failures(insufficient_cells)
    insufficient = insufficient_blocking(insufficient_cells)
    assert failures == sorted(failures)
    assert insufficient == sorted(insufficient)
    assert f"all/{BIAS_PARAM}" in insufficient
    assert f"all/{BIAS_PARAM}" in failures
