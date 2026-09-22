"""Tests for street consensus and positioning."""

import functools

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import Family, InstrumentKind, Positioning, StreetView, Tenor
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.consensus import build_consensus
from pm_traitbench.market.processes.common import ProcessOutput
from pm_traitbench.market.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import Instrument

_ROOT = 0


def _rng_for(root: int):
    return functools.partial(stream, root, "market")


def _equity(instrument_id: str = "EQ-0001", beta: float = 1.0) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        family=Family.EQUITIES,
        kind=InstrumentKind.EQUITY,
        name=instrument_id,
        currency="USD",
        sector="sector_01",
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=beta,
        expiry_rule=None,
    )


def _curve(instrument_id: str = "RT-USD", currency: str = "USD") -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        family=Family.RATES,
        kind=InstrumentKind.SOVEREIGN_CURVE,
        name=f"{currency} sovereign curve",
        currency=currency,
        sector=None,
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=None,
        expiry_rule=None,
    )


def _trending_equity_config(burn_in_days: int, flips_per_year: float = 0.0):
    # Calendar stays at its default 52 weeks: drift config's own week ranges are
    # validated against it, and 260 horizon days costs nothing in these tests.
    return Config.model_validate(
        {
            "market": {
                "burn_in_days": burn_in_days,
                "consensus": {"flips_per_instrument_year": flips_per_year},
            },
        }
    )


def _log_linear_prices(n_days: int, rate: float, base: float = 100.0) -> np.ndarray:
    return base * np.exp(rate * np.arange(n_days))


class _ScriptedFlipRng:
    """Stub forcing a scripted Poisson count and choice of flip day indices."""

    def __init__(self, count: int, days: list[int]) -> None:
        self._count = count
        self._days = days

    def poisson(self, lam: float) -> int:
        return self._count

    def choice(self, n: int, size: int, replace: bool) -> np.ndarray:
        return np.array(self._days[:size], dtype=int)


def _scripted_flip_rng_for(root: int, flip_days: list[int]):
    def rng_for(seed: str, *keys: str):
        if keys and keys[0] == "flips":
            return _ScriptedFlipRng(len(flip_days), flip_days)
        return stream(root, "market", seed, *keys)

    return rng_for


def test_street_score_changes_only_on_update_days() -> None:
    config = _trending_equity_config(burn_in_days=5)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = _equity()
    output = ProcessOutput(prices={"EQ-0001": _log_linear_prices(axis.n_days, rate=0.01)})

    result = build_consensus([instrument], axis, output, {}, config.market, _rng_for(_ROOT), "A")
    score = result.street_score["EQ-0001"]
    revision_weekday = config.market.consensus.revision_weekday

    changed_on_revision = False
    for t in range(1, axis.n_days):
        if axis.dates[t].weekday() == revision_weekday:
            if score[t] != score[t - 1]:
                changed_on_revision = True
        else:
            assert score[t] == score[t - 1]
    assert changed_on_revision


def test_positioning_pct_changes_only_on_report_days() -> None:
    config = _trending_equity_config(burn_in_days=5)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = _equity()
    output = ProcessOutput(prices={"EQ-0001": _log_linear_prices(axis.n_days, rate=0.01)})

    result = build_consensus([instrument], axis, output, {}, config.market, _rng_for(_ROOT), "A")
    pct = result.positioning_pct["EQ-0001"]
    report_weekday = config.market.consensus.report_weekday

    changed_on_report = False
    for t in range(1, axis.n_days):
        if axis.dates[t].weekday() == report_weekday:
            if pct[t] != pct[t - 1]:
                changed_on_report = True
        else:
            assert pct[t] == pct[t - 1]
    assert changed_on_report


def test_flip_day_formula_and_flip_rows_never_in_burn_in() -> None:
    config = _trending_equity_config(burn_in_days=5)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = _equity()
    output = ProcessOutput(prices={"EQ-0001": _log_linear_prices(axis.n_days, rate=0.0)})

    revision_weekday = config.market.consensus.revision_weekday
    horizon_days = axis.n_days - axis.n_burn
    non_revision = [
        t for t in range(horizon_days) if axis.dates[axis.n_burn + t].weekday() != revision_weekday
    ]
    flip_days = non_revision[:3]
    assert len(flip_days) == 3

    rng_for = _scripted_flip_rng_for(_ROOT, flip_days)
    result = build_consensus([instrument], axis, output, {}, config.market, rng_for, "A")
    score = result.street_score["EQ-0001"]
    threshold = config.market.consensus.view_threshold

    assert result.drawn_flips["EQ-0001"] == len(flip_days)
    flip_rows = [row for row in result.flips if row.instrument_id == "EQ-0001"]
    assert len(flip_rows) == len(flip_days)
    for row in flip_rows:
        assert row.date >= axis.dates[axis.n_burn]
        assert row.surprise is None
        assert row.affected == "equities"

    for row, d in zip(sorted(flip_rows, key=lambda r: r.date), flip_days, strict=True):
        assert row.date == axis.dates[axis.n_burn + d]

    for t in flip_days:
        axis_t = axis.n_burn + t
        prev = score[axis_t - 1]
        expected_sign = 1.0 if prev >= 0 else -1.0
        assert score[axis_t] == -expected_sign * 2 * threshold


def test_flips_do_not_touch_positioning() -> None:
    config = _trending_equity_config(burn_in_days=5)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = _equity()
    output = ProcessOutput(prices={"EQ-0001": _log_linear_prices(axis.n_days, rate=0.01)})

    horizon_days = axis.n_days - axis.n_burn
    flip_days = [1, horizon_days - 2]
    flipped_rng_for = _scripted_flip_rng_for(_ROOT, flip_days)

    with_flips = build_consensus(
        [instrument], axis, output, {}, config.market, flipped_rng_for, "A"
    )
    without_flips = build_consensus(
        [instrument], axis, output, {}, config.market, _rng_for(_ROOT), "A"
    )

    assert with_flips.drawn_flips["EQ-0001"] == 2
    assert without_flips.drawn_flips["EQ-0001"] == 0
    np.testing.assert_array_equal(
        with_flips.positioning_pct["EQ-0001"], without_flips.positioning_pct["EQ-0001"]
    )


def test_uptrend_rises_above_thresholds_and_labels_agree() -> None:
    config = _trending_equity_config(burn_in_days=0)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = _equity()
    output = ProcessOutput(prices={"EQ-0001": _log_linear_prices(axis.n_days, rate=0.01)})

    result = build_consensus([instrument], axis, output, {}, config.market, _rng_for(_ROOT), "A")
    threshold = config.market.consensus.view_threshold
    lo, hi = config.market.consensus.positioning_thresholds

    for row in result.rows:
        if row.street_score > threshold:
            assert row.street_view == StreetView.OVERWEIGHT
        elif row.street_score < -threshold:
            assert row.street_view == StreetView.UNDERWEIGHT
        else:
            assert row.street_view == StreetView.NEUTRAL
        if row.positioning_pct > hi:
            assert row.positioning == Positioning.CROWDED_LONG
        elif row.positioning_pct < lo:
            assert row.positioning == Positioning.CROWDED_SHORT
        else:
            assert row.positioning == Positioning.NEUTRAL

    last = result.rows[-1]
    assert last.street_score > threshold
    assert last.street_view == StreetView.OVERWEIGHT
    assert last.positioning_pct > hi
    assert last.positioning == Positioning.CROWDED_LONG


def test_first_horizon_row_nonzero_after_trend_through_burn_in() -> None:
    config = _trending_equity_config(burn_in_days=10)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = _equity()
    output = ProcessOutput(prices={"EQ-0001": _log_linear_prices(axis.n_days, rate=0.01)})

    result = build_consensus([instrument], axis, output, {}, config.market, _rng_for(_ROOT), "A")
    first_row = next(row for row in result.rows if row.date == axis.dates[axis.n_burn])
    assert first_row.street_score != 0.0


def test_falling_curve_gives_positive_score() -> None:
    config = _trending_equity_config(burn_in_days=10)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = _curve()
    yields = 4.0 - 0.01 * np.arange(axis.n_days)
    output = ProcessOutput(curves={("RT-USD", Tenor.Y10): yields})

    result = build_consensus([instrument], axis, output, {}, config.market, _rng_for(_ROOT), "A")
    assert result.street_score["RT-USD"][-1] > 0.0


def test_rows_cover_every_instrument_and_day_and_seeds_differ_only_by_flip_placement() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = build_universe(config, stream(_ROOT, "market", "universe"))

    output = ProcessOutput(
        prices={
            i.instrument_id: np.full(axis.n_days, 100.0)
            for i in instruments
            if i.kind != InstrumentKind.SOVEREIGN_CURVE
        },
        curves={
            (i.instrument_id, Tenor.Y10): np.full(axis.n_days, 4.0)
            for i in instruments
            if i.kind == InstrumentKind.SOVEREIGN_CURVE
        },
    )

    rng_for = _rng_for(_ROOT)
    result_a = build_consensus(instruments, axis, output, {}, config.market, rng_for, "A")
    result_b = build_consensus(instruments, axis, output, {}, config.market, rng_for, "B")

    horizon_dates = axis.dates[axis.n_burn :]
    expected_keys = {(i.instrument_id, d) for i in instruments for d in horizon_dates}
    actual_keys = {(row.instrument_id, row.date) for row in result_a.rows}
    assert actual_keys == expected_keys
    assert len(result_a.rows) == len(instruments) * len(horizon_dates)

    assert set(result_a.drawn_flips) == {i.instrument_id for i in instruments}
    assert set(result_b.drawn_flips) == {i.instrument_id for i in instruments}
    assert result_a.drawn_flips != result_b.drawn_flips

    for instrument in instruments:
        instrument_id = instrument.instrument_id
        dates_a = {row.date for row in result_a.flips if row.instrument_id == instrument_id}
        dates_b = {row.date for row in result_b.flips if row.instrument_id == instrument_id}
        if dates_a == dates_b:
            np.testing.assert_array_equal(
                result_a.street_score[instrument_id], result_b.street_score[instrument_id]
            )


def test_event_day_updates_score_and_flip_wins_over_same_day_event() -> None:
    config = _trending_equity_config(burn_in_days=5)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = _equity()
    output = ProcessOutput(prices={"EQ-0001": _log_linear_prices(axis.n_days, rate=0.01)})
    threshold = config.market.consensus.view_threshold
    revision_weekday = config.market.consensus.revision_weekday
    horizon_days = axis.n_days - axis.n_burn

    non_revision_days = [
        t for t in range(horizon_days) if axis.dates[axis.n_burn + t].weekday() != revision_weekday
    ]
    event_day = non_revision_days[0]
    event_axis_idx = axis.n_burn + event_day
    event_days = {"EQ-0001": {event_axis_idx}}

    result = build_consensus(
        [instrument], axis, output, event_days, config.market, _rng_for(_ROOT), "A"
    )
    score = result.street_score["EQ-0001"]
    assert score[event_axis_idx] != score[event_axis_idx - 1]

    revision_day = next(
        t for t in range(horizon_days) if axis.dates[axis.n_burn + t].weekday() == revision_weekday
    )
    revision_axis_idx = axis.n_burn + revision_day
    coincident_event_days = {"EQ-0001": {revision_axis_idx}}
    flipped_rng_for = _scripted_flip_rng_for(_ROOT, [revision_day])

    flipped_result = build_consensus(
        [instrument], axis, output, coincident_event_days, config.market, flipped_rng_for, "A"
    )
    flipped_score = flipped_result.street_score["EQ-0001"]
    prev = flipped_score[revision_axis_idx - 1]
    expected_sign = 1.0 if prev >= 0 else -1.0
    assert flipped_score[revision_axis_idx] == -expected_sign * 2 * threshold
