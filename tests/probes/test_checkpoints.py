from datetime import date, timedelta

from pm_traitbench.enums import CheckpointLabel as L
from pm_traitbench.enums import DriftEventType
from pm_traitbench.probes.checkpoints import (
    Checkpoint,
    checkpoints_for,
    last_trading_day,
    week_of,
)
from pm_traitbench.timeline import Timeline
from tests.probes.fixtures import PM_A, TIMELINE, drift, spans, weekdays


def friday(week: int, timeline: Timeline = TIMELINE) -> date:
    return timeline.week_start(week) + timedelta(days=4)


def static_regimes():
    starts = [TIMELINE.week_start(w) for w in (1, 17, 31)]
    return spans("P", starts, TIMELINE.week_start(52) + timedelta(days=6))


def drift_in(week: int, trait_id: str = "t_09"):
    return drift(
        PM_A, TIMELINE.week_start(week) + timedelta(days=2), DriftEventType.UPDATE,
        trait_id, "a", "b",
    )  # fmt: skip


def run(drift_events=(), regimes=None, timeline=TIMELINE, days=None, post=4):
    regimes = static_regimes() if regimes is None else regimes
    days = weekdays(timeline) if days is None else days
    return checkpoints_for(drift_events, regimes, timeline, days, post)


def by_label(cps):
    return {(c.label, c.day) for c in cps}


def test_week_of_counts_from_one():
    assert week_of(TIMELINE.start, TIMELINE) == 1
    assert week_of(TIMELINE.start + timedelta(days=6), TIMELINE) == 1
    assert week_of(TIMELINE.start + timedelta(days=7), TIMELINE) == 2


def test_last_trading_day_none_for_empty_week():
    assert last_trading_day(3, TIMELINE, [TIMELINE.start]) is None


def test_static_pm_gets_time_and_regime_checkpoints():
    cps = run()
    assert by_label(cps) == {
        (L.WEEK4, friday(4)),
        (L.WEEK13, friday(13)),
        (L.REGIME_SHIFT, friday(18)),
        (L.REGIME_SHIFT, friday(32)),
        (L.WEEK52, friday(52)),
    }


def test_drift_adds_pre_and_post():
    cps = by_label(run([drift_in(20)]))
    assert (L.PRE_DRIFT, friday(19)) in cps
    assert (L.POST_DRIFT, friday(24)) in cps


def test_pre_drift_wins_shared_date_over_week4():
    cps = by_label(run([drift_in(5)]))
    assert (L.PRE_DRIFT, friday(4)) in cps
    assert (L.WEEK4, friday(4)) not in cps


def test_no_pre_drift_before_timeline():
    labels = {c.label for c in run([drift_in(1)])}
    assert L.PRE_DRIFT not in labels


def test_no_post_drift_after_timeline():
    labels = {c.label for c in run([drift_in(50)])}
    assert L.POST_DRIFT not in labels


def test_regime_shift_wins_over_week4():
    starts = [TIMELINE.week_start(w) for w in (1, 3)]
    regimes = spans("P", starts, TIMELINE.week_start(52) + timedelta(days=6))
    cps = by_label(run(regimes=regimes))
    assert (L.REGIME_SHIFT, friday(4)) in cps
    assert (L.WEEK4, friday(4)) not in cps


def test_regime_shift_on_the_final_week_covers_week52():
    starts = [TIMELINE.week_start(w) for w in (1, 51)]
    regimes = spans("P", starts, TIMELINE.week_start(52) + timedelta(days=6))
    last = run(regimes=regimes)[-1]
    assert (last.label, last.day) == (L.REGIME_SHIFT, friday(52))
    assert last.covers == {L.REGIME_SHIFT, L.WEEK52}


def test_pre_drift_sharing_a_post_drift_week_covers_both():
    cps = run([drift_in(10, "t_09"), drift_in(15, "t_10")])
    (shared,) = [c for c in cps if c.day == friday(14)]
    assert shared.label == L.PRE_DRIFT
    assert shared.covers == {L.PRE_DRIFT, L.POST_DRIFT}


def test_unshared_checkpoint_covers_its_own_label():
    assert all(c.covers == {c.label} for c in run())


def test_same_date_drift_events_give_one_pair():
    events = [drift_in(20, "t_09"), drift_in(20, "t_10")]
    cps = run(events)
    assert [c.label for c in cps].count(L.PRE_DRIFT) == 1
    assert [c.label for c in cps].count(L.POST_DRIFT) == 1


def test_short_timeline_drops_week13_and_labels_last_week():
    short = Timeline(TIMELINE.start, 12)
    regimes = spans("P", [short.start], short.week_start(12) + timedelta(days=6))
    cps = by_label(run(regimes=regimes, timeline=short))
    assert cps == {(L.WEEK4, friday(4, short)), (L.WEEK52, friday(12, short))}


def test_missing_friday_uses_thursday():
    days = [d for d in weekdays(TIMELINE) if d != friday(4)]
    cps = by_label(run(days=days))
    assert (L.WEEK4, friday(4) - timedelta(days=1)) in cps


def test_index_follows_date_order():
    cps = run([drift_in(20)])
    assert [c.index for c in cps] == list(range(len(cps)))
    assert [c.day for c in cps] == sorted(c.day for c in cps)
    assert isinstance(cps[0], Checkpoint)
