"""Session assembly: places one PM's planned signals on dated sessions.

A planned signal only says how many of what a PM's plan needs; a session is a
dated slot a later narrator writes into. This module draws each planned
signal's date (from its carrier, its window, or a drift event), buckets
drawn dates into sessions under the shared capacity rule, adds ledger and
filler sessions to round out the calendar, and gives every stance-bearing
signal a stable id.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from pm_traitbench.config import PlanConfig
from pm_traitbench.enums import Ownership, SessionKind, SignalMode, StanceEntry, Valence
from pm_traitbench.errors import PlanError
from pm_traitbench.signals.carriers import Carrier
from pm_traitbench.signals.inputs import PlanInputs
from pm_traitbench.signals.quotas import PlannedSignal
from pm_traitbench.tables.schema import Signal


@dataclass(frozen=True)
class PlacedSignal:
    """One planned signal after it has been given a date, a session and an id."""

    signal_id: str
    planned: PlannedSignal
    date: date
    trade_idea_id: str | None
    claim_date: date | None
    carrier: Carrier | None


@dataclass(frozen=True)
class PlannedSession:
    """One dated session and the signals whose rows sit on it."""

    session_id: str
    date: date
    kind: SessionKind
    trade_idea_ids: tuple[str, ...]
    signals: tuple[PlacedSignal, ...]
    claims: tuple[PlacedSignal, ...]


@dataclass(frozen=True)
class Assembly:
    """One PM's finished session calendar: its sessions, signal rows and run metadata."""

    sessions: tuple[PlannedSession, ...]
    signals: tuple[Signal, ...]
    warnings: tuple[str, ...]
    counts: dict[str, Any]


def session_id(pm_id: str, day: date, index: int) -> str:
    """The id of the `index`-th session on `day`; `PlanError` once `index` runs past 'z'."""
    if not 0 <= index <= 25:
        raise PlanError(f"PM '{pm_id}': ran out of session letters on {day.isoformat()}")
    letter = chr(ord("a") + index)
    return f"s_{pm_id.replace('_', '')}_{day.isoformat()}_{letter}"


@dataclass
class _Slot:
    """One stance occupying capacity in a draft session, whether a signal or a claim."""

    entry: StanceEntry
    trait_id: str
    record: "_Record"
    order: int


@dataclass
class _DraftSession:
    """A mutable, not-yet-numbered session under construction."""

    date: date
    slots: list[_Slot] = field(default_factory=list)
    next_order: int = 0
    ledger_idea_ids: set[str] = field(default_factory=set)
    forced_kind: SessionKind | None = None
    session_id: str = ""
    letter_rank: int = 0

    def has_room(self, trait_id: str, entry: StanceEntry, cap: int) -> bool:
        """Whether this session can take one more stance of `trait_id`, per the shared rule."""
        if self.forced_kind is not None:
            return False
        if len(self.slots) >= cap:
            return False
        if any(slot.trait_id == trait_id for slot in self.slots):
            return False
        if entry == StanceEntry.REVEALED_REACTION and any(
            slot.entry == StanceEntry.REVEALED_REACTION for slot in self.slots
        ):
            return False
        return True


@dataclass
class _Record:
    """One placed signal's mutable working state, before its id and rows are built."""

    planned: PlannedSignal
    carrier: Carrier | None = None
    trade_idea_id: str | None = None
    main_session: _DraftSession | None = None
    main_order: int = 0
    claim_session: _DraftSession | None = None
    claim_date: date | None = None
    signal_id: str = ""


def assemble(
    inputs: PlanInputs,
    planned: Sequence[PlannedSignal],
    pools: Mapping[str, Sequence[Carrier]],
    knobs: PlanConfig,
    rng: np.random.Generator,
) -> Assembly:
    """Place one PM's planned signals on dated sessions and build its `signals` table rows."""
    pm_id = inputs.persona.pm_id
    cap = knobs.max_signals_per_session
    day_index = {day: i for i, day in enumerate(inputs.trading_days)}
    by_date: dict[date, list[_DraftSession]] = {}
    used_carriers: set[Carrier] = set()
    records: dict[int, _Record] = {}
    warnings: list[str] = []

    def new_session(day: date) -> _DraftSession:
        session = _DraftSession(date=day)
        by_date.setdefault(day, []).append(session)
        return session

    def find_room(day: date, trait_id: str, entry: StanceEntry) -> _DraftSession | None:
        for session in by_date.get(day, ()):
            if session.has_room(trait_id, entry, cap):
                return session
        return None

    def place(session: _DraftSession, trait_id: str, entry: StanceEntry, record: _Record) -> int:
        order = session.next_order
        session.next_order += 1
        session.slots.append(_Slot(entry=entry, trait_id=trait_id, record=record, order=order))
        return order

    def remove_session(session: _DraftSession) -> None:
        lst = by_date.get(session.date)
        if lst is not None:
            lst[:] = [s for s in lst if s is not session]

    # Carrier signals, grouped by trait in trait_id order, plan order within a trait.
    by_trait: dict[str, list[tuple[int, PlannedSignal]]] = {}
    for i, s in enumerate(planned):
        if s.needs_carrier:
            by_trait.setdefault(s.trait_id, []).append((i, s))

    for trait_id in sorted(by_trait):
        pool = pools.get(trait_id, ())
        for i, s in by_trait[trait_id]:
            eligible = [
                c
                for c in pool
                if c not in used_carriers
                and s.window.first <= c.date <= s.window.last
                and (
                    s.mode != SignalMode.CONTRADICTION
                    or day_index[c.date] >= knobs.claim_lead_days[0]
                )
            ]
            if not eligible:
                warnings.append(
                    f"{pm_id} {trait_id}: {s.mode} signal dropped, no carrier in "
                    f"{s.window.first}..{s.window.last}"
                )
                continue
            carrier = eligible[rng.integers(len(eligible))]
            used_carriers.add(carrier)
            session = find_room(carrier.date, trait_id, s.entry) or new_session(carrier.date)
            record = _Record(planned=s, carrier=carrier, trade_idea_id=carrier.trade_idea_id)
            record.main_order = place(session, trait_id, s.entry, record)
            record.main_session = session
            records[i] = record

    # Claims, for every placed contradiction.
    for i in list(records):
        record = records[i]
        s = record.planned
        if s.mode != SignalMode.CONTRADICTION:
            continue
        carrier = record.carrier
        assert carrier is not None, "a placed contradiction was already given a carrier"
        lead_min, lead_max = knobs.claim_lead_days
        carrier_index = day_index[carrier.date]
        lo = max(0, carrier_index - lead_max)
        hi = carrier_index - lead_min
        candidate_days = (
            [
                day
                for day in inputs.trading_days[lo : hi + 1]
                if not inputs.is_dormant(s.trait_id, day)
            ]
            if hi >= lo
            else []
        )
        if not candidate_days:
            warnings.append(
                f"{pm_id} {s.trait_id}: contradiction signal dropped, "
                f"no claim day available before {carrier.date}"
            )
            assert record.main_session is not None
            record.main_session.slots = [
                slot for slot in record.main_session.slots if slot.record is not record
            ]
            used_carriers.discard(carrier)
            if not record.main_session.slots and not record.main_session.ledger_idea_ids:
                remove_session(record.main_session)
            del records[i]
            continue
        claim_day = candidate_days[rng.integers(len(candidate_days))]
        claim_session = find_room(claim_day, s.trait_id, StanceEntry.CLAIM) or new_session(
            claim_day
        )
        place(claim_session, s.trait_id, StanceEntry.CLAIM, record)
        record.claim_session = claim_session
        record.claim_date = claim_day

    # Every remaining non-note signal, in random order.
    packable = [
        (i, s) for i, s in enumerate(planned) if not s.needs_carrier and s.drift_date is None
    ]
    for k in rng.permutation(len(packable)):
        i, s = packable[k]
        allowed_dates = [
            day
            for day in inputs.trading_days
            if s.window.first <= day <= s.window.last and not inputs.is_dormant(s.trait_id, day)
        ]
        if not allowed_dates:
            warnings.append(
                f"{pm_id} {s.trait_id}: {s.mode} signal dropped, no allowed date in "
                f"{s.window.first}..{s.window.last}"
            )
            continue
        candidates = [
            session
            for day in allowed_dates
            for session in by_date.get(day, ())
            if session.has_room(s.trait_id, s.entry, cap)
        ]
        if candidates:
            session = candidates[rng.integers(len(candidates))]
        else:
            day = allowed_dates[rng.integers(len(allowed_dates))]
            session = new_session(day)
        record = _Record(planned=s)
        record.main_order = place(session, s.trait_id, s.entry, record)
        record.main_session = session
        records[i] = record

    # Drift notes, in drift_date order.
    drift_pairs = sorted(
        ((i, s) for i, s in enumerate(planned) if s.drift_date is not None),
        key=lambda pair: pair[1].drift_date,
    )
    for i, s in drift_pairs:
        assert s.drift_date is not None
        candidates = sorted(
            (
                session
                for lst in by_date.values()
                for session in lst
                if session.date >= s.drift_date
            ),
            key=lambda session: session.date,
        )
        session = next(
            (c for c in candidates if c.has_room(s.trait_id, s.entry, cap)),
            None,
        )
        if session is None:
            day = next(d for d in inputs.trading_days if d >= s.drift_date)
            session = new_session(day)
        record = _Record(planned=s)
        record.main_order = place(session, s.trait_id, s.entry, record)
        record.main_session = session
        records[i] = record

    # Ledger sessions, for dates whose orders reach the PM's own risk percentile.
    if inputs.ledger:
        risk_amounts = [row.risk_amount for row in inputs.ledger]
        threshold = float(np.percentile(risk_amounts, knobs.ledger_session_percentile))
        qualifying_dates = sorted(
            {row.date for row in inputs.ledger if row.risk_amount >= threshold}
        )
        for day in qualifying_dates:
            idea_ids = {
                row.trade_idea_id
                for row in inputs.ledger
                if row.date == day and row.risk_amount >= threshold
            }
            existing = [s for s in by_date.get(day, ()) if s.forced_kind is None]
            if existing:
                existing[0].ledger_idea_ids |= idea_ids
            else:
                new_session(day).ledger_idea_ids |= idea_ids

    # Filler sessions, drawn before ids are assigned, to round the calendar out.
    signal_session_count = sum(1 for lst in by_date.values() for s in lst if s.slots)
    total_session_count = sum(len(lst) for lst in by_date.values())
    needed = max(
        0,
        math.ceil(signal_session_count / knobs.signal_session_cap - 1e-9) - total_session_count,
    )
    free_days = sorted(day for day in inputs.trading_days if not by_date.get(day))
    k = min(needed, len(free_days))
    if k > 0:
        chosen = rng.choice(len(free_days), size=k, replace=False)
        for day in sorted(free_days[j] for j in chosen):
            kind = (
                SessionKind.SILENCE
                if rng.random() < knobs.filler_silence_share
                else SessionKind.CHECK_IN
            )
            new_session(day).forced_kind = kind
    if needed > len(free_days):
        warnings.append(
            f"{pm_id}: signal_session_cap cannot be met, needed {needed} filler session(s) "
            f"but only {len(free_days)} free day(s) available"
        )

    # Session and signal ids, assigned once every session and placement is settled.
    for day, lst in by_date.items():
        for idx, session in enumerate(lst):
            session.session_id = session_id(pm_id, day, idx)
            session.letter_rank = idx

    ordered_records = sorted(
        records.values(),
        key=lambda r: (r.main_session.date, r.main_session.letter_rank, r.main_order),
    )
    for n, record in enumerate(ordered_records, start=1):
        record.signal_id = f"sg_{n:03d}"

    placed_by_record: dict[int, PlacedSignal] = {}
    for record in ordered_records:
        assert record.main_session is not None
        placed_by_record[id(record)] = PlacedSignal(
            signal_id=record.signal_id,
            planned=record.planned,
            date=record.main_session.date,
            trade_idea_id=record.trade_idea_id,
            claim_date=record.claim_date,
            carrier=record.carrier,
        )

    signal_rows = [
        Signal(
            signal_id=record.signal_id,
            pm_id=pm_id,
            session_id=record.main_session.session_id,
            date=record.main_session.date,
            trait_id=record.planned.trait_id,
            mode=record.planned.mode,
            trade_idea_id=record.trade_idea_id,
            valence=record.planned.valence,
            ownership=record.planned.ownership,
            third_party_value=record.planned.third_party_value,
            claim_session_id=record.claim_session.session_id if record.claim_session else None,
        )
        for record in ordered_records
    ]
    signal_rows.sort(key=lambda row: row.signal_id)

    sessions_out = []
    for lst in by_date.values():
        for session in lst:
            signal_slots = [slot for slot in session.slots if slot.entry != StanceEntry.CLAIM]
            claim_slots = [slot for slot in session.slots if slot.entry == StanceEntry.CLAIM]
            carrier_ids = {
                slot.record.trade_idea_id for slot in signal_slots if slot.record.trade_idea_id
            }
            trade_idea_ids = tuple(sorted(carrier_ids | session.ledger_idea_ids))
            if session.forced_kind is not None:
                kind = session.forced_kind
            else:
                kind = (
                    SessionKind.DECISION
                    if any(
                        row.date == session.date and row.trade_idea_id in trade_idea_ids
                        for row in inputs.ledger
                    )
                    else SessionKind.CHECK_IN
                )
            sessions_out.append(
                PlannedSession(
                    session_id=session.session_id,
                    date=session.date,
                    kind=kind,
                    trade_idea_ids=trade_idea_ids,
                    signals=tuple(placed_by_record[id(slot.record)] for slot in signal_slots),
                    claims=tuple(placed_by_record[id(slot.record)] for slot in claim_slots),
                )
            )
    sessions_out.sort(key=lambda session: session.session_id)

    counts = _counts(inputs, planned, ordered_records, signal_rows, sessions_out)

    return Assembly(
        sessions=tuple(sessions_out),
        signals=tuple(signal_rows),
        warnings=tuple(warnings),
        counts=counts,
    )


def _counts(
    inputs: PlanInputs,
    planned: Sequence[PlannedSignal],
    ordered_records: list[_Record],
    signal_rows: list[Signal],
    sessions_out: list[PlannedSession],
) -> dict[str, Any]:
    """JSON-safe run counts summarising one PM's assembled calendar."""
    signals_by_trait_mode: dict[str, dict[str, int]] = {}
    for row in signal_rows:
        by_mode = signals_by_trait_mode.setdefault(row.trait_id, {})
        by_mode[row.mode.value] = by_mode.get(row.mode.value, 0) + 1
    signals_by_trait_mode = {
        trait_id: dict(sorted(signals_by_trait_mode[trait_id].items()))
        for trait_id in sorted(signals_by_trait_mode)
    }

    revealed_planned: dict[str, int] = {}
    for s in planned:
        if s.needs_carrier:
            revealed_planned[s.trait_id] = revealed_planned.get(s.trait_id, 0) + 1
    revealed_placed: dict[str, int] = dict.fromkeys(revealed_planned, 0)
    for record in ordered_records:
        if record.planned.needs_carrier:
            revealed_placed[record.planned.trait_id] = (
                revealed_placed.get(record.planned.trait_id, 0) + 1
            )
    revealed_planned = dict(sorted(revealed_planned.items()))
    revealed_placed = dict(sorted(revealed_placed.items()))

    windows_by_trait: dict[str, dict[Any, int]] = {}
    for record in ordered_records:
        p = record.planned
        if p.valence == Valence.CONFIRM and p.ownership == Ownership.SELF and p.drift_date is None:
            by_window = windows_by_trait.setdefault(p.trait_id, {})
            by_window[p.window] = by_window.get(p.window, 0) + 1
    segments: dict[str, list[int]] = {}
    for trait_id in sorted(windows_by_trait):
        by_window = windows_by_trait[trait_id]
        if len(by_window) > 1:
            ordered_windows = sorted(by_window, key=lambda w: w.first)
            segments[trait_id] = [by_window[w] for w in ordered_windows]

    sessions_by_kind: dict[str, int] = {}
    for session in sessions_out:
        sessions_by_kind[session.kind.value] = sessions_by_kind.get(session.kind.value, 0) + 1
    sessions_by_kind = dict(sorted(sessions_by_kind.items()))

    signal_session_count = sum(1 for s in sessions_out if s.signals or s.claims)
    total_session_count = len(sessions_out)
    share = round(signal_session_count / total_session_count, 4) if total_session_count else 0.0

    return {
        "signals_by_trait_mode": signals_by_trait_mode,
        "revealed_planned": revealed_planned,
        "revealed_placed": revealed_placed,
        "segments": segments,
        "sessions_by_kind": sessions_by_kind,
        "signal_session_share": share,
    }
