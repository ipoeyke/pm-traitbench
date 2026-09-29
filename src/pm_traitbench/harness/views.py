"""Allow-list views of corpus rows for a system under test, and per-PM replay plans.

Every public object copies named fields only, so a hidden column never reaches a
system by omission from a deny-list.
"""

import datetime
import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from pm_traitbench.enums import RuleScope
from pm_traitbench.errors import HarnessError
from pm_traitbench.gates.gate2.transcript import session_order_key
from pm_traitbench.harness.protocol import PublicProbe, PublicProfile, PublicSession
from pm_traitbench.tables.schema import Persona, ProbeRow, Rule, Session


def public_profile(persona: Persona, rules: Sequence[Rule]) -> PublicProfile:
    """The persona's mandate, self-description and own PM-scope rules."""
    own = sorted(
        (r for r in rules if r.pm_id == persona.pm_id and r.scope == RuleScope.PM),
        key=lambda r: r.rule_id,
    )
    return PublicProfile(
        pm_id=persona.pm_id,
        mandate=persona.mandate,
        self_description=persona.stated_profile.self_description,
        rules=tuple(own),
    )


def public_session(session: Session, rules: Sequence[Rule]) -> PublicSession:
    """The transcript plus idea-scope rules of the ideas it discusses.

    Idea ids are not copied: they carry no memory content and the rules already name their ideas.
    """
    ideas = set(session.trade_idea_ids)
    discussed = sorted(
        (
            r
            for r in rules
            if r.pm_id == session.pm_id and r.scope == RuleScope.IDEA and r.trade_idea_id in ideas
        ),
        key=lambda r: r.rule_id,
    )
    return PublicSession(
        session_id=session.session_id,
        date=session.date,
        turns=session.turns,
        idea_rules=tuple(discussed),
    )


def opaque_probe_id(root_seed: int, probe_id: str) -> str:
    """A keyed hash of the probe id, so the id a system sees reveals no emission order."""
    digest = hashlib.sha256(f"{root_seed}:{probe_id}".encode()).hexdigest()
    return f"q_{digest[:16]}"


def public_probe(row: ProbeRow, root_seed: int) -> PublicProbe:
    """The probe's question and non-null options; trait, type, slice and answer stay hidden.

    The id is opaque: corpus ids are numbered in construction order, which tracks the answer.
    """
    options = (row.option_a, row.option_b, row.option_c, row.option_d)
    return PublicProbe(
        probe_id=opaque_probe_id(root_seed, row.probe_id),
        form=row.form,
        question=row.question,
        options=tuple(text for text in options if text is not None),
    )


@dataclass(frozen=True)
class Checkpoint:
    """The probes asked on one checkpoint day."""

    day: datetime.date
    probes: tuple[ProbeRow, ...]


@dataclass(frozen=True)
class PmReplay:
    """Everything needed to replay one PM: profile, sessions and checkpoints in time order."""

    profile: PublicProfile
    sessions: tuple[PublicSession, ...]
    checkpoints: tuple[Checkpoint, ...]


def pm_replays(
    personas: Sequence[Persona],
    rules: Sequence[Rule],
    sessions: Sequence[Session],
    probes: Sequence[ProbeRow],
) -> dict[str, PmReplay]:
    """One replay per PM that has probes, keyed and ordered by pm_id."""
    persona_by_pm = {p.pm_id: p for p in personas}
    probes_by_pm: dict[str, list[ProbeRow]] = defaultdict(list)
    for probe in probes:
        probes_by_pm[probe.pm_id].append(probe)
    sessions_by_pm: dict[str, list[Session]] = defaultdict(list)
    for session in sessions:
        sessions_by_pm[session.pm_id].append(session)

    replays: dict[str, PmReplay] = {}
    for pm_id in sorted(probes_by_pm):
        persona = persona_by_pm.get(pm_id)
        if persona is None:
            raise HarnessError(f"PM '{pm_id}' has probes but no persona row")
        by_day: dict[datetime.date, list[ProbeRow]] = defaultdict(list)
        for probe in probes_by_pm[pm_id]:
            by_day[probe.checkpoint_date].append(probe)
        ordered = sorted(sessions_by_pm[pm_id], key=session_order_key)
        replays[pm_id] = PmReplay(
            profile=public_profile(persona, rules),
            sessions=tuple(public_session(s, rules) for s in ordered),
            checkpoints=tuple(
                Checkpoint(day, tuple(sorted(by_day[day], key=lambda p: p.probe_id)))
                for day in sorted(by_day)
            ),
        )
    return replays
