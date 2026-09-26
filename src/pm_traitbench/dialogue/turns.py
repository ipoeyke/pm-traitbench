"""The deterministic turn plan: how many turns a session runs, which PM turn carries
each planted stance, where the day's trades and the opening instruction go, and which
advisor reply carries a scripted violation.

Turn counts are drawn once per session and then raised to fit whatever the
skeleton needs, so the narrator can never be asked to stop before every
planted stance has been delivered.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from pm_traitbench.config import TurnRanges
from pm_traitbench.enums import SessionKind, StanceEntry
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import LedgerRow, Skeleton, Stance

_MAX_TURNS = 8


class Opening(StrEnum):
    """How PM turn 0 frames a session that has no stance already setting its topic."""

    SESSION_IDEAS = "session_ideas"  # decision: open on today's decision about the session's ideas
    OPEN_POSITIONS = "open_positions"  # check_in: a routine check on open positions
    MARKET_QUESTION = "market_question"  # silence: a pure market or factual question


OPENING_BY_KIND: dict[SessionKind, Opening] = {
    SessionKind.DECISION: Opening.SESSION_IDEAS,
    SessionKind.CHECK_IN: Opening.OPEN_POSITIONS,
    SessionKind.SILENCE: Opening.MARKET_QUESTION,
}


@dataclass(frozen=True)
class PmDirective:
    """What one PM turn must deliver: its stance, and, on turn 0 only, trades and opening."""

    stance: Stance | None
    trades: tuple[LedgerRow, ...]
    opening: Opening | None


@dataclass(frozen=True)
class TurnPlan:
    """A session's fixed shape: turn count, one directive per PM turn, and the violation slot."""

    n_turns: int
    pm_directives: tuple[PmDirective, ...]
    violation_advisor_index: int | None


def plan_turns(
    skeleton: Skeleton,
    day_trades: Sequence[LedgerRow],
    ranges: TurnRanges,
    rng: np.random.Generator,
) -> TurnPlan:
    """Draw a session's turn count and place its stances, trades and opening on PM turns."""
    n_turns = int(rng.choice(ranges.for_kind(skeleton.kind)))

    reactions = [s for s in skeleton.stances if s.entry == StanceEntry.REVEALED_REACTION]
    if len(reactions) > 1:
        raise DialogueError(
            f"session '{skeleton.session_id}' has {len(reactions)} revealed_reaction stances"
        )
    if len(reactions) == 1 and skeleton.advisor_violation is None:
        raise DialogueError(
            f"session '{skeleton.session_id}' has a revealed_reaction stance but no "
            "advisor_violation"
        )
    if not reactions and skeleton.advisor_violation is not None:
        raise DialogueError(
            f"session '{skeleton.session_id}' has an advisor_violation but no "
            "revealed_reaction stance"
        )

    minimum_pm_turns = max(len(skeleton.stances), 2 if reactions else 1)
    if n_turns // 2 < minimum_pm_turns:
        n_turns = 2 * minimum_pm_turns
    if n_turns > _MAX_TURNS:
        raise DialogueError(
            f"session '{skeleton.session_id}' needs {n_turns} turns, above the "
            f"{_MAX_TURNS}-turn cap"
        )
    n_pm = n_turns // 2

    reaction = reactions[0] if reactions else None
    other_stances = [s for s in skeleton.stances if s is not reaction]

    stance_by_index: dict[int, Stance] = {}
    violation_advisor_index: int | None = None
    remaining_indices = list(range(n_pm))
    if reaction is not None:
        reaction_index = int(rng.integers(1, n_pm))
        stance_by_index[reaction_index] = reaction
        violation_advisor_index = reaction_index - 1
        remaining_indices.remove(reaction_index)

    order = rng.permutation(remaining_indices)
    for stance, index in zip(other_stances, order[: len(other_stances)], strict=False):
        stance_by_index[int(index)] = stance

    directives = tuple(
        PmDirective(
            stance=stance_by_index.get(i),
            trades=tuple(day_trades) if i == 0 else (),
            opening=OPENING_BY_KIND[skeleton.kind] if i == 0 else None,
        )
        for i in range(n_pm)
    )
    return TurnPlan(
        n_turns=n_turns, pm_directives=directives, violation_advisor_index=violation_advisor_index
    )
