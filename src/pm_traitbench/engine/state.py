"""Engine state: an open position and a PM's running position book.

Both are frozen dataclasses; every helper returns a new value rather than
mutating its argument, in line with the rest of the engine.
"""

from dataclasses import dataclass, replace

from pm_traitbench.engine.series import Series
from pm_traitbench.enums import Expression, Side
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Leg

_SIDE_SIGN: dict[Side, int] = {Side.BUY: 1, Side.SELL: -1}


def idea_id(n: int) -> str:
    """Format a 1-based trade idea number as its id, e.g. 7 -> 'ti_007'."""
    return f"ti_{n:03d}"


def rule_id(n: int) -> str:
    """Format a 1-based rule number as its id, e.g. 7 -> 'r_07'."""
    return f"r_{n:02d}"


@dataclass(frozen=True)
class Position:
    """An open trade idea's position: sizing, levels and per-rule counters."""

    trade_idea_id: str
    expression: Expression
    instrument_id: str
    legs: tuple[Leg, ...]
    series: Series
    side: Side
    entry_t: int
    entry_level: float
    target_level: float
    stop_level: float
    sd_h_at_entry: float
    forecast: float
    size_pct_book: float
    original_size_pct_book: float
    conviction: int
    size_rank: int
    triggers_fired: int
    consumed_rule_ids: frozenset[str]
    run_counters: tuple[tuple[str, int], ...]
    size_changed_t: int
    rolled_offset: float = 0.0
    roll_breached: bool = False

    @property
    def side_sign(self) -> int:
        """+1 for a long position, -1 for a short one."""
        return _SIDE_SIGN[self.side]

    @property
    def adverse_dir(self) -> int:
        """Direction the series moves in that is adverse to this position."""
        return -self.series.bullish_sign * self.side_sign

    def counter(self, rule_id: str) -> int:
        """Consecutive days `rule_id`'s condition has held, or 0 if never tracked."""
        for rid, n in self.run_counters:
            if rid == rule_id:
                return n
        return 0

    def with_counter(self, rule_id: str, n: int) -> "Position":
        """Return a copy with `rule_id`'s run counter set to `n`, keeping tuple order."""
        if any(rid == rule_id for rid, _ in self.run_counters):
            updated = tuple(
                (rid, n) if rid == rule_id else (rid, value) for rid, value in self.run_counters
            )
        else:
            updated = (*self.run_counters, (rule_id, n))
        return replace(self, run_counters=updated)


@dataclass(frozen=True)
class PmState:
    """A PM's position book and running idea/rule numbering."""

    pm_id: str
    positions: tuple[Position, ...]
    next_idea: int
    next_rule: int

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    @property
    def held_instruments(self) -> frozenset[str]:
        return frozenset(p.instrument_id for p in self.positions)

    def position(self, trade_idea_id: str) -> Position:
        for p in self.positions:
            if p.trade_idea_id == trade_idea_id:
                return p
        raise EngineError(f"no position for trade idea '{trade_idea_id}'")

    def add_position(self, pos: Position) -> "PmState":
        if any(p.trade_idea_id == pos.trade_idea_id for p in self.positions):
            raise EngineError(f"position for trade idea '{pos.trade_idea_id}' already exists")
        updated = tuple(sorted((*self.positions, pos), key=lambda p: p.trade_idea_id))
        return replace(self, positions=updated)

    def replace_position(self, pos: Position) -> "PmState":
        if not any(p.trade_idea_id == pos.trade_idea_id for p in self.positions):
            raise EngineError(f"no position for trade idea '{pos.trade_idea_id}'")
        updated = tuple(pos if p.trade_idea_id == pos.trade_idea_id else p for p in self.positions)
        return replace(self, positions=updated)

    def remove_position(self, trade_idea_id: str) -> "PmState":
        if not any(p.trade_idea_id == trade_idea_id for p in self.positions):
            raise EngineError(f"no position for trade idea '{trade_idea_id}'")
        updated = tuple(p for p in self.positions if p.trade_idea_id != trade_idea_id)
        return replace(self, positions=updated)
