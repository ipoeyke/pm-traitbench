"""Per-session context: assembles one skeleton's rows into everything the narrator and
advisor prompt builders need, and filters which PMs a dialogue run covers.

Nothing here renders text; it only selects and orders the rows a session's
prompts will draw from, so ordering and filtering logic is tested once
rather than re-derived inside every prompt builder.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from pm_traitbench.catalogues.models import Catalogue, Voice
from pm_traitbench.config import Config, PmFilter
from pm_traitbench.dialogue.tools import MarketLookup
from pm_traitbench.dialogue.turns import TurnPlan, plan_turns
from pm_traitbench.enums import (
    AssetClass,
    DriftStatus,
    InstrumentKind,
    PositionAction,
    SessionKind,
    Side,
    Tenor,
)
from pm_traitbench.errors import DialogueError
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import (
    DriftEvent,
    Idea,
    Instrument,
    LedgerRow,
    Persona,
    PositionDay,
    Rule,
    RuleScope,
    Skeleton,
    Trait,
)

_ASSET_CLASS_KINDS: dict[AssetClass, tuple[InstrumentKind, ...]] = {
    AssetClass.EQUITIES: (InstrumentKind.EQUITY,),
    AssetClass.RATES_CREDIT: (InstrumentKind.SOVEREIGN_CURVE, InstrumentKind.CREDIT_ISSUER),
    AssetClass.COMMODITIES: (InstrumentKind.COMMODITY,),
}


# A ledger row's identity within one PM-day: the ledger table's key minus (pm_id, date).
type TradeKey = tuple[str, str, Tenor | None, Side]

_KIND_BY_ACTION: dict[PositionAction, str] = {
    PositionAction.ADD: "adds to the position",
    PositionAction.TRIM: "trims the position",
    PositionAction.CUT: "closes the whole position",
    PositionAction.EXIT: "closes the whole position",
    PositionAction.ROLL: "rolls the position",
}
_OPENS = "opens the position"
_HORIZON_END = "at the end of the horizon"
_OWN_CALL = "on your own call"


@dataclass(frozen=True)
class TradeNote:
    """What one day trade does to its position and what drove it, in the narrator's words."""

    kind: str
    trigger: str | None


def trade_key(row: LedgerRow) -> TradeKey:
    return (row.trade_idea_id, row.instrument_id, row.tenor, row.side)


def annotate_day_trades(
    day_trades: Sequence[LedgerRow],
    ideas: Mapping[str, Idea],
    position_days: Sequence[PositionDay],
    rules: Sequence[Rule],
    horizon_end: date | None,
) -> dict[TradeKey, TradeNote]:
    """A `TradeNote` per day trade, so the narrator never guesses whether a sell is a trim.

    The kind comes from the idea's entry date, else its `position_days` action that
    day; a row with neither (a test fixture, say) is read off its side against the
    idea's. The trigger is the rule text behind `rule_id`, else the horizon's end, else
    the PM's own call; an opening trade carries none.
    """
    rule_text = {rule.rule_id: rule.text for rule in rules}
    notes: dict[TradeKey, TradeNote] = {}
    for row in day_trades:
        idea = ideas.get(row.trade_idea_id)
        if idea is None:
            raise DialogueError(
                f"day trade names trade idea '{row.trade_idea_id}' the PM does not have"
            )
        if idea.entry_date == row.date:
            notes[trade_key(row)] = TradeNote(kind=_OPENS, trigger=None)
            continue
        action = next(
            (
                pd.action
                for pd in position_days
                if pd.trade_idea_id == row.trade_idea_id and pd.date == row.date
            ),
            None,
        )
        if action in _KIND_BY_ACTION:
            kind = _KIND_BY_ACTION[action]
        else:
            kind = "adds to the position" if row.side == idea.side else "closes the whole position"
        if row.rule_id is not None:
            text = rule_text.get(row.rule_id)
            if text is None:
                raise DialogueError(f"day trade names rule '{row.rule_id}' the PM does not have")
            trigger = f"on your rule: {text}"
        elif horizon_end is not None and row.date == horizon_end:
            trigger = _HORIZON_END
        else:
            trigger = _OWN_CALL
        notes[trade_key(row)] = TradeNote(kind=kind, trigger=trigger)
    return notes


@dataclass(frozen=True)
class SessionContext:
    """Everything one session's prompt builders read, already filtered and ordered."""

    skeleton: Skeleton
    persona: Persona
    voice: Voice
    pm_rules: tuple[Rule, ...]
    idea_rules: tuple[Rule, ...]
    ideas: tuple[Idea, ...]
    day_trades: tuple[LedgerRow, ...]
    trade_notes: Mapping[TradeKey, TradeNote]
    open_positions: tuple[Idea, ...]
    question_instrument: Instrument | None
    # Id to name of the session's own instruments, else of the PM's whole asset class.
    instrument_names: Mapping[str, str]
    avoid_lines: tuple[str, ...]
    lookup: MarketLookup
    turn_plan: TurnPlan


@dataclass(frozen=True)
class PmTables:
    """One PM's rows across every table the dialogue stage reads."""

    persona: Persona
    traits: tuple[Trait, ...]
    drift_events: tuple[DriftEvent, ...]
    rules: tuple[Rule, ...]
    ideas: Mapping[str, Idea]
    ledger: tuple[LedgerRow, ...]
    position_days: tuple[PositionDay, ...]
    skeletons: tuple[Skeleton, ...]


def select_pms(tables: Sequence[PmTables], pm_filter: PmFilter) -> tuple[PmTables, ...]:
    """Keep a PM when every set filter criterion matches, ordered by `pm_id`."""
    kept = []
    for pm in tables:
        persona = pm.persona
        if pm_filter.split is not None and persona.split != pm_filter.split:
            continue
        if pm_filter.typicality is not None and persona.typicality != pm_filter.typicality:
            continue
        if pm_filter.drift is not None:
            status = DriftStatus.DRIFT if pm.drift_events else DriftStatus.STATIC
            if status != pm_filter.drift:
                continue
        if pm_filter.pm_ids and persona.pm_id not in pm_filter.pm_ids:
            continue
        kept.append(pm)
    return tuple(sorted(kept, key=lambda pm: pm.persona.pm_id))


def _instrument_name(lookup: MarketLookup, instrument_id: str) -> str:
    instrument = lookup.instruments.get(instrument_id)
    if instrument is None:
        raise DialogueError(f"instrument '{instrument_id}' is not in the market lookup")
    return instrument.name


def _ledger_sort_key(row: LedgerRow) -> tuple[str, str, tuple[int, str], str]:
    """The ledger table's own key, minus the (pm_id, date) already fixed by the caller."""
    tenor_key = (0, "") if row.tenor is None else (1, row.tenor.value)
    return (row.trade_idea_id, row.instrument_id, tenor_key, row.side.value)


def _bias_avoid_line(
    catalogue: Catalogue, traits_by_id: Mapping[str, Trait], trait_id: str, skeleton: Skeleton
) -> str:
    trait = traits_by_id.get(trait_id)
    if trait is None:
        raise DialogueError(
            f"session '{skeleton.session_id}' forbids trait '{trait_id}' the PM does not have"
        )
    line = catalogue.avoid.biases.get(trait.param)
    if line is None:
        raise DialogueError(
            f"session '{skeleton.session_id}' forbids trait '{trait_id}' with param "
            f"'{trait.param}', which has no avoid line in the catalogue"
        )
    return line


def _pref_avoid_line(catalogue: Catalogue, param: str, skeleton: Skeleton) -> str:
    line = catalogue.avoid.preferences.get(param)
    if line is None:
        raise DialogueError(
            f"session '{skeleton.session_id}' forbids preference param '{param}', which "
            "has no avoid line in the catalogue"
        )
    return line


def build_contexts(
    pm: PmTables,
    voice: Voice,
    lookup: MarketLookup,
    catalogue: Catalogue,
    config: Config,
) -> tuple[SessionContext, ...]:
    """Build one `SessionContext` per skeleton, in `session_id` order.

    Raises `DialogueError` if a skeleton names a trade idea the PM does not have.
    """
    pm_rules = tuple(
        sorted((rule for rule in pm.rules if rule.scope == RuleScope.PM), key=lambda r: r.rule_id)
    )
    traits_by_id = {trait.trait_id: trait for trait in pm.traits}

    timeline = config.timeline()
    horizon_end = timeline.weekdays_in_weeks(1, timeline.n_weeks)[-1]
    contexts: list[SessionContext] = []
    for skeleton in sorted(pm.skeletons, key=lambda s: s.session_id):
        ideas: list[Idea] = []
        for idea_id in skeleton.trade_idea_ids:
            idea = pm.ideas.get(idea_id)
            if idea is None:
                raise DialogueError(
                    f"session '{skeleton.session_id}' names trade idea '{idea_id}' "
                    "the PM does not have"
                )
            ideas.append(idea)

        idea_rules = tuple(
            sorted(
                (
                    rule
                    for rule in pm.rules
                    if rule.scope == RuleScope.IDEA
                    and rule.trade_idea_id in skeleton.trade_idea_ids
                ),
                key=lambda r: r.rule_id,
            )
        )

        day_trades = tuple(
            sorted(
                (
                    row
                    for row in pm.ledger
                    if row.date == skeleton.date and row.trade_idea_id in skeleton.trade_idea_ids
                ),
                key=_ledger_sort_key,
            )
        )

        trade_notes = annotate_day_trades(
            day_trades, pm.ideas, pm.position_days, pm.rules, horizon_end
        )

        open_position_ideas: list[Idea] = []
        for row in pm.position_days:
            if row.date != skeleton.date:
                continue
            position_idea = pm.ideas.get(row.trade_idea_id)
            if position_idea is None:
                raise DialogueError(
                    f"session '{skeleton.session_id}' has a position_days row for trade "
                    f"idea '{row.trade_idea_id}' the PM does not have"
                )
            open_position_ideas.append(position_idea)
        open_positions = tuple(sorted(open_position_ideas, key=lambda idea: idea.trade_idea_id))

        turn_plan = plan_turns(
            skeleton,
            day_trades,
            config.dialogue.turns_by_kind,
            stream(config.seed.root, "dialogue", pm.persona.pm_id, skeleton.session_id, "turns"),
        )

        kinds = _ASSET_CLASS_KINDS[pm.persona.mandate.asset_class]
        class_instruments = sorted(
            (inst for inst in lookup.instruments.values() if inst.kind in kinds),
            key=lambda inst: inst.instrument_id,
        )
        question_instrument: Instrument | None = None
        if skeleton.kind == SessionKind.SILENCE:
            candidates = class_instruments
            if not candidates:
                raise DialogueError(
                    f"session '{skeleton.session_id}' has no instrument of the PM's asset "
                    "class in the market lookup"
                )
            question_rng = stream(
                config.seed.root, "dialogue", pm.persona.pm_id, skeleton.session_id, "question"
            )
            question_instrument = candidates[int(question_rng.integers(len(candidates)))]

        instrument_ids = {idea.instrument_id for idea in ideas}
        instrument_ids.update(row.instrument_id for row in day_trades)
        instrument_ids.update(idea.instrument_id for idea in open_positions)
        if question_instrument is not None:
            instrument_ids.add(question_instrument.instrument_id)
        if not instrument_ids:
            # A session with nothing of its own: a failed lookup lists the asset class.
            instrument_ids.update(inst.instrument_id for inst in class_instruments)
        instrument_names = {
            instrument_id: _instrument_name(lookup, instrument_id)
            for instrument_id in sorted(instrument_ids)
        }

        bias_lines = {
            _bias_avoid_line(catalogue, traits_by_id, trait_id, skeleton)
            for trait_id in skeleton.forbidden_trait_ids
        }
        pref_lines = {
            _pref_avoid_line(catalogue, param, skeleton) for param in skeleton.forbidden_pref_params
        }
        avoid_lines = tuple(sorted(bias_lines | pref_lines))

        contexts.append(
            SessionContext(
                skeleton=skeleton,
                persona=pm.persona,
                voice=voice,
                pm_rules=pm_rules,
                idea_rules=idea_rules,
                ideas=tuple(ideas),
                day_trades=day_trades,
                trade_notes=trade_notes,
                open_positions=open_positions,
                question_instrument=question_instrument,
                instrument_names=instrument_names,
                avoid_lines=avoid_lines,
                lookup=lookup,
                turn_plan=turn_plan,
            )
        )
    return tuple(contexts)
