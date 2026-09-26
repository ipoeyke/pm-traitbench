"""Per-session context: assembles one skeleton's rows into everything the narrator and
advisor prompt builders need, and filters which PMs a dialogue run covers.

Nothing here renders text; it only selects and orders the rows a session's
prompts will draw from, so ordering and filtering logic is tested once
rather than re-derived inside every prompt builder.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pm_traitbench.catalogues.models import Catalogue, Voice
from pm_traitbench.config import Config, PmFilter
from pm_traitbench.dialogue.tools import MarketLookup
from pm_traitbench.dialogue.turns import TurnPlan, plan_turns
from pm_traitbench.enums import AssetClass, DriftStatus, InstrumentKind, SessionKind
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
    open_positions: tuple[Idea, ...]
    question_instrument: Instrument | None
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
                key=lambda row: row.trade_idea_id,
            )
        )

        open_positions = tuple(
            sorted(
                (
                    pm.ideas[row.trade_idea_id]
                    for row in pm.position_days
                    if row.date == skeleton.date and row.trade_idea_id in pm.ideas
                ),
                key=lambda idea: idea.trade_idea_id,
            )
        )

        turn_plan = plan_turns(
            skeleton,
            day_trades,
            config.dialogue.turns_by_kind,
            stream(config.seed.root, "dialogue", pm.persona.pm_id, skeleton.session_id, "turns"),
        )

        question_instrument: Instrument | None = None
        if skeleton.kind == SessionKind.SILENCE:
            kinds = _ASSET_CLASS_KINDS[pm.persona.mandate.asset_class]
            candidates = sorted(
                (inst for inst in lookup.instruments.values() if inst.kind in kinds),
                key=lambda inst: inst.instrument_id,
            )
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
        instrument_names = {
            instrument_id: _instrument_name(lookup, instrument_id)
            for instrument_id in sorted(instrument_ids)
        }

        bias_lines = {
            catalogue.avoid.biases[traits_by_id[trait_id].param]
            for trait_id in skeleton.forbidden_trait_ids
        }
        pref_lines = {
            catalogue.avoid.preferences[param] for param in skeleton.forbidden_pref_params
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
                open_positions=open_positions,
                question_instrument=question_instrument,
                instrument_names=instrument_names,
                avoid_lines=avoid_lines,
                lookup=lookup,
                turn_plan=turn_plan,
            )
        )
    return tuple(contexts)
