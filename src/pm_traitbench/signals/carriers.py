"""Carrier pools: the dated engine evidence a revealed trait signal can be narrated on.

A carrier is one row where the engine itself recorded that a bias drove the
decision, or an idea that took the form a preference calls for. Pinning every
revealed signal to a carrier keeps the plan's narration from ever describing an
action the engine did not take.
"""

from dataclasses import dataclass
from datetime import date

from pm_traitbench.catalogues.loader import REVEALED_PATTERNS
from pm_traitbench.engine.adapters import FORM_FOR_PREFERENCE
from pm_traitbench.enums import CarrierSource, Kind, RuleResponse
from pm_traitbench.errors import PlanError
from pm_traitbench.signals.inputs import PlanInputs
from pm_traitbench.tables.schema import Trait

FLAG_PREFIX_PARAM: dict[str, str] = {
    "disposition": "disposition_ratio",
    "loss_aversion": "loss_aversion_lambda",
    "exit_deficiency": "exit_deficiency",
    "anchoring": "anchoring_rho",
    "herding": "herding_weight",
    "overconfidence": "overconfidence_coverage",
    "conviction": "conviction_size_miscalibration",
}
HOLD_FLAGS: frozenset[str] = frozenset({"loss_aversion:hold", "disposition:hold_loser"})
BREACH_RESPONSES: frozenset[RuleResponse] = frozenset(
    {RuleResponse.ACKED_NO_ACTION, RuleResponse.ADDED}
)

# Dedup priority when the same (trait, idea, date, pattern) shows up in two tables:
# a ledger row and a position-day snapshot of the same order count once, as the ledger.
_SOURCE_ORDER: dict[CarrierSource, int] = {
    CarrierSource.LEDGER: 0,
    CarrierSource.POSITION_DAY: 1,
    CarrierSource.RULE_EVENT: 2,
    CarrierSource.IDEA: 3,
}


@dataclass(frozen=True)
class Carrier:
    """One dated piece of engine evidence a signal can point at."""

    trait_id: str
    trade_idea_id: str
    date: date
    source: CarrierSource
    pattern: str | None


def _preference_keys(inputs: PlanInputs, pref_traits: dict[str, Trait]) -> dict[str, Trait]:
    """Preference traits whose value in force on some idea's entry date has a mapped form."""
    keys: dict[str, Trait] = {}
    for param, trait in pref_traits.items():
        for idea in inputs.ideas.values():
            value = inputs.value_at(trait.trait_id, idea.entry_date)
            if FORM_FOR_PREFERENCE.get((param, value)) is not None:
                keys[param] = trait
                break
    return keys


def carrier_pools(inputs: PlanInputs) -> dict[str, tuple[Carrier, ...]]:
    """Carrier pool per active bias trait id and per expression preference trait id.

    Raises `PlanError` when a `bias_flag` names a prefix or pattern this stage does
    not understand, so the engine adding a new flag never silently plants a signal
    that no stance line can back.
    """
    pm_id = inputs.persona.pm_id
    active_biases = {
        trait.param: trait for trait in inputs.traits if trait.kind == Kind.BIAS and trait.active
    }
    pref_traits = {trait.param: trait for trait in inputs.traits if trait.kind == Kind.PREFERENCE}
    pref_keys = _preference_keys(inputs, pref_traits)

    pools: dict[str, list[Carrier]] = {trait.trait_id: [] for trait in active_biases.values()}
    for trait in pref_keys.values():
        pools.setdefault(trait.trait_id, [])

    def emit(
        trait: Trait | None,
        trade_idea_id: str,
        day: date,
        source: CarrierSource,
        pattern: str | None,
    ) -> None:
        if trait is not None:
            carrier = Carrier(trait.trait_id, trade_idea_id, day, source, pattern)
            pools[trait.trait_id].append(carrier)

    hold_seen: set[tuple[str, str]] = set()
    flag_rows = (
        (CarrierSource.LEDGER, inputs.ledger),
        (CarrierSource.POSITION_DAY, inputs.position_days),
    )
    for source, rows in flag_rows:
        for row in rows:
            if not row.bias_flag:
                continue
            for flag in row.bias_flag.split(";"):
                prefix, _, pattern = flag.partition(":")
                param = FLAG_PREFIX_PARAM.get(prefix)
                if param is None:
                    raise PlanError(f"PM '{pm_id}': unknown bias flag prefix '{prefix}'")
                if pattern not in REVEALED_PATTERNS[param]:
                    raise PlanError(
                        f"PM '{pm_id}': flag '{flag}' pattern is not a known '{param}' pattern"
                    )
                if flag in HOLD_FLAGS and source is CarrierSource.POSITION_DAY:
                    hold_key = (row.trade_idea_id, flag)
                    if hold_key in hold_seen:
                        continue
                    hold_seen.add(hold_key)
                emit(active_biases.get(param), row.trade_idea_id, row.date, source, pattern)

    exit_trait = active_biases.get("exit_deficiency")
    for event in inputs.rule_events:
        if event.response in BREACH_RESPONSES:
            emit(
                exit_trait,
                event.trade_idea_id,
                event.response_date,
                CarrierSource.RULE_EVENT,
                event.response.value,
            )

    extrapolation_trait = active_biases.get("extrapolation_theta")
    for idea in inputs.ideas.values():
        if idea.chased_trend:
            emit(
                extrapolation_trait,
                idea.trade_idea_id,
                idea.entry_date,
                CarrierSource.IDEA,
                "chased_trend",
            )

    for param, trait in pref_keys.items():
        for idea in inputs.ideas.values():
            value = inputs.value_at(trait.trait_id, idea.entry_date)
            form = FORM_FOR_PREFERENCE.get((param, value))
            if form is not None and idea.expression == form:
                emit(trait, idea.trade_idea_id, idea.entry_date, CarrierSource.IDEA, None)

    return {
        trait_id: _dedupe_and_sort(trait_id, carriers, inputs)
        for trait_id, carriers in pools.items()
    }


def _dedupe_and_sort(
    trait_id: str, carriers: list[Carrier], inputs: PlanInputs
) -> tuple[Carrier, ...]:
    kept: dict[tuple[str, str, date, str | None], Carrier] = {}
    for carrier in carriers:
        if inputs.is_dormant(trait_id, carrier.date):
            continue
        key = (carrier.trait_id, carrier.trade_idea_id, carrier.date, carrier.pattern)
        kept.setdefault(key, carrier)
    return tuple(
        sorted(kept.values(), key=lambda c: (c.date, c.trade_idea_id, _SOURCE_ORDER[c.source]))
    )
