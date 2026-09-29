"""Gate 2 stage: recover every planted trait from a PM's validated transcript with a strong
model, classify each stated signal, and block on nine pooled exact-test rows.

Every recovery and classification request goes through one shared `CachedClient`, scoped by
PM id for recovery and by session id for classification; the run writes all four tables or none.
"""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from pm_traitbench.catalogues.loader import check_gate2_catalogue, load_catalogue
from pm_traitbench.catalogues.models import PreferenceEntry
from pm_traitbench.config import Config
from pm_traitbench.corpus_checks import check_validated_corpus
from pm_traitbench.dialogue.client import AnthropicClient, LlmClient
from pm_traitbench.dialogue.stage import (
    collect_results,
    raise_on_failure,
    run_bounded,
    stage_client,
)
from pm_traitbench.enums import DriftStatus, Kind, RuleScope, SignalMode
from pm_traitbench.errors import CorpusError, DialogueBudgetError, Gate2Error
from pm_traitbench.gates.gate2.aggregate import (
    blocking_failures,
    build_cells,
    insufficient_blocking,
)
from pm_traitbench.gates.gate2.classify import (
    ClassifyUnit,
    classify_request,
    classify_units,
    score_classification,
    send_classification,
)
from pm_traitbench.gates.gate2.overlap import containment_by_pm, summarise
from pm_traitbench.gates.gate2.recover import (
    RecoveryReply,
    pre_update_signal_ids,
    recovery_request,
    send_recovery,
    signal_rows,
    trait_rows,
)
from pm_traitbench.gates.gate2.transcript import pm_turn_text, render_pm
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import (
    DriftEvent,
    Gate2PmRow,
    Gate2SignalRow,
    Gate2TraitRow,
    LedgerRow,
    Rule,
    Session,
    Signal,
    Trait,
)
from pm_traitbench.tables.specs import (
    DIALOGUE_LOGS,
    DRIFT_EVENTS,
    GATE2_CELLS,
    GATE2_PM,
    GATE2_SIGNALS,
    GATE2_TABLES,
    GATE2_TRAITS,
    LEDGER,
    PERSONAS,
    RULES,
    SESSIONS,
    SIGNALS,
    SKELETONS,
    TRAITS,
    TableSpec,
)
from pm_traitbench.tables.store import DataStore
from pm_traitbench.traits_truth import TraitTruth, compute_truth

# Every table gate 2 reads to rebuild each PM's context and score its recovery.
GATE2_READS: tuple[TableSpec, ...] = (
    PERSONAS,
    TRAITS,
    DRIFT_EVENTS,
    RULES,
    LEDGER,
    SIGNALS,
    SKELETONS,
    SESSIONS,
    DIALOGUE_LOGS,
)


@dataclass(frozen=True)
class RecoveryUnit:
    """One PM's recovery request, ready to send, plus the candidate preference entries."""

    pm_id: str
    request: dict[str, Any]
    entries: tuple[PreferenceEntry, ...]


@dataclass(frozen=True)
class ClassifyRequestUnit:
    """One session's classification request, ready to send, plus its scoring unit."""

    session_id: str
    request: dict[str, Any]
    n: int
    unit: ClassifyUnit


def raise_on_failures(extra: dict[str, Any]) -> None:
    """Raise `Gate2Error` naming every failed blocking cell, if any. Insufficient blocking
    cells are reported in `extra["insufficient"]` but never raise.
    """
    failed = extra["failed"]
    if failed:
        raise Gate2Error("gate 2 failed for: " + ", ".join(failed))


def _group_by_pm[T](rows: list[T], pm_id_of: Callable[[T], str]) -> dict[str, list[T]]:
    grouped: dict[str, list[T]] = {}
    for row in rows:
        grouped.setdefault(pm_id_of(row), []).append(row)
    return grouped


def _run(
    config: Config, store: DataStore, client_factory: Callable[[Config], LlmClient]
) -> dict[str, Any]:
    sessions = store.read(SESSIONS)
    sessions_sha256 = hashlib.sha256(store.path(SESSIONS).read_bytes()).hexdigest()
    session_pm_ids = sorted({s.pm_id for s in sessions})

    try:
        validate_meta = check_validated_corpus(store, session_pm_ids)
    except CorpusError as e:
        raise Gate2Error(str(e)) from e

    catalogue = load_catalogue()
    check_gate2_catalogue(catalogue)

    personas = {p.pm_id: p for p in store.read(PERSONAS)}
    traits_by_pm = _group_by_pm(store.read(TRAITS), lambda t: t.pm_id)
    drift_by_pm = _group_by_pm(store.read(DRIFT_EVENTS), lambda d: d.pm_id)
    rules_by_pm = _group_by_pm(store.read(RULES), lambda r: r.pm_id)
    ledger_by_pm = _group_by_pm(store.read(LEDGER), lambda row: row.pm_id)
    signals_by_pm = _group_by_pm(store.read(SIGNALS), lambda sig: sig.pm_id)
    sessions_by_pm = _group_by_pm(sessions, lambda s: s.pm_id)

    sessions_by_id = {s.session_id: s for s in sessions}
    logs_by_id = {log.session_id: log for log in store.read(DIALOGUE_LOGS)}
    skeletons_by_id = {sk.session_id: sk for sk in store.read(SKELETONS)}

    pms_with_sessions = session_pm_ids
    pms_without_sessions = sorted(set(validate_meta["pms"]) - set(pms_with_sessions))

    truth_by_pm: dict[str, dict[str, TraitTruth]] = {}
    recovery_units: list[RecoveryUnit] = []
    classify_units_by_pm: dict[str, tuple[ClassifyUnit, ...]] = {}
    classify_request_units: list[ClassifyRequestUnit] = []

    for pm_id in pms_with_sessions:
        persona = personas[pm_id]
        pm_sessions: list[Session] = sessions_by_pm[pm_id]
        pm_traits: list[Trait] = traits_by_pm.get(pm_id, [])
        pm_drift: list[DriftEvent] = drift_by_pm.get(pm_id, [])
        pm_rules: list[Rule] = rules_by_pm.get(pm_id, [])
        last_date: date = max(s.date for s in pm_sessions)

        entries = catalogue.preferences_for(persona.mandate.asset_class)
        try:
            truth_by_pm[pm_id] = compute_truth(pm_traits, pm_drift, last_date, entries)
        except CorpusError as e:
            raise Gate2Error(str(e)) from e

        transcript = render_pm(pm_sessions)
        request = recovery_request(
            persona, pm_rules, catalogue.bias_definitions, entries, transcript, config.gate2
        )
        recovery_units.append(RecoveryUnit(pm_id=pm_id, request=request, entries=entries))

        pm_signals: list[Signal] = signals_by_pm.get(pm_id, [])
        units = classify_units(sessions_by_id, logs_by_id, skeletons_by_id, pm_signals)
        classify_units_by_pm[pm_id] = units

        idea_rules = [r for r in pm_rules if r.scope == RuleScope.IDEA]
        pm_ledger: list[LedgerRow] = ledger_by_pm.get(pm_id, [])
        for unit in units:
            n = len(unit.signals)
            req = classify_request(
                persona, pm_rules, idea_rules, pm_ledger, unit.session, n, config.gate2
            )
            classify_request_units.append(
                ClassifyRequestUnit(session_id=unit.session.session_id, request=req, n=n, unit=unit)
            )

    client = stage_client(store, client_factory, config, config.gate2.token_budget)
    combined_units: list[RecoveryUnit | ClassifyRequestUnit] = [
        *recovery_units,
        *classify_request_units,
    ]

    async def send_unit(unit: RecoveryUnit | ClassifyRequestUnit) -> tuple[Any, int]:
        if isinstance(unit, RecoveryUnit):
            return await send_recovery(
                client, unit.request, unit.entries, unit.pm_id, config.dialogue.max_retries
            )
        return await send_classification(
            client, unit.request, unit.n, unit.session_id, config.dialogue.max_retries
        )

    results = asyncio.run(
        run_bounded(
            combined_units,
            send_unit,
            client,
            config.gate2.max_concurrency,
            label="gate2",
            unit="units",
        )
    )
    n_recovery = len(recovery_units)
    recovery_results = results[:n_recovery]
    classify_results = results[n_recovery:]

    combined_ids = tuple(
        u.pm_id if isinstance(u, RecoveryUnit) else u.session_id for u in combined_units
    )
    # A budget error on either half must win over a failed unit on the other half.
    if any(isinstance(r, DialogueBudgetError) for r in results):
        raise_on_failure(
            combined_ids, list(results), client, error_type=Gate2Error, budget_label="gate2"
        )

    # No budget error: report a failed unit from each half in one Gate2Error, not just
    # whichever half is checked first.
    messages: list[str] = []
    for unit_ids, unit_results, label in (
        (tuple(u.pm_id for u in recovery_units), recovery_results, "pm"),
        (tuple(u.session_id for u in classify_request_units), classify_results, "session"),
    ):
        try:
            raise_on_failure(
                unit_ids,
                unit_results,
                client,
                error_type=Gate2Error,
                budget_label="gate2",
                label=label,
            )
        except Gate2Error as error:
            messages.append(str(error))
    if messages:
        raise Gate2Error("\n".join(messages))

    recovery_success = collect_results(recovery_results, tuple, n_recovery, Gate2Error)
    classify_success = collect_results(
        classify_results, tuple, len(classify_request_units), Gate2Error
    )

    reply_by_pm: dict[str, RecoveryReply] = {
        unit.pm_id: reply
        for unit, (reply, _rejected) in zip(recovery_units, recovery_success, strict=True)
    }
    statements_by_session = {
        unit.session_id: statements
        for unit, (statements, _rejected) in zip(
            classify_request_units, classify_success, strict=True
        )
    }
    rejected_replies = sum(rejected for _, rejected in (*recovery_success, *classify_success))

    citation_warnings: list[str] = []
    quote_warnings: list[str] = []
    trait_rows_all: list[Gate2TraitRow] = []
    signal_rows_all: list[Gate2SignalRow] = []
    texts_by_pm: dict[str, str] = {}
    pm_stats: dict[str, dict[str, Any]] = {}

    for pm_id in pms_with_sessions:
        persona = personas[pm_id]
        pm_sessions = sessions_by_pm[pm_id]
        pm_traits = traits_by_pm.get(pm_id, [])
        pm_drift = drift_by_pm.get(pm_id, [])
        pm_signals = signals_by_pm.get(pm_id, [])
        session_ids = {s.session_id for s in pm_sessions}

        reply = reply_by_pm[pm_id]
        truth = truth_by_pm[pm_id]
        rows, cite_warnings = trait_rows(pm_id, reply, truth, pm_signals, pm_traits, session_ids)
        citation_warnings.extend(cite_warnings)
        trait_rows_all.extend(rows)

        pre_update = pre_update_signal_ids(pm_signals, pm_traits, pm_drift)
        kind_by_trait = {t.trait_id: t.kind for t in pm_traits}
        classification: dict[str, tuple[Kind | None, bool]] = {}
        for unit in classify_units_by_pm.get(pm_id, ()):
            statements = statements_by_session[unit.session.session_id]
            scores, unit_warnings = score_classification(unit, statements, kind_by_trait)
            classification.update(scores)
            quote_warnings.extend(unit_warnings)

        sig_rows = signal_rows(
            pm_id, pm_signals, pm_traits, rows, pre_update, classification, session_ids
        )
        signal_rows_all.extend(sig_rows)

        texts_by_pm[pm_id] = pm_turn_text(pm_sessions)

        held_rows = [r for r in rows if r.kind == Kind.PREFERENCE and r.truth_value is not None]
        pm_stats[pm_id] = {
            "sessions": len(pm_sessions),
            "context_chars": len(render_pm(pm_sessions)),
            "biases_correct": sum(1 for r in rows if r.kind == Kind.BIAS and r.correct),
            "preferences_held": len(held_rows),
            "preferences_correct": sum(1 for r in held_rows if r.correct),
            "stated_signals": sum(
                1 for r in sig_rows if r.mode == SignalMode.STATED and r.classified
            ),
            "stated_kind_ok": sum(
                1 for r in sig_rows if r.mode == SignalMode.STATED and bool(r.kind_ok)
            ),
        }

    containments = containment_by_pm(texts_by_pm, config.gate2.ngram_n)
    overlap_warnings = [
        f"pm {pm_id}: n-gram containment {containments[pm_id]:.2f} above "
        f"{config.gate2.overlap_warning:.2f}"
        for pm_id in pms_with_sessions
        if containments[pm_id] > config.gate2.overlap_warning
    ]

    pm_rows = [
        Gate2PmRow(
            pm_id=pm_id,
            asset_class=personas[pm_id].mandate.asset_class,
            typicality=personas[pm_id].typicality,
            drift=DriftStatus.DRIFT if drift_by_pm.get(pm_id) else DriftStatus.STATIC,
            seed=personas[pm_id].market_seed,
            ngram_containment=containments[pm_id],
            **pm_stats[pm_id],
        )
        for pm_id in pms_with_sessions
    ]

    k_by_param = {entry.param: len(entry.values) for entry in catalogue.preferences}
    cells = build_cells(trait_rows_all, signal_rows_all, pm_rows, k_by_param, config.gate2)

    store.write(GATE2_TRAITS, trait_rows_all)
    store.write(GATE2_SIGNALS, signal_rows_all)
    store.write(GATE2_PM, pm_rows)
    store.write(GATE2_CELLS, cells)

    return {
        "model": config.gate2.model,
        "pms": pms_with_sessions,
        "pms_without_sessions": pms_without_sessions,
        "sessions_sha256": sessions_sha256,
        "failed": blocking_failures(cells),
        "insufficient": insufficient_blocking(cells),
        "overlap": summarise(list(containments.values())),
        "warnings": [*citation_warnings, *quote_warnings, *overlap_warnings],
        **client.totals.as_metadata(),
        "rejected_replies": rejected_replies,
        "thresholds": config.gate2.model_dump(mode="json"),
    }


def make_stage(client_factory: Callable[[Config], LlmClient]) -> Stage:
    """Build the gate 2 stage around one client factory, so a test can swap in a fake."""

    def run(config: Config, store: DataStore) -> dict[str, Any]:
        return _run(config, store, client_factory)

    return Stage(
        number=8,
        name="gate2",
        help=(
            "recover planted traits from the validated dialogue with one full-context call "
            "per PM; blocks on a per-param exact test"
        ),
        run=run,
        reads=GATE2_READS,
        writes=GATE2_TABLES,
        verdict=raise_on_failures,
    )


def anthropic_client_factory(config: Config) -> LlmClient:
    return AnthropicClient(config.gate2.max_concurrency, config.dialogue.api_max_retries)


GATE2_STAGE = make_stage(anthropic_client_factory)
