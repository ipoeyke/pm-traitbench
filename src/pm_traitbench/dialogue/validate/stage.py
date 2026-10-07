"""Validate stage: checks every narrated session against the ledger, the leakage rule and
the forbidden set, regenerating or dropping a session that keeps failing, then rewrites
`sessions` and `dialogue_logs` to match.

Every session validates through one shared `CachedClient`, with at most
`config.validation.max_concurrency` sessions in flight at once, mirroring the
dialogue stage's own concurrency and budget handling; the whole run still
writes `validation`, `sessions` and `dialogue_logs` or none of them.
"""

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pm_traitbench.catalogues.loader import (
    check_dialogue_catalogue,
    leak_param_names,
    load_catalogue,
)
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import AnthropicClient, LlmClient, UsageTotals
from pm_traitbench.dialogue.context import SessionContext
from pm_traitbench.dialogue.prompts import prompt_sha256, read_advisor_prompt
from pm_traitbench.dialogue.stage import (
    DIALOGUE_READS,
    build_lookups,
    collect_results,
    partition_pm_tables,
    pm_contexts,
    raise_on_failure,
    run_bounded,
    stage_client,
)
from pm_traitbench.dialogue.validate.loop import LAYERS, SessionOutcome, run_session
from pm_traitbench.enums import Typicality, ValidationStatus
from pm_traitbench.errors import ValidateError
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import DialogueLog, LedgerRow, Session, Signal, ValidationRow
from pm_traitbench.tables.specs import DIALOGUE_LOGS, SESSIONS, SIGNALS, VALIDATION
from pm_traitbench.tables.store import DataStore

# Above this share of a typicality cell's sessions regenerating, the narrator is
# reverting to stereotype and the prompt, not the validator, needs fixing.
_REGENERATION_WARNING_THRESHOLD = 0.30


@dataclass(frozen=True)
class _Unit:
    """One session's context, its stage 6 output and the per-PM inputs its checks need."""

    ctx: SessionContext
    session: Session
    log: DialogueLog
    ledger: tuple[LedgerRow, ...]
    trait_param_by_id: Mapping[str, str]
    typicality: Typicality


def _build_units(
    config: Config,
    store: DataStore,
    catalogue: Catalogue,
    voices_meta: Mapping[str, Any],
    sessions_by_key: Mapping[tuple[str, str], Session],
    logs_by_key: Mapping[tuple[str, str], DialogueLog],
    previously_dropped_ids_confirmed: set[str],
) -> tuple[tuple[_Unit, ...], tuple[str, ...]]:
    """Every session this run will check, plus the session ids an earlier run already dropped."""
    pm_ids = set(voices_meta)
    all_pm_tables = partition_pm_tables(store)
    selected = [pm for pm in all_pm_tables if pm.persona.pm_id in pm_ids]
    missing_pms = sorted(pm_ids - {pm.persona.pm_id for pm in selected})
    if missing_pms:
        # `partition_pm_tables` drops a PM with no skeletons; rewriting `sessions` and
        # `dialogue_logs` for it anyway would silently delete its narrated rows.
        raise ValidateError(f"pm(s) {', '.join(missing_pms)} were narrated but have no skeletons")

    lookups = build_lookups(store, (pm.persona.market_seed for pm in selected))

    units: list[_Unit] = []
    previously_dropped: list[str] = []
    for pm in selected:
        voice, contexts = pm_contexts(pm, lookups, catalogue, config)
        expected_voice_id = voices_meta.get(pm.persona.pm_id)
        if voice.voice_id != expected_voice_id:
            raise ValidateError(
                f"pm '{pm.persona.pm_id}': voice draw '{voice.voice_id}' does not match "
                f"dialogue run metadata's voice '{expected_voice_id}'"
            )
        trait_param_by_id = {trait.trait_id: trait.param for trait in pm.traits}
        for ctx in contexts:
            key = (pm.persona.pm_id, ctx.skeleton.session_id)
            session_row = sessions_by_key.get(key)
            log_row = logs_by_key.get(key)
            if session_row is None or log_row is None:
                # Skip only a session both rows are missing for and an earlier run's
                # `validation` table already confirmed dropped; anything else missing is loud.
                confirmed_dropped = (
                    session_row is None
                    and log_row is None
                    and ctx.skeleton.session_id in previously_dropped_ids_confirmed
                )
                if not confirmed_dropped:
                    raise ValidateError(
                        f"session '{ctx.skeleton.session_id}' has no narrated dialogue output"
                    )
                previously_dropped.append(ctx.skeleton.session_id)
                continue
            units.append(
                _Unit(
                    ctx=ctx,
                    session=session_row,
                    log=log_row,
                    ledger=pm.ledger,
                    trait_param_by_id=trait_param_by_id,
                    typicality=pm.persona.typicality,
                )
            )
    return tuple(units), tuple(previously_dropped)


def _run_metadata(
    config: Config,
    typicality_by_session: Mapping[str, Typicality],
    new_rows: Sequence[ValidationRow],
    session_warnings: Sequence[str],
    rejected_replies: int,
    dropped_session_ids: Sequence[str],
    signals: Sequence[Signal],
    pm_ids: set[str],
    totals: UsageTotals,
) -> dict[str, Any]:
    """Build the validate stage's run metadata from this run's own attempts and outcomes.

    `typicality_by_session` covers every session this run checked. `new_rows` is this
    run's own attempt rows (not counting rows carried forward from an earlier run's
    dropped sessions), so `fails_by_layer`, `regenerated` and the typicality rates all
    describe this run alone. `dropped`, `dropped_session_ids`, `void_signal_ids` and
    `void_signals_by_pm` are cumulative across every run instead, since a dropped
    session stays dropped. `fallback_verdicts` counts this run's judge verdicts that came
    from the refusal fallback model.
    """
    dropped = set(dropped_session_ids)
    void_signal_ids: list[str] = []
    void_signals_by_pm: Counter[str] = Counter()
    for sig in signals:
        if sig.session_id in dropped:
            void_signal_ids.append(sig.signal_id)
            void_signals_by_pm[sig.pm_id] += 1

    fails_by_layer = dict.fromkeys(LAYERS, 0)
    regenerate_rows = 0
    # A session regenerated more than once still counts once: the rate is the share of
    # sessions that needed regeneration, not the number of regenerate attempts.
    regenerated_sessions: set[str] = set()
    for row in new_rows:
        for layer in LAYERS:
            fails_by_layer[layer] += not getattr(row, f"{layer}_ok")
        if row.status == ValidationStatus.REGENERATE:
            regenerate_rows += 1
            regenerated_sessions.add(row.session_id)

    sessions_by_typicality = Counter(typicality_by_session.values())
    regenerated_by_typicality = Counter(typicality_by_session[sid] for sid in regenerated_sessions)
    regeneration_rate_by_typicality = {
        t: regenerated_by_typicality[t] / n for t, n in sorted(sessions_by_typicality.items())
    }

    warnings = list(session_warnings)
    for t, rate in regeneration_rate_by_typicality.items():
        if rate > _REGENERATION_WARNING_THRESHOLD:
            warnings.append(
                f"regeneration rate {rate:.2f} above {_REGENERATION_WARNING_THRESHOLD:.2f} "
                f"for typicality {t}"
            )

    return {
        "judge_model": config.validation.judge_model,
        "refusal_fallback_model": config.validation.refusal_fallback_model,
        "fallback_verdicts": sum(len(row.fallback_judges) for row in new_rows),
        "pms": sorted(pm_ids),
        "sessions_checked": len(typicality_by_session),
        "fails_by_layer": fails_by_layer,
        "regenerated": regenerate_rows,
        "dropped": len(dropped_session_ids),
        "dropped_session_ids": sorted(dropped_session_ids),
        "void_signal_ids": sorted(void_signal_ids),
        "void_signals_by_pm": dict(sorted(void_signals_by_pm.items())),
        "regeneration_rate_by_typicality": regeneration_rate_by_typicality,
        "warnings": warnings,
        **totals.as_metadata(config.prices.models),
        "rejected_replies": rejected_replies,
    }


def _run(
    config: Config, store: DataStore, client_factory: Callable[[Config], LlmClient]
) -> dict[str, Any]:
    dialogue_meta = store.read_run_metadata("dialogue")
    if dialogue_meta is None:
        raise ValidateError("dialogue run metadata is missing; run the dialogue stage first")

    catalogue = load_catalogue()
    check_dialogue_catalogue(catalogue)

    advisor_prompt = read_advisor_prompt(config.dialogue.advisor_prompt_path)
    advisor_sha256 = prompt_sha256(advisor_prompt)
    if "advisor_prompt_sha256" not in dialogue_meta:
        raise ValidateError("dialogue run metadata is missing key 'advisor_prompt_sha256'")
    if advisor_sha256 != dialogue_meta["advisor_prompt_sha256"]:
        raise ValidateError(
            "the advisor prompt has changed since the dialogue run narrated against it: "
            f"expected sha256 {dialogue_meta['advisor_prompt_sha256']}, got {advisor_sha256}"
        )
    if "voices" not in dialogue_meta:
        raise ValidateError("dialogue run metadata is missing key 'voices'")

    sessions = store.read(SESSIONS)
    dialogue_logs = store.read(DIALOGUE_LOGS)
    sessions_by_key = {(s.pm_id, s.session_id): s for s in sessions}
    logs_by_key = {(log.pm_id, log.session_id): log for log in dialogue_logs}

    # Every PM stage 6 narrated: deriving this from `sessions` instead would stop
    # visiting a PM once every one of its sessions had been dropped.
    pm_ids = set(dialogue_meta["voices"])

    # A session only counts as already dropped, rather than never narrated,
    # when an earlier run's own `validation` table says so.
    previous_rows = store.read(VALIDATION) if store.exists(VALIDATION) else []
    previously_dropped_ids_confirmed = {
        row.session_id for row in previous_rows if row.status == ValidationStatus.DROPPED
    }

    grep_terms = leak_param_names(catalogue)
    frozen_units, previously_dropped_session_ids = _build_units(
        config,
        store,
        catalogue,
        dialogue_meta["voices"],
        sessions_by_key,
        logs_by_key,
        previously_dropped_ids_confirmed,
    )

    client = stage_client(store, client_factory, config, config.validation.token_budget)

    async def validate(unit: _Unit) -> SessionOutcome:
        return await run_session(
            unit.ctx,
            unit.session,
            unit.log,
            client,
            config,
            catalogue,
            advisor_prompt,
            unit.ledger,
            grep_terms,
            unit.trait_param_by_id,
        )

    outcomes = asyncio.run(
        run_bounded(
            frozen_units,
            validate,
            client,
            config.validation.max_concurrency,
            label="validate",
            unit="sessions",
            unit_name=lambda unit: f"session {unit.ctx.skeleton.session_id}",
        )
    )
    raise_on_failure(
        tuple(unit.ctx.skeleton.session_id for unit in frozen_units),
        outcomes,
        client,
        error_type=ValidateError,
        budget_label="validation",
    )

    session_outcomes = collect_results(outcomes, SessionOutcome, len(frozen_units), ValidateError)

    new_rows = [row for outcome in session_outcomes for row in outcome.rows]
    session_warnings = [w for outcome in session_outcomes for w in outcome.warnings]
    rejected_replies = sum(o.rejected_replies for o in session_outcomes)
    finals = {
        unit.ctx.skeleton.session_id: outcome.final
        for unit, outcome in zip(frozen_units, session_outcomes, strict=True)
    }

    # Validate rewrites the final rows of every PM dialogue narrated: no PM stays
    # unvalidated, so `sessions`/`dialogue_logs` hold exactly this run's passing rows.
    new_sessions = [final.session for final in finals.values() if final is not None]
    new_logs = [final.log for final in finals.values() if final is not None]

    newly_dropped_ids = [session_id for session_id, final in finals.items() if final is None]
    dropped_session_ids = sorted({*newly_dropped_ids, *previously_dropped_session_ids})

    # A session an earlier run already dropped keeps its earlier validation rows,
    # unchanged, since this run never re-checks it.
    previously_dropped_set = set(previously_dropped_session_ids)
    carried_rows = [row for row in previous_rows if row.session_id in previously_dropped_set]

    store.write(VALIDATION, [*carried_rows, *new_rows])
    store.write(SESSIONS, new_sessions)
    store.write(DIALOGUE_LOGS, new_logs)

    signals = store.read(SIGNALS)
    return _run_metadata(
        config,
        {unit.ctx.skeleton.session_id: unit.typicality for unit in frozen_units},
        new_rows,
        session_warnings,
        rejected_replies,
        dropped_session_ids,
        signals,
        pm_ids,
        client.totals,
    )


def make_stage(client_factory: Callable[[Config], LlmClient]) -> Stage:
    """Build the validate stage around one client factory, so a test can swap in a fake."""

    def run(config: Config, store: DataStore) -> dict[str, Any]:
        return _run(config, store, client_factory)

    return Stage(
        number=7,
        name="validate",
        help=(
            "check every narrated session against the ledger, the leakage rule and the "
            "forbidden set; regenerate or drop failures"
        ),
        run=run,
        reads=(*DIALOGUE_READS, SIGNALS),
        writes=(VALIDATION,),
        # Validate checks every PM dialogue narrated, so it rewrites both tables whole.
        rewrites=(SESSIONS, DIALOGUE_LOGS),
    )


def anthropic_client_factory(config: Config) -> LlmClient:
    return AnthropicClient(config.validation.max_concurrency, config.dialogue.api_max_retries)


VALIDATE_STAGE = make_stage(anthropic_client_factory)
