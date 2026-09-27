"""Validate stage: checks every narrated session against the ledger, the leakage rule and
the forbidden set, regenerating or dropping a session that keeps failing, then rewrites
`sessions` and `dialogue_logs` to match.

Every session validates through one shared `CachedClient`, with at most
`config.validation.max_concurrency` sessions in flight at once, mirroring the
dialogue stage's own concurrency and budget handling; the whole run still
writes `validation`, `sessions` and `dialogue_logs` or none of them.
"""

import asyncio
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pm_traitbench.catalogues.loader import (
    check_dialogue_catalogue,
    check_validate_catalogue,
    load_catalogue,
)
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import AnthropicClient, CachedClient, LlmClient, UsageTotals
from pm_traitbench.dialogue.context import SessionContext, build_contexts
from pm_traitbench.dialogue.prompts import read_advisor_prompt
from pm_traitbench.dialogue.stage import partition_pm_tables, raise_on_failure
from pm_traitbench.dialogue.tools import MarketLookup
from pm_traitbench.dialogue.validate.grep import grep_params
from pm_traitbench.dialogue.validate.loop import SessionOutcome, run_session
from pm_traitbench.dialogue.voices import draw_voice
from pm_traitbench.enums import Typicality, ValidationStatus
from pm_traitbench.errors import ValidateError
from pm_traitbench.stages import Append, Stage
from pm_traitbench.tables.schema import DialogueLog, LedgerRow, Session, Signal, ValidationRow
from pm_traitbench.tables.specs import (
    DIALOGUE_LOGS,
    DRIFT_EVENTS,
    IDEAS,
    LEDGER,
    MARKET_CALENDAR,
    MARKET_CONSENSUS,
    MARKET_CURVES,
    MARKET_INSTRUMENTS,
    MARKET_PRICES,
    PERSONAS,
    POSITION_DAYS,
    RULES,
    SESSIONS,
    SIGNALS,
    SKELETONS,
    TRAITS,
    VALIDATION,
)
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


async def _validate_one(
    unit: _Unit,
    client: CachedClient,
    config: Config,
    catalogue: Catalogue,
    advisor_prompt: str,
    grep_terms: Sequence[str],
    semaphore: asyncio.Semaphore,
) -> SessionOutcome:
    async with semaphore:
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


async def _validate_all(
    units: tuple[_Unit, ...],
    client: CachedClient,
    config: Config,
    catalogue: Catalogue,
    advisor_prompt: str,
    grep_terms: Sequence[str],
) -> list[SessionOutcome | BaseException]:
    semaphore = asyncio.Semaphore(config.validation.max_concurrency)
    try:
        return await asyncio.gather(
            *(
                _validate_one(
                    unit, client, config, catalogue, advisor_prompt, grep_terms, semaphore
                )
                for unit in units
            ),
            return_exceptions=True,
        )
    finally:
        await client.aclose()


@dataclass(frozen=True)
class _BuildResult:
    """Every session this run will check, plus session ids an earlier run already dropped."""

    units: tuple[_Unit, ...]
    previously_dropped_session_ids: tuple[str, ...]


def _build_units(
    config: Config,
    store: DataStore,
    catalogue: Catalogue,
    voices_meta: Mapping[str, Any],
    pm_ids: set[str],
    sessions_by_key: Mapping[tuple[str, str], Session],
    logs_by_key: Mapping[tuple[str, str], DialogueLog],
    previously_dropped_ids_confirmed: set[str],
) -> _BuildResult:
    all_pm_tables = partition_pm_tables(store)
    selected = [pm for pm in all_pm_tables if pm.persona.pm_id in pm_ids]
    missing_pms = sorted(pm_ids - {pm.persona.pm_id for pm in selected})
    if missing_pms:
        # `partition_pm_tables` drops a PM with no skeletons; rewriting `sessions` and
        # `dialogue_logs` for it anyway would silently delete its narrated rows.
        raise ValidateError(f"pm(s) {', '.join(missing_pms)} were narrated but have no skeletons")

    instruments = store.read(MARKET_INSTRUMENTS)
    prices = store.read(MARKET_PRICES)
    curves = store.read(MARKET_CURVES)
    consensus = store.read(MARKET_CONSENSUS)
    calendar = store.read(MARKET_CALENDAR)
    lookups = {
        seed: MarketLookup.build(seed, instruments, prices, curves, consensus, calendar)
        for seed in sorted({pm.persona.market_seed for pm in selected})
    }

    units: list[_Unit] = []
    previously_dropped: list[str] = []
    for pm in selected:
        voice = draw_voice(config.seed.root, pm.persona.pm_id, catalogue.voices)
        expected_voice_id = voices_meta.get(pm.persona.pm_id)
        if voice.voice_id != expected_voice_id:
            raise ValidateError(
                f"pm '{pm.persona.pm_id}': voice draw '{voice.voice_id}' does not match "
                f"dialogue run metadata's voice '{expected_voice_id}'"
            )
        lookup = lookups[pm.persona.market_seed]
        trait_param_by_id = {trait.trait_id: trait.param for trait in pm.traits}
        for ctx in build_contexts(pm, voice, lookup, catalogue, config):
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
    return _BuildResult(
        units=tuple(units), previously_dropped_session_ids=tuple(previously_dropped)
    )


def _run_metadata(
    config: Config,
    frozen_units: tuple[_Unit, ...],
    new_rows: Sequence[ValidationRow],
    session_warnings: Sequence[str],
    rejected_replies: int,
    dropped_session_ids: Sequence[str],
    signals: Sequence[Signal],
    pm_ids: set[str],
    totals: UsageTotals,
) -> dict[str, Any]:
    """Build the validate stage's run metadata from this run's own attempts and outcomes.

    `new_rows` is this run's own attempt rows (not counting rows carried
    forward from an earlier run's dropped sessions), so `fails_by_layer`,
    `regenerated` and the typicality rates all describe this run alone.
    `dropped`, `dropped_session_ids`, `void_signal_ids` and `void_signals_by_pm`
    are cumulative across every run instead, since a dropped session stays dropped.
    """
    void_signal_set = set(dropped_session_ids)
    void_signal_ids = sorted(sig.signal_id for sig in signals if sig.session_id in void_signal_set)
    void_signals_by_pm: dict[str, int] = {}
    for sig in signals:
        if sig.session_id in void_signal_set:
            void_signals_by_pm[sig.pm_id] = void_signals_by_pm.get(sig.pm_id, 0) + 1
    void_signals_by_pm = dict(sorted(void_signals_by_pm.items()))

    sessions_by_typicality: dict[Typicality, int] = {}
    typicality_by_pm: dict[str, Typicality] = {}
    for unit in frozen_units:
        typicality_by_pm[unit.ctx.skeleton.pm_id] = unit.typicality
        sessions_by_typicality[unit.typicality] = sessions_by_typicality.get(unit.typicality, 0) + 1
    # A session regenerated more than once still counts once: the rate is the share of
    # sessions that needed regeneration, not the number of regenerate attempts.
    regenerated_sessions_by_typicality: dict[Typicality, set[str]] = {}
    for row in new_rows:
        if row.status == ValidationStatus.REGENERATE:
            t = typicality_by_pm[row.pm_id]
            regenerated_sessions_by_typicality.setdefault(t, set()).add(row.session_id)
    regeneration_rate_by_typicality = {
        t: len(regenerated_sessions_by_typicality.get(t, ())) / n
        for t, n in sorted(sessions_by_typicality.items())
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
        "pms": sorted(pm_ids),
        "sessions_checked": len(frozen_units),
        "fails_by_layer": {
            "ledger": sum(1 for r in new_rows if not r.ledger_ok),
            "grep": sum(1 for r in new_rows if not r.grep_ok),
            "leak": sum(1 for r in new_rows if not r.leak_ok),
            "forbidden": sum(1 for r in new_rows if not r.forbidden_ok),
        },
        "regenerated": sum(1 for r in new_rows if r.status == ValidationStatus.REGENERATE),
        "dropped": len(dropped_session_ids),
        "dropped_session_ids": sorted(dropped_session_ids),
        "void_signal_ids": void_signal_ids,
        "void_signals_by_pm": void_signals_by_pm,
        "regeneration_rate_by_typicality": regeneration_rate_by_typicality,
        "warnings": warnings,
        "calls": totals.calls,
        "cache_hits": totals.cache_hits,
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
        "cache_read_tokens": totals.cache_read_tokens,
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
    check_validate_catalogue(catalogue)

    advisor_prompt = read_advisor_prompt(config.dialogue.advisor_prompt_path)
    prompt_sha256 = hashlib.sha256(advisor_prompt.encode("utf-8")).hexdigest()
    if "advisor_prompt_sha256" not in dialogue_meta:
        raise ValidateError("dialogue run metadata is missing key 'advisor_prompt_sha256'")
    if prompt_sha256 != dialogue_meta["advisor_prompt_sha256"]:
        raise ValidateError(
            "the advisor prompt has changed since the dialogue run narrated against it: "
            f"expected sha256 {dialogue_meta['advisor_prompt_sha256']}, got {prompt_sha256}"
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

    grep_terms = grep_params(catalogue)
    build_result = _build_units(
        config,
        store,
        catalogue,
        dialogue_meta["voices"],
        pm_ids,
        sessions_by_key,
        logs_by_key,
        previously_dropped_ids_confirmed,
    )
    frozen_units = build_result.units
    previously_dropped_session_ids = build_result.previously_dropped_session_ids

    cache_dir = store.data_dir / "cache" / "llm"
    client = CachedClient(lambda: client_factory(config), cache_dir, config.validation.token_budget)
    outcomes = asyncio.run(
        _validate_all(frozen_units, client, config, catalogue, advisor_prompt, grep_terms)
    )
    raise_on_failure(
        tuple(unit.ctx for unit in frozen_units),
        outcomes,
        client,
        error_type=ValidateError,
        budget_label="validation",
    )

    session_outcomes = [o for o in outcomes if isinstance(o, SessionOutcome)]
    if len(session_outcomes) != len(frozen_units):
        raise ValidateError(
            f"expected {len(frozen_units)} session outcomes but got {len(session_outcomes)}"
        )

    new_rows = [row for outcome in session_outcomes for row in outcome.rows]
    session_warnings = [w for outcome in session_outcomes for w in outcome.warnings]
    rejected_replies = sum(o.rejected_replies for o in session_outcomes)
    finals = {
        (unit.ctx.skeleton.pm_id, unit.ctx.skeleton.session_id): outcome.final
        for unit, outcome in zip(frozen_units, session_outcomes, strict=True)
    }

    new_sessions = [s for s in sessions if s.pm_id not in pm_ids]
    new_sessions.extend(final.session for final in finals.values() if final is not None)
    new_logs = [log for log in dialogue_logs if log.pm_id not in pm_ids]
    new_logs.extend(final.log for final in finals.values() if final is not None)

    newly_dropped_ids = [session_id for (_, session_id), final in finals.items() if final is None]
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
        frozen_units,
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
        reads=(
            PERSONAS,
            RULES,
            TRAITS,
            DRIFT_EVENTS,
            IDEAS,
            LEDGER,
            POSITION_DAYS,
            SKELETONS,
            MARKET_INSTRUMENTS,
            MARKET_PRICES,
            MARKET_CURVES,
            MARKET_CONSENSUS,
            MARKET_CALENDAR,
            SIGNALS,
        ),
        writes=(VALIDATION,),
        appends=(
            # Validate checks every PM dialogue narrated, so it owns the whole table.
            Append(SESSIONS, owned=lambda record: True),
            Append(DIALOGUE_LOGS, owned=lambda record: True),
        ),
    )


def anthropic_client_factory(config: Config) -> LlmClient:
    return AnthropicClient(config.validation.max_concurrency, config.dialogue.api_max_retries)


VALIDATE_STAGE = make_stage(anthropic_client_factory)
