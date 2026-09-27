"""Dialogue stage: narrates every session the plan stage skeletoned and writes the public
`sessions` table and the hidden `dialogue_logs` table.

Every session narrates through one shared `CachedClient`, with at most
`config.dialogue.max_concurrency` sessions in flight at once, so a crash or a
rejected reply on one session never blocks the others and only that many
sessions' growing histories sit in memory; the whole run still writes both
tables or neither.
"""

import asyncio
import hashlib
from collections.abc import Callable
from typing import Any

from pm_traitbench.catalogues.loader import check_dialogue_catalogue, load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import AnthropicClient, CachedClient, LlmClient
from pm_traitbench.dialogue.context import PmTables, SessionContext, build_contexts, select_pms
from pm_traitbench.dialogue.prompts import read_advisor_prompt
from pm_traitbench.dialogue.session import SessionResult, narrate_session
from pm_traitbench.dialogue.tools import MarketLookup
from pm_traitbench.dialogue.voices import draw_voice
from pm_traitbench.errors import DialogueBudgetError, DialogueError, PmTraitbenchError
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import (
    DriftEvent,
    Idea,
    LedgerRow,
    PositionDay,
    Rule,
    Skeleton,
    Trait,
)
from pm_traitbench.tables.specs import (
    DIALOGUE_LOGS,
    DIALOGUE_TABLES,
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
    SKELETONS,
    TRAITS,
)
from pm_traitbench.tables.store import DataStore


def partition_pm_tables(store: DataStore) -> tuple[PmTables, ...]:
    """Group every dialogue input table by PM, keeping only PMs with at least one skeleton."""
    personas = {p.pm_id: p for p in store.read(PERSONAS)}

    traits_by_pm: dict[str, list[Trait]] = {}
    for row in store.read(TRAITS):
        traits_by_pm.setdefault(row.pm_id, []).append(row)

    drift_by_pm: dict[str, list[DriftEvent]] = {}
    for row in store.read(DRIFT_EVENTS):
        drift_by_pm.setdefault(row.pm_id, []).append(row)

    rules_by_pm: dict[str, list[Rule]] = {}
    for row in store.read(RULES):
        rules_by_pm.setdefault(row.pm_id, []).append(row)

    ideas_by_pm: dict[str, dict[str, Idea]] = {}
    for row in store.read(IDEAS):
        ideas_by_pm.setdefault(row.pm_id, {})[row.trade_idea_id] = row

    ledger_by_pm: dict[str, list[LedgerRow]] = {}
    for row in store.read(LEDGER):
        ledger_by_pm.setdefault(row.pm_id, []).append(row)

    position_days_by_pm: dict[str, list[PositionDay]] = {}
    for row in store.read(POSITION_DAYS):
        position_days_by_pm.setdefault(row.pm_id, []).append(row)

    skeletons_by_pm: dict[str, list[Skeleton]] = {}
    for row in store.read(SKELETONS):
        skeletons_by_pm.setdefault(row.pm_id, []).append(row)

    return tuple(
        PmTables(
            persona=personas[pm_id],
            traits=tuple(traits_by_pm.get(pm_id, ())),
            drift_events=tuple(drift_by_pm.get(pm_id, ())),
            rules=tuple(rules_by_pm.get(pm_id, ())),
            ideas=ideas_by_pm.get(pm_id, {}),
            ledger=tuple(ledger_by_pm.get(pm_id, ())),
            position_days=tuple(position_days_by_pm.get(pm_id, ())),
            skeletons=tuple(skeletons_by_pm[pm_id]),
        )
        for pm_id in sorted(skeletons_by_pm)
    )


async def _narrate_one(
    ctx: SessionContext,
    client: CachedClient,
    config: Config,
    advisor_prompt: str,
    semaphore: asyncio.Semaphore,
) -> SessionResult:
    async with semaphore:
        return await narrate_session(ctx, client, config.dialogue, advisor_prompt)


async def _narrate_all(
    contexts: tuple[SessionContext, ...], client: CachedClient, config: Config, advisor_prompt: str
) -> list[SessionResult | BaseException]:
    # Bounding sessions in flight, not just letting `gather` start them all,
    # caps the budget's overshoot and memory use at the concurrency limit.
    semaphore = asyncio.Semaphore(config.dialogue.max_concurrency)
    try:
        return await asyncio.gather(
            *(_narrate_one(ctx, client, config, advisor_prompt, semaphore) for ctx in contexts),
            return_exceptions=True,
        )
    finally:
        # Closed here, inside the loop `asyncio.run` owns, whether the run
        # succeeded or failed: the SDK client's HTTP pool cannot be closed
        # once that loop has torn down.
        await client.aclose()


def _reason_of(ctx: SessionContext, error: PmTraitbenchError) -> str:
    """The failure reason for one session, stripped of a `session {id}: ` prefix if present.

    An error raised inside `narrate_session` already carries that prefix; one
    raised by the inner client (a credential or 400 error) does not, so this
    normalises both to a plain reason before regrouping by session.
    """
    message = str(error)
    prefix = f"session {ctx.skeleton.session_id}: "
    return message[len(prefix) :] if message.startswith(prefix) else message


def raise_on_failure(
    contexts: tuple[SessionContext, ...],
    results: list[Any],
    client: CachedClient,
    *,
    error_type: type[PmTraitbenchError] = DialogueError,
    budget_label: str = "dialogue",
) -> None:
    """Raise a budget error, else a combined error of `error_type`, else re-raise any other
    exception.

    A budget error takes priority since it means the whole run should stop
    spending; its message names `budget_label` so a validate run's budget
    error points at `validation.token_budget`, not the dialogue config.
    Otherwise every failed session is named, in session order, with sessions
    that failed for the identical reason collapsed onto one line, so a
    rerun's cache can skip the sessions that already succeeded. Any
    `PmTraitbenchError` result is grouped this way and raised as
    `error_type`, so the validate stage's `ValidateError` results are
    reported under validate's own error, not dialogue's.
    """
    if any(isinstance(r, DialogueBudgetError) for r in results):
        totals = client.totals
        raise DialogueBudgetError(
            f"{budget_label} token budget spent: {totals.input_tokens} input, "
            f"{totals.output_tokens} output tokens across {totals.calls} calls"
        )

    ids_by_reason: dict[str, list[str]] = {}
    for ctx, result in zip(contexts, results, strict=True):
        if isinstance(result, PmTraitbenchError):
            ids_by_reason.setdefault(_reason_of(ctx, result), []).append(ctx.skeleton.session_id)
    if ids_by_reason:
        lines = []
        for reason, ids in ids_by_reason.items():
            who = f"session {ids[0]}" if len(ids) == 1 else f"sessions {', '.join(ids)}"
            lines.append(f"{who}: {reason}")
        raise error_type("\n".join(lines))

    for result in results:
        if isinstance(result, BaseException):
            raise result


def _run(
    config: Config, store: DataStore, client_factory: Callable[[Config], LlmClient]
) -> dict[str, Any]:
    meta = store.read_run_metadata("plan")
    if meta is None:
        raise DialogueError("plan run metadata is missing; run the plan stage first")

    catalogue = load_catalogue()
    check_dialogue_catalogue(catalogue)
    advisor_prompt = read_advisor_prompt(config.dialogue.advisor_prompt_path)

    all_pm_tables = partition_pm_tables(store)
    selected = select_pms(all_pm_tables, config.dialogue.pm_filter)
    if not selected:
        raise DialogueError("dialogue.pm_filter selects no PMs")

    instruments = store.read(MARKET_INSTRUMENTS)
    prices = store.read(MARKET_PRICES)
    curves = store.read(MARKET_CURVES)
    consensus = store.read(MARKET_CONSENSUS)
    calendar = store.read(MARKET_CALENDAR)
    lookups = {
        seed: MarketLookup.build(seed, instruments, prices, curves, consensus, calendar)
        for seed in sorted({pm.persona.market_seed for pm in selected})
    }

    contexts: list[SessionContext] = []
    voices: dict[str, str] = {}
    for pm in selected:
        voice = draw_voice(config.seed.root, pm.persona.pm_id, catalogue.voices)
        voices[pm.persona.pm_id] = voice.voice_id
        lookup = lookups[pm.persona.market_seed]
        contexts.extend(build_contexts(pm, voice, lookup, catalogue, config))

    cache_dir = store.data_dir / "cache" / "llm"
    client = CachedClient(lambda: client_factory(config), cache_dir, config.dialogue.token_budget)
    frozen_contexts = tuple(contexts)
    results = asyncio.run(_narrate_all(frozen_contexts, client, config, advisor_prompt))
    raise_on_failure(frozen_contexts, results, client)

    session_results = [r for r in results if isinstance(r, SessionResult)]
    if len(session_results) != len(frozen_contexts):
        raise DialogueError(
            f"expected {len(frozen_contexts)} session results but got {len(session_results)}"
        )

    sessions = [r.session for r in session_results]
    logs = [r.log for r in session_results]
    warnings = [w for r in session_results for w in r.warnings]
    rejected_replies = sum(r.rejected_replies for r in session_results)
    session_counts: dict[str, int] = {}
    for result in session_results:
        kind = result.session.kind.value
        session_counts[kind] = session_counts.get(kind, 0) + 1

    store.write(SESSIONS, sessions)
    store.write(DIALOGUE_LOGS, logs)

    totals = client.totals
    return {
        "narrator_model": config.dialogue.narrator_model,
        "advisor_model": config.dialogue.advisor_model,
        "advisor_prompt_sha256": hashlib.sha256(advisor_prompt.encode("utf-8")).hexdigest(),
        "skipped": meta["skipped"],
        "pms": [pm.persona.pm_id for pm in selected],
        "voices": voices,
        "sessions": session_counts,
        "calls": totals.calls,
        "cache_hits": totals.cache_hits,
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
        "cache_read_tokens": totals.cache_read_tokens,
        "warnings": warnings,
        "rejected_replies": rejected_replies,
    }


def make_stage(client_factory: Callable[[Config], LlmClient]) -> Stage:
    """Build the dialogue stage around one client factory, so a test can swap in a fake."""

    def run(config: Config, store: DataStore) -> dict[str, Any]:
        return _run(config, store, client_factory)

    return Stage(
        number=6,
        name="dialogue",
        help="narrate every planned session with a PM narrator and a simulated advisor",
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
        ),
        writes=DIALOGUE_TABLES,
    )


def anthropic_client_factory(config: Config) -> LlmClient:
    return AnthropicClient(config.dialogue.max_concurrency, config.dialogue.api_max_retries)


DIALOGUE_STAGE = make_stage(anthropic_client_factory)
