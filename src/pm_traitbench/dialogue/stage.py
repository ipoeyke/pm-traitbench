"""Dialogue stage: narrates every session the plan stage skeletoned and writes the public
`sessions` table and the hidden `dialogue_logs` table.

Every session narrates through one shared `CachedClient`, with at most
`config.dialogue.max_concurrency` sessions in flight at once, so a crash or a
rejected reply on one session never blocks the others and only that many
sessions' growing histories sit in memory; the whole run still writes both
tables or neither.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from pm_traitbench.catalogues.loader import check_dialogue_catalogue, load_catalogue
from pm_traitbench.catalogues.models import Catalogue, Voice
from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import AnthropicClient, CachedClient, LlmClient, scope_prefix
from pm_traitbench.dialogue.context import PmTables, SessionContext, build_contexts, select_pms
from pm_traitbench.dialogue.progress import Progress
from pm_traitbench.dialogue.prompts import prompt_sha256, read_advisor_prompt
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
    TableSpec,
)
from pm_traitbench.tables.store import DataStore

# Every table stage 6 reads to rebuild each session's context; stage 7 rebuilds the same.
DIALOGUE_READS: tuple[TableSpec, ...] = (
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
)


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


def build_lookups(store: DataStore, seeds: Iterable[str]) -> dict[str, MarketLookup]:
    """One `MarketLookup` per market seed, all built from a single read of the market tables."""
    instruments = store.read(MARKET_INSTRUMENTS)
    prices = store.read(MARKET_PRICES)
    curves = store.read(MARKET_CURVES)
    consensus = store.read(MARKET_CONSENSUS)
    calendar = store.read(MARKET_CALENDAR)
    return {
        seed: MarketLookup.build(seed, instruments, prices, curves, consensus, calendar)
        for seed in sorted(set(seeds))
    }


def pm_contexts(
    pm: PmTables, lookups: Mapping[str, MarketLookup], catalogue: Catalogue, config: Config
) -> tuple[Voice, tuple[SessionContext, ...]]:
    """One PM's drawn voice and a context per skeleton, on its market seed's lookup."""
    voice = draw_voice(config.seed.root, pm.persona.pm_id, catalogue.voices)
    lookup = lookups[pm.persona.market_seed]
    return voice, build_contexts(pm, voice, lookup, catalogue, config)


def stage_client(
    store: DataStore,
    client_factory: Callable[[Config], LlmClient],
    config: Config,
    token_budget: int | None,
) -> CachedClient:
    """A `CachedClient` over the store's shared LLM cache, building its backend on first use."""
    return CachedClient(
        lambda: client_factory(config), store.data_dir / "cache" / "llm", token_budget
    )


def collect_results[T](
    results: Iterable[T | BaseException],
    result_type: type[T],
    expected: int,
    error_type: type[PmTraitbenchError],
) -> list[T]:
    """The results of `result_type`, raising `error_type` unless there are `expected` of them."""
    collected = [r for r in results if isinstance(r, result_type)]
    if len(collected) != expected:
        raise error_type(f"expected {expected} results but got {len(collected)}")
    return collected


async def run_bounded[I, T](
    items: Iterable[I],
    worker: Callable[[I], Awaitable[T]],
    client: CachedClient,
    max_concurrency: int,
    *,
    label: str,
    unit: str,
    unit_name: Callable[[I], str],
) -> list[T | BaseException]:
    """Run `worker` over `items` with at most `max_concurrency` in flight, then close `client`.

    Results come back in `items` order, with a failure returned in place, not raised.
    A progress line named `label` and counting `unit`s prints as each finishes, led for a
    failure by a line naming the item by `unit_name` (e.g. `session s_001`) and its reason.
    """
    pending = tuple(items)
    progress = Progress(label, unit, len(pending), client.totals, client.token_budget)
    # Bounding sessions in flight, not just letting `gather` start them all,
    # caps the budget's overshoot and memory use at the concurrency limit.
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(item: I) -> T:
        async with semaphore:
            try:
                result = await worker(item)
            except BaseException as error:
                progress.finish(unit_name(item), error)
                raise
            progress.finish(unit_name(item))
            return result

    progress.start(max_concurrency)
    try:
        return await asyncio.gather(*(bounded(item) for item in pending), return_exceptions=True)
    finally:
        # Closed here, inside the loop `asyncio.run` owns, whether the run
        # succeeded or failed: the SDK client's HTTP pool cannot be closed
        # once that loop has torn down.
        await client.aclose()


def _reason_of(unit_id: str, error: PmTraitbenchError, label: str) -> str:
    """The failure reason for one unit, stripped of a `{label} {id}: ` prefix if present.

    Normalises a reason already prefixed by the unit's own worker and one raised bare
    by the inner client to the same plain form, so both group by unit consistently.
    """
    message = str(error)
    prefix = scope_prefix(label, unit_id)
    return message[len(prefix) :] if message.startswith(prefix) else message


def raise_on_failure(
    unit_ids: tuple[str, ...],
    results: list[Any],
    client: CachedClient,
    *,
    error_type: type[PmTraitbenchError] = DialogueError,
    budget_label: str = "dialogue",
    label: str = "session",
) -> None:
    """Raise a budget error, else a combined error of `error_type`, else re-raise any other
    exception.

    A budget error (named by `budget_label`) takes priority; otherwise every failed unit
    is named under `label` singular or `label + "s"` plural, in `unit_ids` order, with
    units sharing a reason collapsed onto one line and raised together as `error_type`.
    """
    if any(isinstance(r, DialogueBudgetError) for r in results):
        totals = client.totals
        raise DialogueBudgetError(
            f"{budget_label} token budget spent: {totals.input_tokens} input, "
            f"{totals.output_tokens} output tokens across {totals.calls} calls"
        )

    ids_by_reason: dict[str, list[str]] = {}
    for unit_id, result in zip(unit_ids, results, strict=True):
        if isinstance(result, PmTraitbenchError):
            ids_by_reason.setdefault(_reason_of(unit_id, result, label), []).append(unit_id)
    if ids_by_reason:
        lines = []
        for reason, ids in ids_by_reason.items():
            who = f"{label} {ids[0]}" if len(ids) == 1 else f"{label}s {', '.join(ids)}"
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

    lookups = build_lookups(store, (pm.persona.market_seed for pm in selected))

    contexts: list[SessionContext] = []
    voices: dict[str, str] = {}
    for pm in selected:
        voice, pm_session_contexts = pm_contexts(pm, lookups, catalogue, config)
        voices[pm.persona.pm_id] = voice.voice_id
        contexts.extend(pm_session_contexts)

    client = stage_client(store, client_factory, config, config.dialogue.token_budget)
    frozen_contexts = tuple(contexts)

    async def narrate(ctx: SessionContext) -> SessionResult:
        return await narrate_session(ctx, client, config.dialogue, advisor_prompt)

    results = asyncio.run(
        run_bounded(
            frozen_contexts,
            narrate,
            client,
            config.dialogue.max_concurrency,
            label="dialogue",
            unit="sessions",
            unit_name=lambda ctx: f"session {ctx.skeleton.session_id}",
        )
    )
    raise_on_failure(tuple(ctx.skeleton.session_id for ctx in frozen_contexts), results, client)

    session_results = collect_results(results, SessionResult, len(frozen_contexts), DialogueError)

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

    return {
        "narrator_model": config.dialogue.narrator_model,
        "advisor_model": config.dialogue.advisor_model,
        "advisor_prompt_sha256": prompt_sha256(advisor_prompt),
        "skipped": meta["skipped"],
        "pms": [pm.persona.pm_id for pm in selected],
        "voices": voices,
        "sessions": session_counts,
        **client.totals.as_metadata(),
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
        reads=DIALOGUE_READS,
        writes=DIALOGUE_TABLES,
    )


def anthropic_client_factory(config: Config) -> LlmClient:
    return AnthropicClient(config.dialogue.max_concurrency, config.dialogue.api_max_retries)


DIALOGUE_STAGE = make_stage(anthropic_client_factory)
