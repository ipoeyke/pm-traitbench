"""Gate 1 stage: recover each planted bias per PM, pool per asset class, and
decide pass or fail against the neutral baseline the engine itself produced.

Reads the engine's four tables plus the market tables `build_views` needs to
rebuild each idea's series, and blocks the pipeline only when the pooled
synthetic comparison over the full horizon fails or is insufficient.
"""

from typing import Any

from pm_traitbench.config import Config
from pm_traitbench.engine.stage import build_views
from pm_traitbench.enums import AssetClass
from pm_traitbench.errors import Gate1Error
from pm_traitbench.gates.gate1.aggregate import aggregate, estimate_all
from pm_traitbench.gates.gate1.checks import check_counts
from pm_traitbench.gates.gate1.inputs import build_inputs
from pm_traitbench.gates.gate1.verdict import blocking_failures, count_warnings, judge
from pm_traitbench.stages import Stage
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    ENGINE_TABLES,
    GATE1_CELLS,
    GATE1_PM,
    GATE1_TABLES,
    IDEAS,
    LEDGER,
    MARKET_TABLES,
    PERSONAS,
    POSITION_DAYS,
    RULE_EVENTS,
    TRAITS,
)
from pm_traitbench.tables.store import DataStore


def run(config: Config, store: DataStore) -> dict[str, Any]:
    """Recover every bias parameter for every direct-asset PM and pool by asset class.

    Raises `Gate1Error` when the engine's run metadata is missing, or when a
    recomputed opportunity count disagrees with the engine's own counter for
    some PM. Never raises on a verdict: the caller's `verdict` hook does that
    once this run's tables and metadata are on disk.
    """
    meta = store.read_run_metadata("engine")
    if meta is None:
        raise Gate1Error("engine run metadata is missing; run the engine stage first")

    personas = store.read(PERSONAS)
    traits = store.read(TRAITS)
    drift_events = store.read(DRIFT_EVENTS)
    ideas = store.read(IDEAS)
    ledger = store.read(LEDGER)
    rule_events = store.read(RULE_EVENTS)
    position_days = store.read(POSITION_DAYS)

    skipped = meta["skipped"]
    kept_seeds = (
        persona.market_seed
        for persona in personas
        if persona.mandate.asset_class != AssetClass.MULTI_ASSET and persona.pm_id not in skipped
    )
    views = build_views(config, store, kept_seeds)

    inputs = build_inputs(
        config,
        personas,
        traits,
        drift_events,
        ideas,
        ledger,
        rule_events,
        position_days,
        views,
        engine_counts=meta["opportunities"],
        skipped=skipped,
    )
    for pm_inputs in inputs:
        check_counts(pm_inputs)

    estimates = estimate_all(inputs, config.gate1)
    cells = aggregate(
        estimates,
        engine_counts=meta["opportunities"],
        synthetic_seeds=set(config.market.seeds),
        real_seeds=set(config.market.real.seeds),
        knobs=config.gate1,
    )
    rows = [judge(cell, config.gate1) for cell in cells]

    store.write(GATE1_PM, [estimate.to_row() for estimate in estimates])
    store.write(GATE1_CELLS, rows)

    return {
        "failed": blocking_failures(rows),
        "warnings": count_warnings(rows),
        "thresholds": config.gate1.model_dump(mode="json"),
    }


def raise_on_failures(extra: dict[str, Any]) -> None:
    """Raise `Gate1Error` naming every failed blocking cell, if any."""
    failed = extra["failed"]
    if failed:
        raise Gate1Error("gate 1 failed for: " + ", ".join(failed))


GATE1_STAGE = Stage(
    number=4,
    name="gate1",
    help="recover planted biases from the engine's ledger per asset class",
    run=run,
    reads=(PERSONAS, TRAITS, DRIFT_EVENTS, *ENGINE_TABLES, *MARKET_TABLES),
    writes=GATE1_TABLES,
    verdict=raise_on_failures,
)
