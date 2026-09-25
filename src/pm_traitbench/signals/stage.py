"""Plan stage: places every PM's planted trait signals on dated sessions and renders one
narrator skeleton per session.

Each direct-asset PM is planned independently, so a signal never crosses a PM
boundary and one PM's random draws never perturb another's.
"""

from typing import Any

from pm_traitbench.catalogues.loader import check_stances, load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.errors import PlanError
from pm_traitbench.rng import stream
from pm_traitbench.signals.assemble import assemble
from pm_traitbench.signals.carriers import carrier_pools
from pm_traitbench.signals.inputs import build_inputs, trading_days
from pm_traitbench.signals.quotas import plan_quotas
from pm_traitbench.signals.skeleton import render_skeletons
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import Signal, Skeleton
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    ENGINE_TABLES,
    IDEAS,
    LEDGER,
    PERSONAS,
    PLAN_TABLES,
    POSITION_DAYS,
    RULE_EVENTS,
    SIGNALS,
    SKELETONS,
    TRAITS,
)
from pm_traitbench.tables.store import DataStore


def run(config: Config, store: DataStore) -> dict[str, Any]:
    """Plant trait signals on dated sessions and write one narrator skeleton per session.

    Raises `PlanError` when the engine's run metadata is missing. Never raises on a
    shortfall: every warning `assemble` produces for a PM is collected in the
    returned extras instead of blocking the run.
    """
    meta = store.read_run_metadata("engine")
    if meta is None:
        raise PlanError("engine run metadata is missing; run the engine stage first")

    catalogue = load_catalogue()
    check_stances(catalogue)

    skipped = meta["skipped"]
    inputs = build_inputs(
        store.read(PERSONAS),
        store.read(TRAITS),
        store.read(DRIFT_EVENTS),
        store.read(IDEAS),
        store.read(LEDGER),
        store.read(RULE_EVENTS),
        store.read(POSITION_DAYS),
        days=trading_days(config),
        skipped=skipped,
    )

    root = config.seed.root
    all_signals: list[Signal] = []
    all_skeletons: list[Skeleton] = []
    warnings: list[str] = []
    pm_counts: dict[str, Any] = {}

    for pm in inputs:
        pm_id = pm.persona.pm_id
        pools = carrier_pools(pm)
        planned = plan_quotas(
            pm, pools, catalogue, config.plan, stream(root, "plan", pm_id, "quotas")
        )
        assembly = assemble(
            pm, planned, pools, config.plan, stream(root, "plan", pm_id, "assemble")
        )
        skeletons = render_skeletons(
            pm, assembly, catalogue, stream(root, "plan", pm_id, "skeleton")
        )

        all_signals.extend(assembly.signals)
        all_skeletons.extend(skeletons)
        warnings.extend(assembly.warnings)
        pm_counts[pm_id] = assembly.counts

    store.write(SIGNALS, all_signals)
    store.write(SKELETONS, all_skeletons)

    return {
        "skipped": sorted(skipped),
        "pms": pm_counts,
        "warnings": warnings,
    }


PLAN_STAGE = Stage(
    number=5,
    name="plan",
    help="plant trait signals on dated sessions and write one narrator skeleton per session",
    run=run,
    reads=(PERSONAS, TRAITS, DRIFT_EVENTS, *ENGINE_TABLES),
    writes=PLAN_TABLES,
)
