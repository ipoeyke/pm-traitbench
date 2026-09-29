"""Probes stage: write probes with deterministic ground truth at every checkpoint of each
narrated PM. No model is called; the questions come from an authored bank.
"""

import hashlib
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from pm_traitbench.catalogues.loader import check_probes_catalogue, load_catalogue
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.corpus_checks import check_validated_corpus
from pm_traitbench.engine.adapters import adapter_for
from pm_traitbench.engine.stage import build_views
from pm_traitbench.enums import Kind, ProbeSkip, RuleScope
from pm_traitbench.errors import CorpusError, ProbesError
from pm_traitbench.probes.actions import horizons
from pm_traitbench.probes.builders import (
    Draft,
    PmInputs,
    governance_drafts,
    in_situ_drafts,
    mcq_drafts,
    presence_drafts,
    profile_params,
    routine_drafts,
)
from pm_traitbench.probes.checkpoints import checkpoints_for
from pm_traitbench.probes.context import context_chars, in_context, sessions_until
from pm_traitbench.probes.situations import MarketEnv, day_index
from pm_traitbench.rng import stream
from pm_traitbench.signals.inputs import trading_days
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import ProbeRow, probe_id
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    MARKET_REGIMES,
    MARKET_TABLES,
    PERSONAS,
    PROBES,
    PROBES_TABLES,
    RULES,
    SESSIONS,
    SIGNALS,
    TRAITS,
    TableSpec,
)
from pm_traitbench.tables.store import DataStore

# Every table the stage reads to rebuild each PM's context, ground truth and market.
PROBES_READS: tuple[TableSpec, ...] = (
    PERSONAS,
    TRAITS,
    DRIFT_EVENTS,
    RULES,
    SIGNALS,
    SESSIONS,
    *MARKET_TABLES,
)


def _group_by_pm[T](rows: list[T], pm_id_of: Callable[[T], str]) -> dict[str, list[T]]:
    grouped: dict[str, list[T]] = {}
    for row in rows:
        grouped.setdefault(pm_id_of(row), []).append(row)
    return grouped


def _row(pm_id: str, n: int, label, day, chars: int, draft: Draft) -> ProbeRow:
    a, b, c, d = draft.options
    sa, sb, sc, sd = (draft.sources + (None,) * 4)[:4]
    return ProbeRow(
        probe_id=probe_id(pm_id, n),
        pm_id=pm_id,
        checkpoint_date=day,
        checkpoint_label=label,
        probe_type=draft.probe_type,
        trait_id=draft.trait_id,
        form=draft.form,
        question=draft.question,
        option_a=a,
        option_b=b,
        option_c=c,
        option_d=d,
        answer=draft.answer,
        source_a=sa,
        source_b=sb,
        source_c=sc,
        source_d=sd,
        supporting_signal_ids=draft.supporting_signal_ids,
        context_chars=chars,
    )


def run(config: Config, store: DataStore) -> dict[str, Any]:
    """Build every narrated PM's probes at each checkpoint and write the `probes` table."""
    sessions = store.read(SESSIONS)
    sessions_sha256 = hashlib.sha256(store.path(SESSIONS).read_bytes()).hexdigest()
    session_pm_ids = sorted({s.pm_id for s in sessions})
    try:
        validate_meta = check_validated_corpus(store, session_pm_ids)
    except CorpusError as e:
        raise ProbesError(str(e)) from e
    pms_without_sessions = sorted(set(validate_meta["pms"]) - set(session_pm_ids))

    catalogue = load_catalogue()
    check_probes_catalogue(catalogue)
    bank = catalogue.probes
    mcq_horizon = horizons(config)

    personas = {p.pm_id: p for p in store.read(PERSONAS)}
    traits_by_pm = _group_by_pm(store.read(TRAITS), lambda t: t.pm_id)
    drift_by_pm = _group_by_pm(store.read(DRIFT_EVENTS), lambda d: d.pm_id)
    rules_by_pm = _group_by_pm(store.read(RULES), lambda r: r.pm_id)
    signals_by_pm = _group_by_pm(store.read(SIGNALS), lambda s: s.pm_id)
    sessions_by_pm = _group_by_pm(sessions, lambda s: s.pm_id)
    regimes = store.read(MARKET_REGIMES)

    views = build_views(config, store, (personas[pm].market_seed for pm in session_pm_ids))
    days = trading_days(config)
    timeline = config.timeline()

    rows: list[ProbeRow] = []
    by_type: dict[str, dict[str, int]] = {}
    skipped_checkpoints: list[str] = []
    skipped_probes: Counter[ProbeSkip] = Counter()

    for pm_id in session_pm_ids:
        persona = personas[pm_id]
        traits = tuple(traits_by_pm.get(pm_id, ()))
        missing = set(BIAS_PARAMS) - {t.param for t in traits if t.kind == Kind.BIAS}
        if missing:
            raise ProbesError(f"pm {pm_id}: missing bias trait(s) {sorted(missing)}")
        pm_rules = tuple(r for r in rules_by_pm.get(pm_id, ()) if r.scope == RuleScope.PM)
        pm = PmInputs(
            persona=persona,
            traits=traits,
            drift_events=tuple(drift_by_pm.get(pm_id, ())),
            pm_rules=pm_rules,
            entries=catalogue.preferences_for(persona.mandate.asset_class),
            profile_params=profile_params(traits, config),
        )
        view = views[persona.market_seed]
        adapter = adapter_for(
            persona.mandate.asset_class, persona.mandate.sub_style, config.engine.horizon_days
        )
        env = MarketEnv(
            view=view,
            adapter=adapter,
            universe=adapter.universe(view.instruments, pm_rules),
            asset_class=persona.mandate.asset_class,
        )
        checkpoints = checkpoints_for(
            pm.drift_events,
            [r for r in regimes if r.seed == persona.market_seed],
            timeline,
            days,
            config.probes.post_drift_weeks,
        )

        n = 0
        counts: Counter[str] = Counter()
        for cp in checkpoints:
            seen = sessions_until(sessions_by_pm[pm_id], cp.day)
            if not seen:
                skipped_checkpoints.append(f"{pm_id}:{cp.day}:{cp.label.value}")
                continue
            signals = in_context(signals_by_pm.get(pm_id, ()), {s.session_id for s in seen}, cp.day)
            chars = context_chars(seen)
            t = day_index(view, cp.day)

            def rng_for(
                purpose: str, *keys: str | int, pm_id: str = pm_id, day=cp.day
            ) -> np.random.Generator:
                return stream(config.seed.root, "probes", pm_id, day.isoformat(), purpose, *keys)

            built = (
                presence_drafts(pm, cp, signals, bank, rng_for, config),
                mcq_drafts(pm, cp, signals, bank, rng_for, env, t, mcq_horizon, config),
                in_situ_drafts(pm, cp, signals, bank, rng_for, env, t, config),
                routine_drafts(pm, cp, signals, bank, rng_for, env, t, config),
                governance_drafts(pm, cp, signals, bank, rng_for),
            )
            for drafts, skips in built:
                skipped_probes.update(skips)
                for draft in drafts:
                    n += 1
                    counts[draft.probe_type.value] += 1
                    rows.append(_row(pm_id, n, cp.label, cp.day, chars, draft))
        by_type[pm_id] = dict(counts)

    store.write(PROBES, rows)

    return {
        "pms": session_pm_ids,
        "pms_without_sessions": pms_without_sessions,
        "sessions_sha256": sessions_sha256,
        "probes_by_type": by_type,
        "skipped_checkpoints": skipped_checkpoints,
        "skipped_probes": {reason.value: count for reason, count in skipped_probes.items()},
        "mcq_horizon": mcq_horizon,
    }


PROBES_STAGE = Stage(
    number=9,
    name="probes",
    help="write probes with deterministic ground truth at every checkpoint of each narrated PM",
    run=run,
    reads=PROBES_READS,
    writes=PROBES_TABLES,
)
