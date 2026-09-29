"""Replays each PM's sessions and probes into a system under test, in date order.

Each PM is one independent run whose responses land in a part file when the run
finishes, so an interrupted evaluation resumes without redoing completed PMs.
"""

import hashlib
import importlib
import re
import secrets
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pm_traitbench import rng
from pm_traitbench.config import Config
from pm_traitbench.enums import RunStatus
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.protocol import SutFactory
from pm_traitbench.harness.views import PmReplay, pm_replays, public_probe
from pm_traitbench.tables.schema import ResponseRow
from pm_traitbench.tables.specs import PERSONAS, PROBES, RESPONSES, RULES, SESSIONS, parts_spec
from pm_traitbench.tables.store import DataStore

RUN_NAME_PATTERN = r"^[a-z0-9_-]+$"
RUN_METADATA = "eval-run"


def default_run_name(sut: str) -> str:
    """A run directory name derived from the system's import path."""
    return sut.replace(".", "_").replace(":", "_").lower()


def check_run_name(name: str) -> str:
    """The name itself; it becomes a directory, so path separators and spaces are rejected."""
    if not re.fullmatch(RUN_NAME_PATTERN, name):
        raise HarnessError(f"run name '{name}' must match {RUN_NAME_PATTERN}")
    return name


def run_dir(data_dir: Path, run_name: str) -> Path:
    """The directory holding one evaluation run's tables and metadata."""
    return data_dir / "eval" / run_name


def load_factory(path: str) -> SutFactory:
    """Import a `package.module:attr` factory."""
    module_name, sep, attr = path.partition(":")
    if not sep or not module_name or not attr:
        raise HarnessError(f"factory '{path}' must have the form 'package.module:attr'")
    try:
        module = importlib.import_module(module_name)
    except Exception as e:  # import-time errors of any type mean the factory is unusable
        raise HarnessError(f"cannot import module '{module_name}' from '{path}': {e!r}") from e
    try:
        factory = getattr(module, attr)
    except AttributeError as e:
        raise HarnessError(f"module '{module_name}' has no attribute '{attr}'") from e
    if not callable(factory):
        raise HarnessError(f"'{path}' is not callable")
    return factory


def probes_sha256(store: DataStore) -> str:
    """Digest of the probes table file, which identifies the question set a run answered."""
    return hashlib.sha256(store.path(PROBES).read_bytes()).hexdigest()


def check_probes_fresh(store: DataStore) -> None:
    """Fail when sessions changed after the probes stage derived its probes from them."""
    metadata = store.read_run_metadata("probes")
    if metadata is None:
        raise HarnessError("probes run metadata is missing; run the probes stage first")
    current = hashlib.sha256(store.path(SESSIONS).read_bytes()).hexdigest()
    if metadata.get("sessions_sha256") != current:
        raise HarnessError(
            "sessions changed after the probes stage ran; rerun the probes stage so "
            "probes match the sessions"
        )


@dataclass(frozen=True)
class RunResult:
    """Outcome of one `run_sut` call."""

    run_store: DataStore
    completed: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: dict[str, str]


def _replay_pm(replay: PmReplay, factory: SutFactory, key: str) -> list[ResponseRow]:
    """Feed a PM's sessions in date order, answering each checkpoint after its sessions.

    A session dated on a checkpoint day is observed first, so a system never sees
    the future and never misses what was known when it was asked. Probes are asked in
    a shuffle keyed by a per-run secret so the system under test cannot recompute
    order or ids, and emission order cannot reveal answers.
    """
    sut = factory(replay.profile)
    try:
        i = 0
        rows: list[ResponseRow] = []
        for checkpoint in replay.checkpoints:
            while i < len(replay.sessions) and replay.sessions[i].date <= checkpoint.day:
                sut.observe(replay.sessions[i])
                i += 1
            order = rng.stream(
                int(key, 16) % 2**63, "harness", replay.profile.pm_id, checkpoint.day.isoformat()
            ).permutation(len(checkpoint.probes))
            for row in (checkpoint.probes[j] for j in order):
                start = time.perf_counter()
                reply = sut.answer(checkpoint.day, public_probe(row, key))
                latency_ms = int((time.perf_counter() - start) * 1000)
                if not isinstance(reply, str):
                    raise HarnessError(
                        f"probe {row.probe_id}: answer returned {type(reply).__name__}"
                    )
                rows.append(
                    ResponseRow(
                        probe_id=row.probe_id,
                        pm_id=row.pm_id,
                        response=reply,
                        latency_ms=latency_ms,
                    )
                )
        return rows
    finally:
        close = getattr(sut, "close", None)
        if callable(close):
            close()


def run_sut(
    config: Config,
    store: DataStore,
    factory: SutFactory,
    *,
    sut_name: str,
    run_name: str,
    workers: int = 1,
    force: bool = False,
) -> RunResult:
    """Replay every PM with probes into fresh systems and merge their responses.

    PMs with an existing part file are skipped; `force` clears the run first.
    """
    check_run_name(run_name)
    if workers < 1:
        raise HarnessError(f"workers must be at least 1, got {workers}")
    check_probes_fresh(store)
    digest = probes_sha256(store)
    rd = run_dir(store.data_dir, run_name)
    if force:
        shutil.rmtree(rd, ignore_errors=True)
    run_store = DataStore(rd, config.output)

    previous = run_store.read_run_metadata(RUN_METADATA)
    if previous is not None:
        current_harness = config.harness.model_dump(mode="json")
        if previous.get("probes_sha256") != digest:
            reason = "different probes"
        elif previous.get("sut") != sut_name:
            reason = f"system '{previous.get('sut')}'"
        elif previous.get("config", {}).get("harness") != current_harness:
            reason = "a different harness config"
        else:
            reason = None
        if reason is not None:
            raise HarnessError(
                f"run '{run_name}' was made from {reason}; rerun with --force to replace it"
            )
    # Reused on resume so a PM's ids and order stay stable across reruns.
    probe_key = (previous or {}).get("probe_key") or secrets.token_hex(16)

    # Recorded before any part is written, so an interrupted run still pins its probes.
    def write_metadata(status: RunStatus, completed, skipped, failed, seconds) -> None:
        run_store.write_run_metadata(
            RUN_METADATA,
            config,
            {
                "status": status,
                "sut": sut_name,
                "run_name": run_name,
                "probes_sha256": digest,
                "probe_key": probe_key,
                "pms_completed": completed,
                "pms_skipped": skipped,
                "pms_failed": dict(sorted(failed.items())),
                "pm_seconds": dict(sorted(seconds.items())),
            },
        )

    write_metadata(RunStatus.RUNNING, [], [], {}, {})
    replays = pm_replays(
        store.read(PERSONAS), store.read(RULES), store.read(SESSIONS), store.read(PROBES)
    )
    skipped = {pm for pm in replays if run_store.exists(parts_spec(pm))}
    pending = [pm for pm in replays if pm not in skipped]
    failed: dict[str, str] = {}
    seconds: dict[str, float] = {}

    def run_pm(pm_id: str) -> None:
        start = time.perf_counter()
        try:
            rows = _replay_pm(replays[pm_id], factory, probe_key)
            run_store.write(parts_spec(pm_id), rows)
        except Exception:
            failed[pm_id] = traceback.format_exc()
        seconds[pm_id] = round(time.perf_counter() - start, 3)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_pm, pending))

    completed = [pm for pm in replays if run_store.exists(parts_spec(pm))]
    merged: list[ResponseRow] = []
    for pm_id in completed:
        merged.extend(run_store.read(parts_spec(pm_id)))
    run_store.write(RESPONSES, merged)
    write_metadata(RunStatus.FINISHED, completed, sorted(skipped), failed, seconds)
    return RunResult(
        run_store=run_store,
        completed=tuple(sorted(completed)),
        skipped=tuple(sorted(skipped)),
        failed=dict(sorted(failed.items())),
    )
