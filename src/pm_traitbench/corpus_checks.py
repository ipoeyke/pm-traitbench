"""Checks that a stage reading the narrated corpus runs on a validated, current one."""

from collections.abc import Collection
from datetime import datetime

from pm_traitbench.errors import CorpusError
from pm_traitbench.tables.store import DataStore


def check_validated_corpus(store: DataStore, session_pm_ids: Collection[str]) -> dict:
    """The validate run's metadata, after checking the corpus may be read at all.

    Raises `CorpusError` when the validate or dialogue run metadata is missing, when
    sessions were re-narrated after the last validate run, or when a PM with sessions
    is absent from the validate run's own PM list.
    """
    validate_meta = store.read_run_metadata("validate")
    if validate_meta is None:
        raise CorpusError("validate run metadata is missing; run the validate stage first")
    dialogue_meta = store.read_run_metadata("dialogue")
    if dialogue_meta is None:
        raise CorpusError("dialogue run metadata is missing; run the dialogue stage first")

    validate_created_at = datetime.fromisoformat(validate_meta["created_at"])
    dialogue_created_at = datetime.fromisoformat(dialogue_meta["created_at"])
    if validate_created_at < dialogue_created_at:
        raise CorpusError("sessions were re-narrated after the last validate run; rerun validate")

    missing = sorted(set(session_pm_ids) - set(validate_meta["pms"]))
    if missing:
        raise CorpusError(
            f"pm(s) {', '.join(missing)} have sessions but are not in validate run metadata's pms"
        )
    return validate_meta
