"""Tests for the stage runner: read/write preconditions and postconditions."""

import json
from pathlib import Path

import pytest

from pm_traitbench.config import Config, OutputConfig
from pm_traitbench.enums import Kind
from pm_traitbench.errors import StageIOError
from pm_traitbench.stages import Stage, run_stage
from pm_traitbench.tables.schema import Trait
from pm_traitbench.tables.specs import PERSONAS, TRAITS
from pm_traitbench.tables.store import DataStore


def _trait(pm_id: str, trait_id: str) -> Trait:
    return Trait(
        pm_id=pm_id,
        trait_id=trait_id,
        kind=Kind.BIAS,
        param="loss_aversion_lambda",
        value=2.6,
        active=True,
        mult_range=1.1,
        mult_risk_off=1.3,
        mult_risk_on=0.9,
    )


def _write_two_traits(config: Config, store: DataStore) -> None:
    store.write(TRAITS, [_trait("pm_001", "t_01"), _trait("pm_002", "t_01")])


def test_run_stage_happy_path_writes_table_and_metadata(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    stage = Stage(number=1, name="fake", help="fake stage", run=_write_two_traits, writes=(TRAITS,))

    run_stage(stage, config, store)

    assert store.exists(TRAITS)
    assert len(store.read(TRAITS)) == 2
    metadata_path = tmp_path / "run_metadata" / "fake.json"
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text())["stage"] == "fake"


def test_run_stage_missing_read_raises_and_never_calls_run(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    called = False

    def _run(config: Config, store: DataStore) -> None:
        nonlocal called
        called = True

    stage = Stage(number=1, name="fake", help="fake stage", run=_run, reads=(PERSONAS,))

    with pytest.raises(StageIOError, match="personas"):
        run_stage(stage, config, store)
    assert called is False


def test_run_stage_existing_output_without_force_raises_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write(TRAITS, [_trait("pm_000", "t_00")])
    original_bytes = (tmp_path / "traits.jsonl").read_bytes()
    stage = Stage(number=1, name="fake", help="fake stage", run=_write_two_traits, writes=(TRAITS,))

    with pytest.raises(StageIOError, match="--force"):
        run_stage(stage, config, store)

    assert (tmp_path / "traits.jsonl").read_bytes() == original_bytes


def test_run_stage_with_force_overwrites_existing_output(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write(TRAITS, [_trait("pm_000", "t_00")])
    stage = Stage(number=1, name="fake", help="fake stage", run=_write_two_traits, writes=(TRAITS,))

    run_stage(stage, config, store, force=True)

    assert len(store.read(TRAITS)) == 2


def test_run_stage_writes_nothing_raises_after_run(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    called = False

    def _run(config: Config, store: DataStore) -> None:
        nonlocal called
        called = True

    stage = Stage(number=1, name="fake", help="fake stage", run=_run, writes=(TRAITS,))

    with pytest.raises(StageIOError, match="fake"):
        run_stage(stage, config, store)
    assert called is True
    assert not (tmp_path / "run_metadata" / "fake.json").exists()


def test_run_stage_raising_leaves_no_metadata(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()

    def _run(config: Config, store: DataStore) -> None:
        raise RuntimeError("boom")

    stage = Stage(number=1, name="fake", help="fake stage", run=_run, writes=(TRAITS,))

    with pytest.raises(RuntimeError, match="boom"):
        run_stage(stage, config, store)
    assert not (tmp_path / "run_metadata" / "fake.json").exists()
    assert not store.exists(TRAITS)


def test_run_stage_with_force_rejects_a_stage_that_skips_a_table_and_keeps_the_old_file(
    tmp_path: Path,
) -> None:
    config = Config()
    store = DataStore(tmp_path, config.output)
    writer = Stage(number=1, name="fake", help="h", run=_write_two_traits, writes=(TRAITS,))
    run_stage(writer, config, store)
    before = store.path(TRAITS).read_bytes()
    metadata = tmp_path / "run_metadata" / "fake.json"
    metadata_before = metadata.read_bytes()

    def _writes_nothing(config: Config, store: DataStore) -> None:
        return None

    skipper = Stage(number=1, name="fake", help="h", run=_writes_nothing, writes=(TRAITS,))
    with pytest.raises(StageIOError, match="did not write"):
        run_stage(skipper, config, DataStore(tmp_path, config.output), force=True)

    assert store.path(TRAITS).read_bytes() == before
    assert metadata.read_bytes() == metadata_before
