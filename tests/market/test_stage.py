"""Tests for the market stage: universe, per-seed generation, check and table writes."""

import json
from pathlib import Path

import pytest

from pm_traitbench import pipeline
from pm_traitbench.cli import main
from pm_traitbench.config import Config, load_config
from pm_traitbench.errors import MarketCheckError, StageIOError
from pm_traitbench.market.stage import MARKET_STAGE, referenced_seeds
from pm_traitbench.market.synthetic.check import check_market as real_check_market
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.specs import MARKET_TABLES
from pm_traitbench.tables.store import DataStore

_DEMO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"


def _demo_config():
    return load_config(_DEMO_CONFIG)


def test_run_stage_writes_the_six_market_tables_and_they_read_back(tmp_path: Path) -> None:
    config = _demo_config()
    store = DataStore(tmp_path, config.output)
    run_stage(MARKET_STAGE, config, store)

    for spec in MARKET_TABLES:
        assert store.path(spec).exists()
        assert store.path(spec).parent == tmp_path / "market"
        store.read(spec)  # validates and checks for duplicate keys without raising


def test_run_stage_writes_run_metadata_with_check_for_every_seed(tmp_path: Path) -> None:
    config = _demo_config()
    store = DataStore(tmp_path, config.output)
    run_stage(MARKET_STAGE, config, store)

    metadata_path = tmp_path / "run_metadata" / "market.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "A" in metadata["check"]


def test_parquet_override_for_prices_writes_market_prices_parquet(tmp_path: Path) -> None:
    config = _demo_config()
    output = config.output.model_copy(update={"tables": {"market/prices": "parquet"}})
    config = config.model_copy(update={"output": output})
    store = DataStore(tmp_path, config.output)
    run_stage(MARKET_STAGE, config, store)

    assert (tmp_path / "market" / "prices.parquet").exists()
    assert (tmp_path / "market" / "curves.jsonl").exists()


def test_second_run_without_force_raises_stage_io_error(tmp_path: Path) -> None:
    config = _demo_config()
    store = DataStore(tmp_path, config.output)
    run_stage(MARKET_STAGE, config, store)

    with pytest.raises(StageIOError):
        run_stage(MARKET_STAGE, config, store)


def test_second_run_with_force_succeeds(tmp_path: Path) -> None:
    config = _demo_config()
    store = DataStore(tmp_path, config.output)
    run_stage(MARKET_STAGE, config, store)
    run_stage(MARKET_STAGE, config, store, force=True)

    assert (tmp_path / "market" / "instruments.jsonl").exists()


def test_cli_runs_market_stage_and_returns_0(tmp_path: Path) -> None:
    result = main(["market", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
    assert result == 0
    for spec in MARKET_TABLES:
        assert (tmp_path / f"{spec.name}.jsonl").exists()


def test_cli_returns_1_and_writes_no_market_files_when_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args, **kwargs):
        raise MarketCheckError("rigged failure")

    monkeypatch.setattr("pm_traitbench.market.stage.check_market", _raise)

    result = main(["market", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
    assert result == 1

    for spec in MARKET_TABLES:
        assert not (tmp_path / f"{spec.name}.jsonl").exists()


def test_cli_returns_1_and_writes_nothing_when_a_later_seed_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed A's check passing must not let seed B's failure leave a partial write."""
    demo_text = _DEMO_CONFIG.read_text(encoding="utf-8")
    two_seed_text = demo_text.replace("market_seeds: [A]", "market_seeds: [A, B]")
    assert two_seed_text != demo_text
    config_path = tmp_path / "two_seed_config.yaml"
    config_path.write_text(two_seed_text, encoding="utf-8")

    def _fail_only_on_b(market, instruments, config):
        if market.seed == "B":
            raise MarketCheckError("rigged failure on seed B")
        return real_check_market(market, instruments, config)

    monkeypatch.setattr("pm_traitbench.market.stage.check_market", _fail_only_on_b)

    data_dir = tmp_path / "data"
    result = main(["market", "--config", str(config_path), "--data-dir", str(data_dir)])

    assert result == 1
    assert not (data_dir / "market").exists()
    assert not (data_dir / "run_metadata" / "market.json").exists()


def test_pipeline_stage_names_are_sample_then_market() -> None:
    assert tuple(stage.name for stage in pipeline.STAGES) == ("sample", "market")


def test_run_stage_raises_stage_io_error_for_unfetched_real_pilot_seed(tmp_path: Path) -> None:
    config = Config()  # binding defaults pilot the population on real seed R1
    store = DataStore(tmp_path, config.output)

    with pytest.raises(StageIOError, match="run fetch-market first"):
        run_stage(MARKET_STAGE, config, store)

    assert not (tmp_path / "market").exists()
    assert not (tmp_path / "run_metadata").exists()


def test_referenced_seeds_orders_pilot_first_and_dedupes() -> None:
    config = Config.model_validate(
        {"population": {"pilot_market_seeds": ["A"], "market_seeds": ["A", "B", "C"]}}
    )
    assert referenced_seeds(config) == ["A", "B", "C"]
