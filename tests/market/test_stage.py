"""Tests for the market stage: universe, per-seed generation, check and table writes."""

import json
import re
from pathlib import Path

import pytest

from pm_traitbench import pipeline
from pm_traitbench.cli import main
from pm_traitbench.config import Config, load_config
from pm_traitbench.enums import Family, InstrumentKind
from pm_traitbench.errors import MarketCheckError, StageIOError
from pm_traitbench.market.real.fetch import cache_dir
from pm_traitbench.market.stage import MARKET_STAGE, _merge_instruments, referenced_seeds
from pm_traitbench.market.synthetic.check import check_market as real_check_market
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.schema import Instrument
from pm_traitbench.tables.specs import MARKET_INSTRUMENTS, MARKET_PRICES, MARKET_TABLES
from pm_traitbench.tables.store import DataStore
from tests.market.real.conftest import fake_cache  # noqa: F401

_DEMO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"


def _demo_config():
    return load_config(_DEMO_CONFIG)


def test_run_stage_writes_the_six_market_tables_and_they_read_back(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)
    run_stage(MARKET_STAGE, config, store)

    for spec in MARKET_TABLES:
        assert store.path(spec).exists()
        assert store.path(spec).parent == result.data_dir / "market"
        store.read(spec)  # validates and checks for duplicate keys without raising


def test_instruments_contain_both_universes_with_shared_commodity_deduped(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)
    run_stage(MARKET_STAGE, config, store)

    instruments = store.read(MARKET_INSTRUMENTS)
    ids = [inst.instrument_id for inst in instruments]
    assert "EQ-R001" in ids
    assert "EQ-0001" in ids
    assert ids.count("CM-CRD") == 1


def _equity_instrument(instrument_id: str, name: str) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        family=Family.EQUITIES,
        kind=InstrumentKind.EQUITY,
        name=name,
        currency="USD",
        sector="sector_01",
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=1.0,
        expiry_rule=None,
    )


def test_merge_instruments_raises_when_the_same_id_has_different_facts() -> None:
    synthetic = _equity_instrument("EQ-SAME", "synthetic version")
    real = _equity_instrument("EQ-SAME", "real version")

    with pytest.raises(
        MarketCheckError, match="instrument 'EQ-SAME' differs between real and synthetic"
    ):
        _merge_instruments([synthetic], [real])


def test_prices_have_rows_for_both_the_real_and_synthetic_seed(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)
    run_stage(MARKET_STAGE, config, store)

    prices = store.read(MARKET_PRICES)
    seeds = {row.seed for row in prices}
    assert seeds == {"R1", "A"}


def test_run_stage_writes_run_metadata_with_check_for_every_seed(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)
    run_stage(MARKET_STAGE, config, store)

    metadata_path = result.data_dir / "run_metadata" / "market.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "R1" in metadata["check"]
    assert "A" in metadata["check"]
    assert metadata["raw_manifest"]["files"] > 0
    assert "window" not in metadata["raw_manifest"]


def test_synthetic_only_config_writes_no_raw_manifest(tmp_path: Path) -> None:
    demo_text = _DEMO_CONFIG.read_text(encoding="utf-8")
    synthetic_only_text = demo_text.replace("pilot_market_seeds: [R1]", "pilot_market_seeds: [A]")
    assert synthetic_only_text != demo_text
    config_path = tmp_path / "synthetic_only_config.yaml"
    config_path.write_text(synthetic_only_text, encoding="utf-8")
    config = load_config(config_path)

    store = DataStore(tmp_path / "out", config.output)
    run_stage(MARKET_STAGE, config, store)

    metadata_path = tmp_path / "out" / "run_metadata" / "market.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "raw_manifest" not in metadata


def test_parquet_override_for_prices_writes_market_prices_parquet(fake_cache) -> None:
    config = _demo_config()
    output = config.output.model_copy(update={"tables": {"market/prices": "parquet"}})
    config = config.model_copy(update={"output": output})
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)
    run_stage(MARKET_STAGE, config, store)

    assert (result.data_dir / "market" / "prices.parquet").exists()
    assert (result.data_dir / "market" / "curves.jsonl").exists()


def test_second_run_without_force_raises_stage_io_error(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)
    run_stage(MARKET_STAGE, config, store)

    with pytest.raises(StageIOError):
        run_stage(MARKET_STAGE, config, store)


def test_second_run_with_force_succeeds(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)
    run_stage(MARKET_STAGE, config, store)
    run_stage(MARKET_STAGE, config, store, force=True)

    assert (result.data_dir / "market" / "instruments.jsonl").exists()


def test_cli_runs_market_stage_and_returns_0(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    cli_result = main(["market", "--config", str(_DEMO_CONFIG), "--data-dir", str(result.data_dir)])
    assert cli_result == 0
    for spec in MARKET_TABLES:
        assert (result.data_dir / f"{spec.name}.jsonl").exists()


def test_cli_returns_1_and_writes_no_market_files_when_check_fails(
    fake_cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _demo_config()
    result = fake_cache(config)

    def _raise(*args, **kwargs):
        raise MarketCheckError("rigged failure")

    monkeypatch.setattr("pm_traitbench.market.stage.check_market", _raise)

    cli_result = main(["market", "--config", str(_DEMO_CONFIG), "--data-dir", str(result.data_dir)])
    assert cli_result == 1

    for spec in MARKET_TABLES:
        assert not (result.data_dir / f"{spec.name}.jsonl").exists()


def test_cli_returns_1_and_writes_nothing_when_a_later_seed_fails(
    tmp_path: Path, fake_cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed A's check passing must not let seed B's failure leave a partial write."""
    demo_text = _DEMO_CONFIG.read_text(encoding="utf-8")
    two_seed_text = demo_text.replace("market_seeds: [A]", "market_seeds: [A, B]")
    assert two_seed_text != demo_text
    config_path = tmp_path / "two_seed_config.yaml"
    config_path.write_text(two_seed_text, encoding="utf-8")
    config = load_config(config_path)
    result = fake_cache(config)

    def _fail_only_on_b(market, instruments, config):
        if market.seed == "B":
            raise MarketCheckError("rigged failure on seed B")
        return real_check_market(market, instruments, config)

    monkeypatch.setattr("pm_traitbench.market.stage.check_market", _fail_only_on_b)

    cli_result = main(["market", "--config", str(config_path), "--data-dir", str(result.data_dir)])

    assert cli_result == 1
    assert not (result.data_dir / "market").exists()
    assert not (result.data_dir / "run_metadata" / "market.json").exists()


def test_pipeline_stage_names_are_sample_then_market_then_engine() -> None:
    assert tuple(stage.name for stage in pipeline.STAGES) == ("sample", "market", "engine")


def test_run_stage_raises_stage_io_error_for_unfetched_real_pilot_seed(tmp_path: Path) -> None:
    config = Config()  # binding defaults pilot the population on real seed R1
    store = DataStore(tmp_path, config.output)

    with pytest.raises(StageIOError, match="run fetch-market first"):
        run_stage(MARKET_STAGE, config, store)

    assert not (tmp_path / "market").exists()
    assert not (tmp_path / "run_metadata").exists()


def test_tampered_cache_fails_naming_the_file(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)

    tampered = cache_dir(result.data_dir) / "fred" / "DGS10.csv"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "extra garbage\n", encoding="utf-8")

    with pytest.raises(StageIOError, match=re.escape(str(tampered))):
        run_stage(MARKET_STAGE, config, store)

    assert not (result.data_dir / "market").exists()
    assert not (result.data_dir / "run_metadata").exists()


def test_cache_narrower_than_the_config_needs_raises_stage_io_error(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)

    wider = config.model_copy(
        update={
            "market": config.market.model_copy(
                update={"burn_in_days": config.market.burn_in_days + 500}
            )
        }
    )

    with pytest.raises(StageIOError, match="run fetch-market again"):
        run_stage(MARKET_STAGE, wider, store)

    assert not (result.data_dir / "market").exists()


def test_cache_ending_before_the_config_needs_raises_stage_io_error(fake_cache) -> None:
    config = _demo_config()
    result = fake_cache(config)
    store = DataStore(result.data_dir, config.output)

    manifest_path = cache_dir(result.data_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["window"][1] = "2019-01-01"  # earlier than what the config needs
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StageIOError, match="run fetch-market again"):
        run_stage(MARKET_STAGE, config, store)

    assert not (result.data_dir / "market").exists()


def test_referenced_seeds_orders_pilot_first_and_dedupes() -> None:
    config = Config.model_validate(
        {"population": {"pilot_market_seeds": ["C"], "market_seeds": ["A", "B", "C"]}}
    )
    assert referenced_seeds(config) == ["C", "A", "B"]
