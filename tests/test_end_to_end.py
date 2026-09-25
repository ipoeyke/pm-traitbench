"""End-to-end tests for the CLI: the pipeline stages run through main()."""

import json
from pathlib import Path

import pytest

from pm_traitbench import pipeline
from pm_traitbench.cli import build_parser, main
from pm_traitbench.config import OutputConfig, load_config
from pm_traitbench.enums import AssetClass
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    ENGINE_TABLES,
    GATE1_TABLES,
    IDEAS,
    MARKET_TABLES,
    PERSONAS,
    RULES,
    SKELETONS,
    TRAITS,
)
from pm_traitbench.tables.store import DataStore
from tests.market.real.conftest import fake_cache  # noqa: F401

_DEMO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
_TABLES = (PERSONAS, TRAITS, RULES, DRIFT_EVENTS)


def test_sample_writes_default_files_and_run_metadata(tmp_path: Path) -> None:
    result = main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
    assert result == 0
    assert (tmp_path / "personas.jsonl").exists()
    assert (tmp_path / "traits.jsonl").exists()
    assert (tmp_path / "rules.jsonl").exists()
    assert (tmp_path / "drift_events.jsonl").exists()
    assert (tmp_path / "run_metadata" / "sample.json").exists()


def test_sample_output_reads_back_and_validates_with_eight_personas(tmp_path: Path) -> None:
    assert main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)]) == 0
    store = DataStore(tmp_path, OutputConfig())
    personas = store.read(PERSONAS)
    assert len(personas) == 8
    for spec in _TABLES:
        store.read(spec)  # validates without raising


def test_two_runs_into_separate_directories_give_byte_identical_tables(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    assert main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(dir_a)]) == 0
    assert main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(dir_b)]) == 0
    store_a = DataStore(dir_a, OutputConfig())
    store_b = DataStore(dir_b, OutputConfig())
    for spec in _TABLES:
        assert store_a.path(spec).read_bytes() == store_b.path(spec).read_bytes()


def test_rerun_without_force_returns_1(tmp_path: Path) -> None:
    assert main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)]) == 0
    result = main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
    assert result == 1


def test_rerun_with_force_returns_0(tmp_path: Path) -> None:
    assert main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)]) == 0
    result = main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path), "--force"])
    assert result == 0


def test_parquet_output_config_matches_default_format_rows(tmp_path: Path) -> None:
    default_dir = tmp_path / "default"
    parquet_dir = tmp_path / "parquet"
    default_config = tmp_path / "default_config.yaml"
    parquet_config = tmp_path / "parquet_config.yaml"
    demo_text = _DEMO_CONFIG.read_text(encoding="utf-8")
    default_config.write_text(demo_text, encoding="utf-8")
    parquet_config.write_text(demo_text + "\noutput:\n  format: parquet\n", encoding="utf-8")

    assert main(["sample", "--config", str(default_config), "--data-dir", str(default_dir)]) == 0
    assert main(["sample", "--config", str(parquet_config), "--data-dir", str(parquet_dir)]) == 0

    default_store = DataStore(default_dir, OutputConfig())
    parquet_store = DataStore(parquet_dir, OutputConfig(format="parquet"))
    for spec in _TABLES:
        assert parquet_store.path(spec).suffix == ".parquet"
        assert default_store.read(spec) == parquet_store.read(spec)


def test_help_output_lists_the_sample_subcommand() -> None:
    help_text = build_parser(pipeline.STAGES).format_help()
    assert "sample" in help_text


def test_non_mapping_config_yaml_exits_2_with_error_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("- a\n- b\n")
    result = main(["sample", "--config", str(bad_config), "--data-dir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert result == 2
    assert captured.err.startswith("error:")


def test_sample_then_market_run_through_the_cli_against_a_fetched_cache(fake_cache) -> None:
    """The demo config's pilot seed is real: `market` needs a raw cache fetched
    first, then `sample` and `market` both write their tables through the CLI.
    """
    config = load_config(_DEMO_CONFIG)
    result = fake_cache(config)
    data_dir = result.data_dir

    assert main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(data_dir)]) == 0
    assert main(["market", "--config", str(_DEMO_CONFIG), "--data-dir", str(data_dir)]) == 0

    for spec in (*_TABLES, *MARKET_TABLES):
        assert (data_dir / f"{spec.name}.jsonl").exists()
    assert (data_dir / "run_metadata" / "sample.json").exists()
    assert (data_dir / "run_metadata" / "market.json").exists()


def test_sample_then_market_then_engine_run_through_the_cli_against_a_fetched_cache(
    fake_cache,
) -> None:
    """The full pipeline, stage by stage, on the demo config's small commodities-only
    population: sample, then market against a fetched raw cache, then engine.
    """
    config = load_config(_DEMO_CONFIG)
    result = fake_cache(config)
    data_dir = result.data_dir

    assert main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(data_dir)]) == 0
    assert main(["market", "--config", str(_DEMO_CONFIG), "--data-dir", str(data_dir)]) == 0
    assert main(["engine", "--config", str(_DEMO_CONFIG), "--data-dir", str(data_dir)]) == 0

    for spec in ENGINE_TABLES:
        assert (data_dir / f"{spec.name}.jsonl").exists()
    assert (data_dir / "run_metadata" / "engine.json").exists()


def test_sample_market_engine_then_gate1_run_through_the_cli_against_a_fetched_cache(
    fake_cache,
) -> None:
    """Gate 1 pools the demo config's small commodities-only population and blocks:
    too few PMs per cell to clear the per-cell minimum, so `gate1` exits 1.
    """
    config = load_config(_DEMO_CONFIG)
    result = fake_cache(config)
    data_dir = result.data_dir
    args = ["--config", str(_DEMO_CONFIG), "--data-dir", str(data_dir)]

    assert main(["sample", *args]) == 0
    assert main(["market", *args]) == 0
    assert main(["engine", *args]) == 0
    assert main(["gate1", *args]) == 1

    for spec in GATE1_TABLES:
        assert (data_dir / f"{spec.name}.jsonl").exists()
    assert (data_dir / "run_metadata" / "gate1.json").exists()


def test_help_output_lists_the_gate1_subcommand() -> None:
    help_text = build_parser(pipeline.STAGES).format_help()
    assert "gate1" in help_text


def test_engine_force_twice_leaves_one_set_of_idea_rules(fake_cache) -> None:
    config = load_config(_DEMO_CONFIG)
    data_dir = fake_cache(config).data_dir
    args = ["--config", str(_DEMO_CONFIG), "--data-dir", str(data_dir)]
    assert main(["sample", *args]) == 0
    assert main(["market", *args]) == 0

    rules_path = data_dir / f"{RULES.name}.jsonl"
    assert main(["engine", *args]) == 0
    first = rules_path.read_bytes()
    assert main(["engine", "--force", *args]) == 0
    assert main(["engine", "--force", *args]) == 0

    assert rules_path.read_bytes() == first


_SYNTHETIC_CONFIG = """\
population:
  asset_classes: [equities, rates_credit, commodities, multi_asset]
  market_seeds: [A]
  pilot_market_seeds: [B]
  pilot_per_cell: 1
  full_per_cell: 1
market:
  universe:
    n_equities: 20
    n_sectors: 5
    n_credit_issuers: 12
"""


def test_sample_market_engine_on_synthetic_seeds_runs_every_direct_asset_class(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(_SYNTHETIC_CONFIG, encoding="utf-8")
    data_dir = tmp_path / "data"
    args = ["--config", str(config_path), "--data-dir", str(data_dir)]

    for stage in ("sample", "market", "engine"):
        assert main([stage, *args]) == 0

    store = DataStore(data_dir, load_config(config_path).output)
    personas = store.read(PERSONAS)
    asset_class_by_pm = {p.pm_id: p.mandate.asset_class for p in personas}
    classes_with_ideas = {asset_class_by_pm[idea.pm_id] for idea in store.read(IDEAS)}
    assert classes_with_ideas == {
        AssetClass.EQUITIES,
        AssetClass.RATES_CREDIT,
        AssetClass.COMMODITIES,
    }
    metadata = json.loads((data_dir / "run_metadata" / "engine.json").read_text())
    multi_asset = sorted(
        pm_id for pm_id, ac in asset_class_by_pm.items() if ac == AssetClass.MULTI_ASSET
    )
    assert multi_asset
    assert metadata["skipped"] == multi_asset


def test_help_output_lists_the_engine_subcommand() -> None:
    help_text = build_parser(pipeline.STAGES).format_help()
    assert "engine" in help_text


def test_sample_market_engine_then_plan_on_synthetic_seeds(tmp_path: Path) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(_SYNTHETIC_CONFIG, encoding="utf-8")
    data_dir = tmp_path / "data"
    args = ["--config", str(config_path), "--data-dir", str(data_dir)]

    for stage in ("sample", "market", "engine", "plan"):
        assert main([stage, *args]) == 0

    assert (data_dir / "run_metadata" / "plan.json").exists()

    store = DataStore(data_dir, load_config(config_path).output)
    personas = store.read(PERSONAS)
    asset_class_by_pm = {p.pm_id: p.mandate.asset_class for p in personas}
    direct_asset_pms = {
        pm_id for pm_id, ac in asset_class_by_pm.items() if ac != AssetClass.MULTI_ASSET
    }

    skeletons = store.read(SKELETONS)
    pms_with_skeletons = {s.pm_id for s in skeletons}
    assert direct_asset_pms <= pms_with_skeletons


def test_help_output_lists_the_plan_subcommand() -> None:
    help_text = build_parser(pipeline.STAGES).format_help()
    assert "plan" in help_text
