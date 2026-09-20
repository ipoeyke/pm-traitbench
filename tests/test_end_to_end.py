"""End-to-end tests for the CLI: the sample stage run through main()."""

from pathlib import Path

import pytest

from pm_traitbench import pipeline
from pm_traitbench.cli import build_parser, main
from pm_traitbench.config import OutputConfig
from pm_traitbench.tables.specs import DRIFT_EVENTS, PERSONAS, RULES, TRAITS
from pm_traitbench.tables.store import DataStore

_DEMO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
_TABLES = (PERSONAS, TRAITS, RULES, DRIFT_EVENTS)


def test_sample_writes_default_files_and_run_metadata(tmp_path: Path) -> None:
    result = main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
    assert result == 0
    assert (tmp_path / "personas.jsonl").exists()
    assert (tmp_path / "traits.csv").exists()
    assert (tmp_path / "rules.csv").exists()
    assert (tmp_path / "drift_events.csv").exists()
    assert (tmp_path / "run_metadata" / "sample.json").exists()


def test_sample_output_reads_back_and_validates_with_twelve_personas(tmp_path: Path) -> None:
    main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
    store = DataStore(tmp_path, OutputConfig())
    personas = store.read(PERSONAS)
    assert len(personas) == 12
    for spec in _TABLES:
        store.read(spec)  # validates without raising


def test_two_runs_into_separate_directories_give_byte_identical_tables(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(dir_a)])
    main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(dir_b)])
    store_a = DataStore(dir_a, OutputConfig())
    store_b = DataStore(dir_b, OutputConfig())
    for spec in _TABLES:
        assert store_a.path(spec).read_bytes() == store_b.path(spec).read_bytes()


def test_rerun_without_force_returns_1(tmp_path: Path) -> None:
    main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
    result = main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
    assert result == 1


def test_rerun_with_force_returns_0(tmp_path: Path) -> None:
    main(["sample", "--config", str(_DEMO_CONFIG), "--data-dir", str(tmp_path)])
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
