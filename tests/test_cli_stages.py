"""Tests for the CLI: subcommand dispatch, help output and error handling."""

from pathlib import Path

import pytest

from pm_traitbench.cli import build_parser, main
from pm_traitbench.config import Config
from pm_traitbench.enums import Kind
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import Trait
from pm_traitbench.tables.specs import TRAITS
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


def _fake_stages() -> tuple[Stage, ...]:
    return (
        Stage(
            number=1,
            name="fake",
            help="fake stage for tests",
            run=_write_two_traits,
            writes=(TRAITS,),
        ),
    )


def test_main_no_subcommand_prints_help_and_returns_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([], stages=_fake_stages()) == 0
    out = capsys.readouterr().out
    assert "stage 1:" in out


def test_main_runs_fake_stage_and_writes_table(tmp_path: Path) -> None:
    result = main(["fake", "--data-dir", str(tmp_path)], stages=_fake_stages())
    assert result == 0
    assert (tmp_path / "traits.csv").exists()


def test_main_second_run_without_force_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["fake", "--data-dir", str(tmp_path)], stages=_fake_stages())
    result = main(["fake", "--data-dir", str(tmp_path)], stages=_fake_stages())
    assert result == 1
    assert capsys.readouterr().err.startswith("error:")


def test_main_second_run_with_force_returns_0(tmp_path: Path) -> None:
    main(["fake", "--data-dir", str(tmp_path)], stages=_fake_stages())
    result = main(["fake", "--data-dir", str(tmp_path), "--force"], stages=_fake_stages())
    assert result == 0


def test_main_invalid_config_returns_2(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unknown_key: 1\n", encoding="utf-8")
    result = main(
        ["fake", "--data-dir", str(tmp_path / "data"), "--config", str(config_path)],
        stages=_fake_stages(),
    )
    assert result == 2


def test_main_version_flag_exits_with_0() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"], stages=_fake_stages())
    assert exc.value.code == 0


def test_build_parser_duplicate_stage_names_raises_value_error_naming_the_stage() -> None:
    stage_a = Stage(number=1, name="fake", help="a", run=_write_two_traits)
    stage_b = Stage(number=2, name="fake", help="b", run=_write_two_traits)
    with pytest.raises(ValueError, match="fake"):
        build_parser((stage_a, stage_b))


def test_build_parser_duplicate_stage_numbers_raises_value_error_naming_the_number() -> None:
    stage_a = Stage(number=1, name="fake_a", help="a", run=_write_two_traits)
    stage_b = Stage(number=1, name="fake_b", help="b", run=_write_two_traits)
    with pytest.raises(ValueError, match=r"duplicate stage number: 1$"):
        build_parser((stage_a, stage_b))


def test_stage_subcommand_help_documents_config_data_dir_and_force(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser(_fake_stages())
    with pytest.raises(SystemExit):
        parser.parse_args(["fake", "--help"])
    help_text = capsys.readouterr().out
    assert "YAML file overriding default settings" in help_text
    assert "directory for pipeline tables (default: data)" in help_text
    assert "overwrite existing output tables" in help_text


def test_main_with_real_pipeline_stages_returns_0() -> None:
    assert main([]) == 0
