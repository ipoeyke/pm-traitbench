import pytest

from pm_traitbench.cli import main


def test_main_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "pm-traitbench" in capsys.readouterr().out


def test_version_flag_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
