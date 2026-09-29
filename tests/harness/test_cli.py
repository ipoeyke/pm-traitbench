"""Tests for the `eval run` and `eval score` commands and their separation from the stages."""

import json
import re
from pathlib import Path

import pytest

from pm_traitbench import cli, pipeline
from pm_traitbench.cli import build_parser, main
from pm_traitbench.stages import Stage
from tests.dialogue.fixtures import FakeClient, fake_message
from tests.harness.fixtures import validated_corpus_with_probes

ECHO = "tests.harness.fixtures:ECHO_FACTORY"
FAILING = "tests.harness.fixtures:FAILING_FACTORY"


@pytest.fixture
def corpus(tmp_path, fixture_market, neutral_pm, monkeypatch) -> Path:
    """A probes corpus on disk; the CLI loads the fixture config instead of the default."""
    config, store = validated_corpus_with_probes(tmp_path, fixture_market, neutral_pm, monkeypatch)
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    return store.data_dir


def test_eval_run_then_score(corpus: Path, capsys) -> None:
    args = ["--data-dir", str(corpus)]
    assert main(["eval", "run", "--sut", ECHO, "--run-name", "echo", *args]) == 0
    assert "completed" in capsys.readouterr().out

    assert main(["eval", "score", "--run-name", "echo", *args]) == 0
    assert "trait_presence" in capsys.readouterr().out
    assert (corpus / "eval" / "echo" / "summary.json").exists()


def test_eval_run_exit_1_on_failed_pm(corpus: Path, capsys) -> None:
    code = main(["eval", "run", "--sut", FAILING, "--run-name", "bad", "--data-dir", str(corpus)])

    assert code == 1
    assert "pm_" in capsys.readouterr().err


def test_eval_run_bad_sut_exit_1(corpus: Path, capsys) -> None:
    code = main(["eval", "run", "--sut", "nosuch.module:x", "--data-dir", str(corpus)])

    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_eval_run_default_run_name(corpus: Path) -> None:
    assert main(["eval", "run", "--sut", ECHO, "--data-dir", str(corpus)]) == 0

    assert (corpus / "eval" / "tests_harness_fixtures_echo_factory").is_dir()


def test_eval_run_rejects_zero_workers(corpus: Path) -> None:
    with pytest.raises(SystemExit):
        main(["eval", "run", "--sut", ECHO, "--workers", "0", "--data-dir", str(corpus)])


def test_baseline_name_resolves(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reply = fake_message([{"type": "text", "text": json.dumps({"answer": "A"})}])
    monkeypatch.setattr(
        "pm_traitbench.harness.baselines.AnthropicClient",
        lambda *args, **kwargs: FakeClient(lambda request: reply),
    )

    assert main(["eval", "run", "--sut", "no-memory", "--data-dir", str(corpus)]) == 0


def test_stage_named_eval_rejected() -> None:
    stub = Stage(number=99, name="eval", help="stub", run=lambda config, store: None)

    with pytest.raises(ValueError, match="eval"):
        build_parser([stub])


def test_harness_not_a_pipeline_stage() -> None:
    assert "eval" not in {stage.name for stage in pipeline.STAGES}
    script = (Path(__file__).parents[2] / "scripts" / "generate.sh").read_text(encoding="utf-8")
    stages_line = next(line for line in script.splitlines() if line.startswith("stages=("))
    assert "eval" not in re.findall(r"[\w-]+", stages_line)[1:]
