"""Tests for the `eval run` and `eval score` commands and their separation from the stages."""

import csv
import json
import re
from pathlib import Path

import pytest
import yaml

from pm_traitbench import pipeline
from pm_traitbench.cli import build_parser, main
from pm_traitbench.config import load_config
from pm_traitbench.harness.judge import load_judge_inputs, select_items
from pm_traitbench.harness.runner import run_dir
from pm_traitbench.harness.sample import SAMPLE_COLUMNS
from pm_traitbench.stages import Stage
from pm_traitbench.tables.specs import JUDGEMENTS, PROBES
from pm_traitbench.tables.store import DataStore
from tests.dialogue.fixtures import FakeClient, fake_message
from tests.engine.fixtures import stage_config, stage_config_overrides
from tests.harness.fixtures import validated_corpus_with_probes
from tests.harness.judge_fixtures import (
    GOVERNANCE_ANSWERS,
    IN_SITU_ANSWERS,
    governance_row,
    in_situ_row,
    responder_for,
)

ECHO = "tests.harness.fixtures:ECHO_FACTORY"
FAILING = "tests.harness.fixtures:FAILING_FACTORY"
CONFIG_NAME = "config.yaml"
_R1_REGIMES = [
    ["range", "2018-06-04"],
    ["risk_off", "2018-07-02"],
    ["risk_on", "2018-08-06"],
]


@pytest.fixture
def corpus(tmp_path, fixture_market, neutral_pm, monkeypatch) -> Path:
    """A probes corpus on disk, with the config it was built with written beside its tables."""
    config, store = validated_corpus_with_probes(tmp_path, fixture_market, neutral_pm, monkeypatch)
    assert config == stage_config()
    # YAML overrides deep-merge onto the defaults, so the default real seed is refit to 12 weeks.
    overrides = stage_config_overrides()
    overrides["market"]["real"] = {"seeds": {"R1": {"regime_starts": _R1_REGIMES}}}
    path = tmp_path / CONFIG_NAME
    path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.calendar == config.calendar and loaded.population == config.population
    return store.data_dir


def _eval(corpus: Path, *argv: str) -> int:
    """Run `pm-traitbench eval` against `corpus` under its fixture config."""
    config = corpus / CONFIG_NAME
    return main(["eval", *argv, "--config", str(config), "--data-dir", str(corpus)])


def test_eval_run_then_score(corpus: Path, capsys) -> None:
    assert _eval(corpus, "run", "--sut", ECHO, "--run-name", "echo") == 0
    assert "completed" in capsys.readouterr().out

    assert _eval(corpus, "score", "--run-name", "echo") == 0
    assert "trait_presence" in capsys.readouterr().out
    assert (corpus / "eval" / "echo" / "summary.json").exists()


def test_eval_run_exit_1_on_failed_pm(corpus: Path, capsys) -> None:
    code = _eval(corpus, "run", "--sut", FAILING, "--run-name", "bad")

    assert code == 1
    err = capsys.readouterr().err
    assert "pm_" in err
    assert "scripted failure" in err


def test_eval_run_bad_sut_exit_1(corpus: Path, capsys) -> None:
    assert _eval(corpus, "run", "--sut", "nosuch.module:x") == 1
    assert "error:" in capsys.readouterr().err


def test_eval_run_default_run_name(corpus: Path) -> None:
    assert _eval(corpus, "run", "--sut", ECHO) == 0

    assert (corpus / "eval" / "tests_harness_fixtures_echo_factory").is_dir()


def test_eval_run_rejects_zero_workers(corpus: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        _eval(corpus, "run", "--sut", ECHO, "--workers", "0")

    assert "must be at least 1" in capsys.readouterr().err


def test_baseline_name_resolves(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reply = fake_message([{"type": "text", "text": json.dumps({"answer": "A"})}])
    monkeypatch.setattr(
        "pm_traitbench.harness.baselines.AnthropicClient",
        lambda *args, **kwargs: FakeClient(lambda request: reply),
    )

    assert _eval(corpus, "run", "--sut", "no-memory") == 0


def _inject_judge_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pm_traitbench.harness.judge.AnthropicClient",
        lambda *args, **kwargs: FakeClient(responder_for()),
    )


def _add_open_cases(corpus: Path) -> None:
    """Append in-situ and governance probes; the neutral fixture PM has nothing else to judge."""
    store = DataStore(corpus, load_config(corpus / CONFIG_NAME).output)
    probes = store.read(PROBES)
    pm_id, day = probes[0].pm_id, probes[0].checkpoint_date
    extra = [
        *(in_situ_row(900 + i, case, pm_id, day) for i, case in enumerate(IN_SITU_ANSWERS)),
        *(governance_row(910 + i, kind, pm_id, day) for i, kind in enumerate(GOVERNANCE_ANSWERS)),
    ]
    store.write(PROBES, [*probes, *extra])


def test_eval_judge_sample_score_end_to_end(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _inject_judge_client(monkeypatch)
    _add_open_cases(corpus)
    assert _eval(corpus, "run", "--sut", ECHO, "--run-name", "echo") == 0

    assert _eval(corpus, "judge", "--run-name", "echo") == 0
    assert "judged" in capsys.readouterr().out
    config = load_config(corpus / CONFIG_NAME)
    store = DataStore(corpus, config.output)
    run_store = DataStore(run_dir(corpus, "echo"), config.output)
    expected = len(select_items(load_judge_inputs(store, run_store)).items)
    assert expected > 0
    assert len(run_store.read(JUDGEMENTS)) == expected

    assert _eval(corpus, "sample", "--run-name", "echo", "--size", "5") == 0
    sample = run_dir(corpus, "echo") / "human_sample.csv"
    with sample.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert 0 < len(rows) <= 5
    assert tuple(rows[0]) == SAMPLE_COLUMNS
    for row in rows:
        row["human_correct"] = "yes"
    with sample.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    capsys.readouterr()

    assert _eval(corpus, "score", "--run-name", "echo") == 0
    assert "agreement judge_" in capsys.readouterr().out
    summary = json.loads((run_dir(corpus, "echo") / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["agreement"]) == {row["judge"] for row in rows}
    for key, value in summary["awaiting_judge"].items():
        if key != "routine_question/intrusion":
            assert value == 0


def test_eval_sample_before_judge_exit_1(corpus: Path, capsys) -> None:
    assert _eval(corpus, "run", "--sut", ECHO, "--run-name", "echo") == 0
    capsys.readouterr()

    assert _eval(corpus, "sample", "--run-name", "echo") == 1
    assert "eval judge" in capsys.readouterr().err


def test_eval_judge_bad_run_name_exit_1(corpus: Path, capsys) -> None:
    assert _eval(corpus, "judge", "--run-name", "Bad Name") == 1
    assert "error:" in capsys.readouterr().err


def test_eval_sample_rejects_zero_size(corpus: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        _eval(corpus, "sample", "--run-name", "echo", "--size", "0")

    assert "must be at least 1" in capsys.readouterr().err


def test_stage_named_eval_rejected() -> None:
    stub = Stage(number=99, name="eval", help="stub", run=lambda config, store: None)

    with pytest.raises(ValueError, match="eval"):
        build_parser([stub])


def test_harness_not_a_pipeline_stage() -> None:
    assert "eval" not in {stage.name for stage in pipeline.STAGES}
    script = (Path(__file__).parents[2] / "scripts" / "generate.sh").read_text(encoding="utf-8")
    stages_line = next(line for line in script.splitlines() if line.startswith("stages=("))
    assert "eval" not in re.findall(r"[\w-]+", stages_line)[1:]
