"""Tests for the `fetch-market` CLI subcommand."""

from pathlib import Path
from urllib.error import URLError

import pytest

import pm_traitbench.market.real.fetch as fetch_module
from pm_traitbench.cli import main


def _isolate_to_one_fred_series(monkeypatch: pytest.MonkeyPatch, series: str) -> None:
    monkeypatch.setattr(fetch_module, "fred_series", lambda: [series])
    monkeypatch.setattr(fetch_module, "yahoo_tickers", lambda: [])
    monkeypatch.setattr(fetch_module, "_nasdaq_days", lambda config: [])


def test_fetch_market_writes_manifest_and_returns_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # fetch_all's default sleeper resolves time.sleep at call time, so patching
    # the module attribute here (not an explicit sleeper) keeps this test fast.
    monkeypatch.setattr(fetch_module.time, "sleep", lambda seconds: None)
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    monkeypatch.setattr(
        fetch_module, "urlopen_bytes", lambda url: b"observation_date,DGS2\n2018-01-02,2.0\n"
    )

    result = main(["fetch-market", "--data-dir", str(tmp_path)])

    assert result == 0
    assert (tmp_path / "raw" / "market" / "manifest.json").exists()
    out = capsys.readouterr().out
    assert f"fetched 1 files into {tmp_path}" in out


def test_fetch_market_bad_url_returns_1_via_stage_io_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch_module.time, "sleep", lambda seconds: None)
    _isolate_to_one_fred_series(monkeypatch, "DGS2")

    def _raise(url: str) -> bytes:
        raise URLError("bad url")

    monkeypatch.setattr(fetch_module, "urlopen_bytes", _raise)

    result = main(["fetch-market", "--data-dir", str(tmp_path)])

    assert result == 1
    assert capsys.readouterr().err.startswith("error:")


def test_fetch_market_with_no_referenced_real_seeds_prints_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch_module.time, "sleep", lambda seconds: None)

    def _raise(url: str) -> bytes:
        raise AssertionError(f"should not be called: {url}")

    monkeypatch.setattr(fetch_module, "urlopen_bytes", _raise)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("population:\n  pilot_market_seeds: [A]\n", encoding="utf-8")

    result = main(
        [
            "fetch-market",
            "--data-dir",
            str(tmp_path / "data"),
            "--config",
            str(config_path),
        ]
    )

    assert result == 0
    out = capsys.readouterr().out
    assert "no real market seeds configured" in out
