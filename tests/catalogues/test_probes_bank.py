"""Tests for the probe question bank: the shipped file and the loader's checks on it."""

import shutil
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import yaml

from pm_traitbench.catalogues.loader import (
    PROBE_ACTION_COUNTS,
    check_probes_catalogue,
    load_catalogue,
)
from pm_traitbench.catalogues.models import ProbeBank
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import AssetClass
from pm_traitbench.errors import CatalogueError


def _check_modified(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], name: str = "probes.yaml"
) -> None:
    """Copy the shipped catalogue, mutate one YAML file, then run the probe check."""
    source = Path(str(resources.files("pm_traitbench.catalogues")))
    for path in source.glob("*.yaml"):
        shutil.copy(path, tmp_path / path.name)
    data = yaml.safe_load((tmp_path / name).read_text())
    mutate(data)
    (tmp_path / name).write_text(yaml.safe_dump(data, allow_unicode=True))
    check_probes_catalogue(load_catalogue(tmp_path))


def test_shipped_bank_passes_check() -> None:
    check_probes_catalogue(load_catalogue())


def test_action_counts_cover_every_bias() -> None:
    assert set(PROBE_ACTION_COUNTS) == set(BIAS_PARAMS)


def test_pick_prefers_asset_class_lines_over_all() -> None:
    lines = {"all": ("a", "b"), "equities": ("c", "d")}
    assert ProbeBank.pick(lines, AssetClass.EQUITIES) == ("c", "d")
    assert ProbeBank.pick(lines, AssetClass.COMMODITIES) == ("a", "b")


def test_missing_bias_key_raises(tmp_path: Path) -> None:
    with pytest.raises(CatalogueError, match="exit_deficiency"):
        _check_modified(tmp_path, lambda d: d["biases"].pop("exit_deficiency"))


@pytest.mark.parametrize("n_values", [2, 5])
def test_preference_with_off_range_value_count_raises(tmp_path: Path, n_values: int) -> None:
    def mutate(d: dict[str, Any]) -> None:
        entry = d["preferences"][0]
        entry["values"] = [f"option {i}" for i in range(n_values)]

    with pytest.raises(CatalogueError, match=r"need 3 or 4"):
        _check_modified(tmp_path, mutate, "preferences.yaml")


def test_preference_in_situ_value_slot_raises(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["preferences"]["workflow"]["in_situ"]["all"][0] = "Do it as {value} on {instrument}."

    with pytest.raises(CatalogueError, match="workflow.*in_situ.*value"):
        _check_modified(tmp_path, mutate)


def test_loss_aversion_situation_without_horizon_raises(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["biases"]["loss_aversion_lambda"]["situation"]["all"][0] = (
            "{instrument} is down at {level}. What next?"
        )

    with pytest.raises(CatalogueError, match="loss_aversion_lambda.*situation.*horizon"):
        _check_modified(tmp_path, mutate)


def test_banned_word_in_presence_raises(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["biases"]["herding_weight"]["presence"]["all"][0] = "Is this PM anchored to a view?"

    with pytest.raises(CatalogueError, match="herding_weight.*presence.*anchor"):
        _check_modified(tmp_path, mutate)


def test_wrong_action_count_raises(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["biases"]["loss_aversion_lambda"]["actions"].pop()

    with pytest.raises(CatalogueError, match="loss_aversion_lambda.*actions"):
        _check_modified(tmp_path, mutate)


def test_decline_on_a_non_decline_bias_raises(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["biases"]["herding_weight"]["decline"] = {
            "all": [
                "Put {size} of book in {instrument} at {level}.",
                "Give me {size} in {instrument}.",
            ]
        }

    with pytest.raises(CatalogueError, match="herding_weight.*decline"):
        _check_modified(tmp_path, mutate)


def test_em_dash_raises(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["routine"]["all"][0] = "Where is {instrument} — it shows {level}?"

    with pytest.raises(CatalogueError, match="routine.*em dash"):
        _check_modified(tmp_path, mutate)


def test_single_line_key_raises(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["routine"]["all"] = d["routine"]["all"][:1]

    with pytest.raises(CatalogueError, match="routine.*fewer than 2"):
        _check_modified(tmp_path, mutate)


def test_bias_param_name_in_situation_raises(tmp_path: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["biases"]["herding_weight"]["situation"]["all"][0] = (
            "The street is {street} {instrument} at {level}, the PM's read is {own_side}, "
            "and its loss_aversion_lambda is high."
        )

    with pytest.raises(CatalogueError, match="herding_weight.*names param 'loss_aversion_lambda'"):
        _check_modified(tmp_path, mutate)
