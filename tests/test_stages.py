"""Tests for the stage runner: read/write preconditions and postconditions."""

import json
from pathlib import Path
from typing import Any

import pytest

from pm_traitbench.config import Config, OutputConfig
from pm_traitbench.enums import Action, Kind, Op, RuleScope, RuleSource
from pm_traitbench.errors import StageIOError
from pm_traitbench.stages import Append, Stage, run_stage
from pm_traitbench.tables.schema import Rule, Trait, to_record
from pm_traitbench.tables.specs import PERSONAS, RULES, TRAITS
from pm_traitbench.tables.store import DataStore


def _rule(
    pm_id: str,
    rule_id: str,
    *,
    scope: RuleScope = RuleScope.PM,
    trade_idea_id: str | None = None,
    text: str = "Trim position when it exceeds 10% of book.",
) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id=rule_id,
        source=RuleSource.MANDATE,
        scope=scope,
        trade_idea_id=trade_idea_id,
        param="max_position_pct",
        field="position_pct",
        op=Op.GT,
        level=10.0,
        unit="pct",
        window=1,
        action=Action.TRIM_HALF,
        text=text,
    )


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


def test_run_stage_happy_path_writes_table_and_metadata(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    stage = Stage(number=1, name="fake", help="fake stage", run=_write_two_traits, writes=(TRAITS,))

    run_stage(stage, config, store)

    assert store.exists(TRAITS)
    assert len(store.read(TRAITS)) == 2
    metadata_path = tmp_path / "run_metadata" / "fake.json"
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text())["stage"] == "fake"


def test_run_stage_missing_read_raises_and_never_calls_run(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    called = False

    def _run(config: Config, store: DataStore) -> None:
        nonlocal called
        called = True

    stage = Stage(number=1, name="fake", help="fake stage", run=_run, reads=(PERSONAS,))

    with pytest.raises(StageIOError, match="personas"):
        run_stage(stage, config, store)
    assert called is False


def test_run_stage_existing_output_without_force_raises_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write(TRAITS, [_trait("pm_000", "t_00")])
    original_bytes = (tmp_path / "traits.jsonl").read_bytes()
    stage = Stage(number=1, name="fake", help="fake stage", run=_write_two_traits, writes=(TRAITS,))

    with pytest.raises(StageIOError, match="--force"):
        run_stage(stage, config, store)

    assert (tmp_path / "traits.jsonl").read_bytes() == original_bytes


def test_run_stage_with_force_overwrites_existing_output(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write(TRAITS, [_trait("pm_000", "t_00")])
    stage = Stage(number=1, name="fake", help="fake stage", run=_write_two_traits, writes=(TRAITS,))

    run_stage(stage, config, store, force=True)

    assert len(store.read(TRAITS)) == 2


def test_run_stage_writes_nothing_raises_after_run(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    called = False

    def _run(config: Config, store: DataStore) -> None:
        nonlocal called
        called = True

    stage = Stage(number=1, name="fake", help="fake stage", run=_run, writes=(TRAITS,))

    with pytest.raises(StageIOError, match="fake"):
        run_stage(stage, config, store)
    assert called is True
    assert not (tmp_path / "run_metadata" / "fake.json").exists()


def test_run_stage_raising_leaves_no_metadata(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()

    def _run(config: Config, store: DataStore) -> None:
        raise RuntimeError("boom")

    stage = Stage(number=1, name="fake", help="fake stage", run=_run, writes=(TRAITS,))

    with pytest.raises(RuntimeError, match="boom"):
        run_stage(stage, config, store)
    assert not (tmp_path / "run_metadata" / "fake.json").exists()
    assert not store.exists(TRAITS)


def test_run_stage_passes_run_return_value_as_metadata_extra(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()

    def _run(config: Config, store: DataStore) -> dict:
        store.write(TRAITS, [_trait("pm_001", "t_01")])
        return {"check": {"ok": True}}

    stage = Stage(number=1, name="fake", help="fake stage", run=_run, writes=(TRAITS,))

    run_stage(stage, config, store)

    metadata = json.loads((tmp_path / "run_metadata" / "fake.json").read_text())
    assert metadata["check"] == {"ok": True}


def test_run_stage_with_none_return_keeps_metadata_shape(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    stage = Stage(number=1, name="fake", help="fake stage", run=_write_two_traits, writes=(TRAITS,))

    run_stage(stage, config, store)

    metadata = json.loads((tmp_path / "run_metadata" / "fake.json").read_text())
    assert set(metadata.keys()) == {
        "stage",
        "created_at",
        "package_version",
        "git_commit",
        "root_seed",
        "config",
    }


def test_run_stage_with_force_rejects_a_stage_that_skips_a_table_and_keeps_the_old_file(
    tmp_path: Path,
) -> None:
    config = Config()
    store = DataStore(tmp_path, config.output)
    writer = Stage(number=1, name="fake", help="h", run=_write_two_traits, writes=(TRAITS,))
    run_stage(writer, config, store)
    before = store.path(TRAITS).read_bytes()
    metadata = tmp_path / "run_metadata" / "fake.json"
    metadata_before = metadata.read_bytes()

    def _writes_nothing(config: Config, store: DataStore) -> None:
        return None

    skipper = Stage(number=1, name="fake", help="h", run=_writes_nothing, writes=(TRAITS,))
    with pytest.raises(StageIOError, match="did not write"):
        run_stage(skipper, config, DataStore(tmp_path, config.output), force=True)

    assert store.path(TRAITS).read_bytes() == before
    assert metadata.read_bytes() == metadata_before


def _owns_idea_rules(record: dict) -> bool:
    return record["scope"] == RuleScope.IDEA


_RULES_APPEND = Append(RULES, owned=_owns_idea_rules)


def _append_idea_rule(config: Config, store: DataStore) -> None:
    rows = store.read(RULES)
    idea_rule = _rule("pm_001", "r_02", scope=RuleScope.IDEA, trade_idea_id="ti_001")
    store.write(RULES, [*rows, idea_rule])


def test_run_stage_appends_existing_table_without_force_and_keeps_original_rows(
    tmp_path: Path,
) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write(RULES, [_rule("pm_001", "r_01")])
    stage = Stage(number=1, name="fake", help="h", run=_append_idea_rule, appends=(_RULES_APPEND,))

    # no force: an appends table pre-existing is expected, not an overwrite to guard against
    run_stage(stage, config, store)

    rows = [to_record(row) for row in store.read(RULES)]
    assert rows == [
        to_record(_rule("pm_001", "r_01")),
        to_record(_rule("pm_001", "r_02", scope=RuleScope.IDEA, trade_idea_id="ti_001")),
    ]


def _replace_owned_rows(config: Config, store: DataStore) -> None:
    kept = [row for row in store.read(RULES) if row.scope == RuleScope.PM]
    fresh = _rule("pm_001", "r_03", scope=RuleScope.IDEA, trade_idea_id="ti_001")
    store.write(RULES, [*kept, fresh])


def test_run_stage_append_may_replace_rows_it_owns(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    stale = _rule("pm_001", "r_02", scope=RuleScope.IDEA, trade_idea_id="ti_001")
    store.write(RULES, [_rule("pm_001", "r_01"), stale])
    stage = Stage(
        number=1, name="fake", help="h", run=_replace_owned_rows, appends=(_RULES_APPEND,)
    )

    run_stage(stage, config, store)

    assert [row.rule_id for row in store.read(RULES)] == ["r_01", "r_03"]


def test_run_stage_missing_append_raises_and_never_calls_run(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    called = False

    def _run(config: Config, store: DataStore) -> None:
        nonlocal called
        called = True

    stage = Stage(number=1, name="fake", help="h", run=_run, appends=(_RULES_APPEND,))

    with pytest.raises(StageIOError, match="rules"):
        run_stage(stage, config, store)
    assert called is False


def _drop_original_rule(config: Config, store: DataStore) -> None:
    store.read(RULES)
    store.write(RULES, [_rule("pm_001", "r_02", scope=RuleScope.IDEA, trade_idea_id="ti_001")])


def test_run_stage_append_dropping_original_row_raises(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write(RULES, [_rule("pm_001", "r_01")])
    stage = Stage(
        number=1, name="fake", help="h", run=_drop_original_rule, appends=(_RULES_APPEND,)
    )

    with pytest.raises(StageIOError, match="rules; the table\\(s\\) have been rewritten"):
        run_stage(stage, config, store)


def _change_original_rule_text(config: Config, store: DataStore) -> None:
    store.read(RULES)
    changed = _rule("pm_001", "r_01", text="Changed rule text.")
    store.write(
        RULES, [changed, _rule("pm_001", "r_02", scope=RuleScope.IDEA, trade_idea_id="ti_001")]
    )


def test_run_stage_append_changing_original_row_text_raises(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write(RULES, [_rule("pm_001", "r_01")])
    stage = Stage(
        number=1, name="fake", help="h", run=_change_original_rule_text, appends=(_RULES_APPEND,)
    )

    with pytest.raises(StageIOError, match="altered"):
        run_stage(stage, config, store)


def test_run_stage_append_not_written_raises_did_not_write(tmp_path: Path) -> None:
    config = Config()
    DataStore(tmp_path, config.output).write(RULES, [_rule("pm_001", "r_01")])

    def _run(config: Config, store: DataStore) -> None:
        store.read(RULES)

    stage = Stage(number=1, name="fake", help="h", run=_run, appends=(_RULES_APPEND,))

    with pytest.raises(StageIOError, match="did not write"):
        run_stage(stage, config, DataStore(tmp_path, config.output))


def test_run_stage_rewrites_existing_table_without_force_and_drops_every_old_row(
    tmp_path: Path,
) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write(RULES, [_rule("pm_001", "r_01")])
    stage = Stage(number=1, name="fake", help="h", run=_drop_original_rule, rewrites=(RULES,))

    # no force, and no kept-row check: the whole table belongs to this stage
    run_stage(stage, config, store)

    assert [row.rule_id for row in store.read(RULES)] == ["r_02"]


def test_run_stage_missing_rewrite_raises_and_never_calls_run(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    called = False

    def _run(config: Config, store: DataStore) -> None:
        nonlocal called
        called = True

    stage = Stage(number=1, name="fake", help="h", run=_run, rewrites=(RULES,))

    with pytest.raises(StageIOError, match="missing required table\\(s\\): rules"):
        run_stage(stage, Config(), store)
    assert called is False


def test_run_stage_rewrite_not_written_raises_did_not_write(tmp_path: Path) -> None:
    config = Config()
    DataStore(tmp_path, config.output).write(RULES, [_rule("pm_001", "r_01")])

    def _run(config: Config, store: DataStore) -> None:
        store.read(RULES)

    stage = Stage(number=1, name="fake", help="h", run=_run, rewrites=(RULES,))

    with pytest.raises(StageIOError, match="did not write expected table\\(s\\): rules"):
        run_stage(stage, config, DataStore(tmp_path, config.output))


def test_run_stage_calls_verdict_after_metadata_is_written(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    received: dict[str, Any] = {}

    def _run(config: Config, store: DataStore) -> dict[str, Any]:
        store.write(TRAITS, [_trait("pm_001", "t_01")])
        return {"check": {"ok": True}}

    def _verdict(extra: dict[str, Any]) -> None:
        assert (tmp_path / "run_metadata" / "fake.json").exists()
        received.update(extra)

    stage = Stage(number=1, name="fake", help="h", run=_run, writes=(TRAITS,), verdict=_verdict)

    run_stage(stage, config, store)

    assert received == {"check": {"ok": True}}


def test_run_stage_verdict_raising_leaves_tables_and_metadata(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()

    def _verdict(extra: dict[str, Any]) -> None:
        raise StageIOError("verdict failed")

    stage = Stage(
        number=1, name="fake", help="h", run=_write_two_traits, writes=(TRAITS,), verdict=_verdict
    )

    with pytest.raises(StageIOError, match="verdict failed"):
        run_stage(stage, config, store)

    assert store.exists(TRAITS)
    assert (tmp_path / "run_metadata" / "fake.json").exists()


def test_run_stage_without_verdict_is_unchanged(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    stage = Stage(number=1, name="fake", help="h", run=_write_two_traits, writes=(TRAITS,))

    run_stage(stage, config, store)

    assert store.exists(TRAITS)
    assert (tmp_path / "run_metadata" / "fake.json").exists()


def test_run_stage_verdict_receives_empty_dict_when_run_returns_none(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    received: dict[str, Any] = {"unset": True}

    def _verdict(extra: dict[str, Any]) -> None:
        received.clear()
        received.update(extra)

    stage = Stage(
        number=1, name="fake", help="h", run=_write_two_traits, writes=(TRAITS,), verdict=_verdict
    )

    run_stage(stage, config, store)

    assert received == {}
