"""Tests for the replay runner: chronology, isolation, resume and run-directory guards."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from pm_traitbench.enums import ProbeForm
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.protocol import PublicProbe, PublicProfile, PublicSession
from pm_traitbench.harness.runner import (
    RUN_METADATA,
    check_run_name,
    default_run_name,
    load_factory,
    run_dir,
    run_sut,
)
from pm_traitbench.harness.views import opaque_probe_id
from pm_traitbench.tables.schema import probe_id
from pm_traitbench.tables.specs import PERSONAS, PROBES, RESPONSES, RULES, SESSIONS
from pm_traitbench.tables.store import DataStore
from tests.engine.fixtures import stage_config
from tests.harness.fixtures import (
    ECHO_FACTORY,
    persona_row,
    probe_row,
    recording_factory,
    session_row,
    validated_corpus_with_probes,
)

SYNTHETIC_PMS = ("pm_001", "pm_002", "pm_003")


def _synthetic_corpus(tmp_path: Path):
    """Three PMs with two sessions and two checkpoints each, plus fresh probes metadata."""
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    store.write(PERSONAS, [persona_row(pm) for pm in SYNTHETIC_PMS])
    store.write(RULES, [])
    sessions = []
    probes = []
    for pm in SYNTHETIC_PMS:
        sessions += [
            session_row(pm, f"s_{pm.replace('_', '')}_2026-01-06_a", date(2026, 1, 6)),
            session_row(pm, f"s_{pm.replace('_', '')}_2026-01-20_a", date(2026, 1, 20)),
        ]
        probes += [probe_row(pm, 1, date(2026, 1, 13)), probe_row(pm, 2, date(2026, 1, 27))]
    store.write(SESSIONS, sessions)
    store.write(PROBES, probes)
    digest = hashlib.sha256(store.path(SESSIONS).read_bytes()).hexdigest()
    store.write_run_metadata("probes", config, {"sessions_sha256": digest})
    return config, store


def _set_probes_hash(config, store, digest: str) -> None:
    store.write_run_metadata("probes", config, {"sessions_sha256": digest})


def _run(config, store, factory, **kwargs):
    return run_sut(config, store, factory, sut_name="echo", run_name="r1", **kwargs)


def _responses(result):
    return [(r.pm_id, r.probe_id, r.response) for r in result.run_store.read(RESPONSES)]


def test_every_probe_gets_one_response(tmp_path, fixture_market, neutral_pm, monkeypatch) -> None:
    config, store = validated_corpus_with_probes(tmp_path, fixture_market, neutral_pm, monkeypatch)
    factory, _ = recording_factory()

    result = _run(config, store, factory)

    ids = sorted(r.probe_id for r in store.read(PROBES))
    assert sorted(r.probe_id for r in result.run_store.read(RESPONSES)) == ids
    assert result.failed == {}


def test_replay_invariant(tmp_path, fixture_market, neutral_pm, monkeypatch) -> None:
    config, store = validated_corpus_with_probes(tmp_path, fixture_market, neutral_pm, monkeypatch)
    factory, built = recording_factory()
    _run(config, store, factory)

    sessions = store.read(SESSIONS)
    assert built
    for sut in built.values():
        observed: list[PublicSession] = []
        answered = 0
        for event in sut.events:
            if event[0] == "observe":
                observed.append(event[1])
                continue
            answered += 1
            as_of = event[1]
            pm_id = sut.profile.pm_id
            expected = {s.session_id for s in sessions if s.pm_id == pm_id and s.date <= as_of}
            assert {s.session_id for s in observed} == expected
        assert answered
        all_sessions = [e[1] for e in sut.events if e[0] == "observe"]
        assert [(s.date, s.session_id) for s in all_sessions] == sorted(
            (s.date, s.session_id) for s in all_sessions
        )


def test_sessions_due_at_checkpoint_precede_its_first_probe(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    store.write(SESSIONS, [session_row("pm_001", "s_pm001_2026-01-13_a", date(2026, 1, 13))])
    digest = hashlib.sha256(store.path(SESSIONS).read_bytes()).hexdigest()
    _set_probes_hash(config, store, digest)
    factory, built = recording_factory()

    _run(config, store, factory)

    kinds = [e[0] for e in built["pm_001"].events]
    assert kinds == ["observe", "answer", "answer"]


def test_received_objects_carry_no_hidden_fields(
    tmp_path, fixture_market, neutral_pm, monkeypatch
) -> None:
    config, store = validated_corpus_with_probes(tmp_path, fixture_market, neutral_pm, monkeypatch)
    factory, built = recording_factory()
    _run(config, store, factory)

    hidden = {r.probe_id: r.answer for r in store.read(PROBES) if r.form == ProbeForm.OPEN}
    assert hidden
    for sut in built.values():
        assert isinstance(sut.profile, PublicProfile)
        for event in sut.events:
            obj = event[1] if event[0] == "observe" else event[2]
            assert isinstance(obj, PublicSession | PublicProbe)
            if isinstance(obj, PublicProbe) and obj.probe_id in hidden:
                assert hidden[obj.probe_id] not in obj.model_dump_json()


def test_close_called_per_pm(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    factory, built = recording_factory(
        fail_on_probe_id=opaque_probe_id(config.seed.root, probe_id("pm_002", 1))
    )

    _run(config, store, factory)

    assert len(built) == 3
    assert all(sut.closed for sut in built.values())


def _first_probe_id(config, store: DataStore, pm_id: str) -> str:
    """The opaque id of the PM's lowest probe id, as the system under test sees it."""
    real = min(r.probe_id for r in store.read(PROBES) if r.pm_id == pm_id)
    return opaque_probe_id(config.seed.root, real)


def test_failing_pm_does_not_stop_others(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    bad = _first_probe_id(config, store, "pm_002")
    factory, _ = recording_factory(fail_on_probe_id=bad)

    result = _run(config, store, factory)

    assert set(result.failed) == {"pm_002"}
    assert "RuntimeError" in result.failed["pm_002"]
    assert result.completed == ("pm_001", "pm_003")
    assert {pm for pm, _, _ in _responses(result)} == {"pm_001", "pm_003"}
    meta = result.run_store.read_run_metadata(RUN_METADATA)
    assert list(meta["pms_failed"]) == ["pm_002"]


def test_resume_skips_completed_pms(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    failing, _ = recording_factory(fail_on_probe_id=_first_probe_id(config, store, "pm_002"))
    _run(config, store, failing)
    factory, built = recording_factory()

    result = _run(config, store, factory)

    assert result.skipped == ("pm_001", "pm_003")
    assert list(built) == ["pm_002"]
    assert result.failed == {}
    assert len(_responses(result)) == len(store.read(PROBES))


def test_force_reruns_everything(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    factory, _ = recording_factory()
    _run(config, store, factory)
    again, built = recording_factory()

    result = _run(config, store, again, force=True)

    assert sorted(built) == list(SYNTHETIC_PMS)
    assert result.skipped == ()


def test_stale_probes_metadata_raises(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    _set_probes_hash(config, store, "bad")
    factory, built = recording_factory()

    with pytest.raises(HarnessError, match="sessions changed"):
        _run(config, store, factory)

    assert not (tmp_path / "eval").exists()
    assert not built


def test_missing_probes_metadata_raises(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    (tmp_path / "run_metadata" / "probes.json").unlink()

    with pytest.raises(HarnessError, match="probes stage"):
        _run(config, store, recording_factory()[0])


def test_changed_probes_without_force_raises(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    _run(config, store, recording_factory()[0])
    path = run_dir(tmp_path, "r1") / "run_metadata" / f"{RUN_METADATA}.json"
    meta = json.loads(path.read_text())
    meta["probes_sha256"] = "changed"
    path.write_text(json.dumps(meta))

    with pytest.raises(HarnessError, match="--force"):
        _run(config, store, recording_factory()[0])


def test_resume_with_a_different_sut_raises_without_force(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    _run(config, store, recording_factory()[0])

    with pytest.raises(HarnessError, match="--force"):
        run_sut(config, store, recording_factory()[0], sut_name="other", run_name="r1")

    factory, built = recording_factory()
    run_sut(config, store, factory, sut_name="other", run_name="r1", force=True)
    assert sorted(built) == list(SYNTHETIC_PMS)


def test_resume_with_changed_harness_config_raises_without_force(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    _run(config, store, recording_factory()[0])
    harness = config.harness.model_copy(
        update={"max_answer_tokens": config.harness.max_answer_tokens + 1}
    )
    changed = config.model_copy(update={"harness": harness})

    with pytest.raises(HarnessError, match="--force"):
        _run(changed, store, recording_factory()[0])

    factory, built = recording_factory()
    _run(changed, store, factory, force=True)
    assert sorted(built) == list(SYNTHETIC_PMS)


def test_workers_give_same_responses(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)

    def answer(as_of, probe):
        return "B" if probe.probe_id.endswith("2") else "A"

    one = _responses(
        run_sut(
            config,
            store,
            recording_factory(answer_fn=answer)[0],
            sut_name="echo",
            run_name="w1",
            workers=1,
        )
    )
    four = _responses(
        run_sut(
            config,
            store,
            recording_factory(answer_fn=answer)[0],
            sut_name="echo",
            run_name="w4",
            workers=4,
        )
    )

    assert one == four
    assert len(one) == len(store.read(PROBES))


def test_non_str_answer_fails_pm(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    factory, _ = recording_factory(answer_fn=lambda as_of, probe: None)

    result = _run(config, store, factory)

    assert set(result.failed) == set(SYNTHETIC_PMS)
    assert "probe p_pm001_" in result.failed["pm_001"]
    assert "NoneType" in result.failed["pm_001"]


def _ten_probe_corpus(tmp_path: Path):
    config, store = _synthetic_corpus(tmp_path)
    store.write(PERSONAS, [persona_row("pm_001")])
    store.write(SESSIONS, [session_row("pm_001", "s_pm001_2026-01-06_a", date(2026, 1, 6))])
    store.write(PROBES, [probe_row("pm_001", n, date(2026, 1, 13)) for n in range(1, 11)])
    digest = hashlib.sha256(store.path(SESSIONS).read_bytes()).hexdigest()
    _set_probes_hash(config, store, digest)
    return config, store


def test_ask_order_is_seeded_and_not_probe_id_order(tmp_path) -> None:
    config, store = _ten_probe_corpus(tmp_path)
    orders = []
    for name in ("o1", "o2"):
        factory, built = recording_factory()
        run_sut(config, store, factory, sut_name="echo", run_name=name)
        orders.append([e[2].probe_id for e in built["pm_001"].events if e[0] == "answer"])

    assert orders[0] == orders[1]
    natural = [opaque_probe_id(config.seed.root, probe_id("pm_001", n)) for n in range(1, 11)]
    assert sorted(orders[0]) == sorted(natural)
    assert orders[0] != natural

    other = config.model_copy(update={"seed": config.seed.model_copy(update={"root": 7})})
    factory, built = recording_factory()
    run_sut(other, store, factory, sut_name="echo", run_name="o3")
    natural_other = [opaque_probe_id(7, probe_id("pm_001", n)) for n in range(1, 11)]
    assert [e[2].probe_id for e in built["pm_001"].events if e[0] == "answer"] != natural_other


def test_system_sees_opaque_ids_and_responses_keep_real_ids(tmp_path) -> None:
    config, store = _ten_probe_corpus(tmp_path)
    factory, built = recording_factory()
    result = run_sut(config, store, factory, sut_name="echo", run_name="o1")

    seen = [e[2].probe_id for e in built["pm_001"].events if e[0] == "answer"]
    assert all(q.startswith("q_") and "pm001" not in q and "pm_001" not in q for q in seen)
    real = {r.probe_id for r in store.read(PROBES)}
    assert {r.probe_id for r in result.run_store.read(RESPONSES)} == real


def test_default_run_name() -> None:
    assert default_run_name("mycopilot.bench:make_adapter") == "mycopilot_bench_make_adapter"


@pytest.mark.parametrize("name", ["a/b", "a b", "", "A"])
def test_check_run_name_rejects_slash_and_space(name: str) -> None:
    with pytest.raises(HarnessError):
        check_run_name(name)


def test_check_run_name_returns_valid_name() -> None:
    assert check_run_name("run-1_a") == "run-1_a"


def test_load_factory_errors() -> None:
    for bad in ("no_colon", "no_such_module_xyz:attr", "tests.harness.fixtures:missing"):
        with pytest.raises(HarnessError):
            load_factory(bad)
    with pytest.raises(HarnessError):
        load_factory("tests.harness.fixtures:_LETTERS")


def test_load_factory_loads_callable() -> None:
    assert load_factory("tests.harness.fixtures:ECHO_FACTORY") is ECHO_FACTORY


def test_interrupted_run_with_other_probes_raises_without_force(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    _run(config, store, recording_factory()[0])
    path = run_dir(tmp_path, "r1") / "run_metadata" / f"{RUN_METADATA}.json"
    meta = json.loads(path.read_text())
    meta.update(status="running", pms_completed=[], probes_sha256="old-probes")
    path.write_text(json.dumps(meta))

    with pytest.raises(HarnessError, match="--force"):
        _run(config, store, recording_factory()[0])


def test_metadata_is_recorded_before_any_pm_runs(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    seen = []

    def factory(profile):
        meta = DataStore(run_dir(tmp_path, "r1"), config.output).read_run_metadata(RUN_METADATA)
        seen.append(meta["status"])
        return recording_factory()[0](profile)

    result = _run(config, store, factory)

    assert set(seen) == {"running"}
    assert result.run_store.read_run_metadata(RUN_METADATA)["status"] == "finished"


def test_zero_workers_raises(tmp_path) -> None:
    config, store = _synthetic_corpus(tmp_path)
    with pytest.raises(HarnessError, match="workers"):
        _run(config, store, recording_factory()[0], workers=0)
