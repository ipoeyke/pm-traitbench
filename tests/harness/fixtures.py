"""Shared harness test fixtures: small keyword-overridable corpus row builders."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

from pm_traitbench.config import Config
from pm_traitbench.enums import (
    Action,
    AssetClass,
    Op,
    OptionSource,
    ProbeForm,
    ProbeType,
    RuleScope,
    RuleSource,
    Split,
    Typicality,
)
from pm_traitbench.harness.protocol import PublicProbe, PublicProfile, PublicSession, SutFactory
from pm_traitbench.probes.stage import PROBES_STAGE
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.schema import (
    Mandate,
    Persona,
    ProbeRow,
    Rule,
    Session,
    StatedProfile,
    probe_id,
)
from pm_traitbench.tables.store import DataStore
from tests.gates.gate2.fixtures import session_of
from tests.probes.test_stage import _run_validated_corpus

_LETTERS = "abcd"


def persona_row(pm_id: str = "pm_001", **overrides) -> Persona:
    """An equities persona for `pm_id`, overridable by keyword."""
    fields = dict(
        pm_id=pm_id,
        market_seed="P",
        split=Split.PILOT,
        mandate=Mandate(
            asset_class=AssetClass.EQUITIES,
            sub_style="equity_long_short",
            book_size=1e8,
            risk_unit="pct_nav",
            benchmark="cash",
        ),
        stated_profile=StatedProfile(self_description="I run a disciplined, rules-based book."),
        typicality=Typicality.TYPICAL,
    )
    fields.update(overrides)
    return Persona(**fields)


def rule_row(
    pm_id: str,
    rule_id: str,
    scope: RuleScope,
    trade_idea_id: str | None = None,
    text: str = "Exit a position after a 5 percent drawdown from entry.",
) -> Rule:
    """A stop-loss rule of the given scope."""
    return Rule(
        pm_id=pm_id,
        rule_id=rule_id,
        source=RuleSource.SELF,
        scope=scope,
        trade_idea_id=trade_idea_id,
        param="stop_loss",
        field="pnl_pct",
        op=Op.LE,
        level=-5.0,
        unit="pct",
        window=1,
        action=Action.EXIT,
        text=text,
    )


def session_row(
    pm_id: str,
    session_id: str,
    day: date,
    trade_idea_ids: tuple[str, ...] = (),
    turns=None,
) -> Session:
    """A session on `day` with the given id; `turns` defaults to one PM and advisor exchange."""
    base = session_of(pm_id, day, ["hello"], trade_idea_ids=trade_idea_ids)
    return base.model_copy(
        update={"session_id": session_id, "turns": base.turns if turns is None else turns}
    )


def probe_row(
    pm_id: str,
    n: int,
    day: date,
    form: ProbeForm = ProbeForm.MCQ,
    probe_type: ProbeType = ProbeType.TRAIT_MCQ,
    options: tuple[str, ...] = ("x", "y", "z"),
    answer: str = "A",
    trait_id: str | None = "t_01",
) -> ProbeRow:
    """A probe with `options` in A-D order; the first is the current option, the rest pre-update."""
    padded = [*options, *([None] * (4 - len(options)))]
    sources = [
        None if text is None else (OptionSource.CURRENT if i == 0 else OptionSource.PRE_UPDATE)
        for i, text in enumerate(padded)
    ]
    fields = {}
    for i, letter in enumerate(_LETTERS):
        fields[f"option_{letter}"] = padded[i]
        fields[f"source_{letter}"] = sources[i]
    return ProbeRow(
        probe_id=probe_id(pm_id, n),
        pm_id=pm_id,
        checkpoint_date=day,
        checkpoint_label="week4",
        probe_type=probe_type,
        trait_id=trait_id,
        form=form,
        question="Which do you prefer?",
        answer=answer,
        supporting_signal_ids=(),
        context_chars=0,
        **fields,
    )


class RecordingSut:
    """A system under test that logs every call and answers from a scripted callable."""

    def __init__(
        self,
        profile: PublicProfile,
        answer_fn: Callable[[date, PublicProbe], str] | None = None,
        fail_on: Callable[[str], bool] | None = None,
    ) -> None:
        self.profile = profile
        self.events: list[tuple] = []
        self.closed = False
        self._answer_fn = answer_fn or (lambda as_of, probe: "A")
        self._fail_on = fail_on

    def observe(self, session: PublicSession) -> None:
        self.events.append(("observe", session))

    def answer(self, as_of: date, probe: PublicProbe) -> str:
        self.events.append(("answer", as_of, probe))
        if self._fail_on is not None and self._fail_on(probe.probe_id):
            raise RuntimeError(f"scripted failure on {probe.probe_id}")
        return self._answer_fn(as_of, probe)

    def close(self) -> None:
        self.closed = True


def recording_factory(**kwargs) -> tuple[SutFactory, dict[str, RecordingSut]]:
    """A factory of `RecordingSut`s built with `kwargs`, plus the instances keyed by pm_id."""
    built: dict[str, RecordingSut] = {}

    def factory(profile: PublicProfile) -> RecordingSut:
        sut = RecordingSut(profile, **kwargs)
        built[profile.pm_id] = sut
        return sut

    return factory, built


def _raise_in_answer(as_of: date, probe: PublicProbe) -> str:
    raise RuntimeError("scripted failure")


def _echo_factory(profile: PublicProfile) -> RecordingSut:
    return RecordingSut(profile)


def _failing_factory(profile: PublicProfile) -> RecordingSut:
    return RecordingSut(profile, answer_fn=_raise_in_answer)


ECHO_FACTORY: SutFactory = _echo_factory
FAILING_FACTORY: SutFactory = _failing_factory


def validated_corpus_with_probes(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch
) -> tuple[Config, DataStore]:
    """A validated dialogue corpus with the probes stage run on top."""
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    run_stage(PROBES_STAGE, config, store)
    return config, store
