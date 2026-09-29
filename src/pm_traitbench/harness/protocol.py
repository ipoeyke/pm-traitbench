"""The contract between the harness and a system under test, and the objects it receives."""

import datetime
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from pm_traitbench.enums import ProbeForm
from pm_traitbench.tables.schema import Mandate, Rule, Turn


class PublicProfile(BaseModel):
    """What a PM's copilot is told up front: mandate, self-description and PM-scope rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str
    mandate: Mandate
    self_description: str
    rules: tuple[Rule, ...]


class PublicSession(BaseModel):
    """One session transcript with the idea-scope rules of the ideas it discusses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    date: datetime.date
    turns: tuple[Turn, ...]
    idea_rules: tuple[Rule, ...]


class PublicProbe(BaseModel):
    """A question put to the system: its text and, for a multiple-choice probe, its options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: str
    form: ProbeForm
    question: str
    options: tuple[str, ...]


class SystemUnderTest(Protocol):
    """A copilot memory system that is fed sessions and asked questions.

    One instance serves one PM. `observe` is called in date order. `answer` must
    not write memory, since a probe's premise may be deliberately stale. An
    optional `close()` method is called once after the PM finishes or fails.
    """

    def observe(self, session: PublicSession) -> None: ...

    def answer(self, as_of: datetime.date, probe: PublicProbe) -> str: ...


SutFactory = Callable[[PublicProfile], SystemUnderTest]
