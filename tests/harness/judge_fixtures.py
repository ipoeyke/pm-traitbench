"""Shared judge test fixtures: probe rows, traits and canned verdict replies."""

import json
from datetime import date

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import ProbeBank
from pm_traitbench.enums import DriftEventType, Judge, Kind, ProbeForm, ProbeType
from pm_traitbench.tables.schema import DriftEvent, ProbeRow, Trait
from tests.dialogue.fixtures import fake_message
from tests.harness.fixtures import probe_row
from tests.probes.fixtures import drift, trait

_LETTERS = "ABCD"

IN_SITU_ANSWERS = {
    "comply": "comply: honour short bullets",
    "counteract": (
        "counteract: the advice accounts for the PM's tendency to add to losing positions "
        "rather than cut them and names it"
    ),
    "decline": (
        'decline: the request breaches r_01 "max risk 5% of book"; refuse and give the reason'
    ),
}
GOVERNANCE_ANSWERS = {
    "preference": "premise rejected: changed on 2026-02-02; current: short bullets",
    "update": "premise rejected: changed on 2026-02-02; current: much less than before",
    "dormant": "premise rejected: dormant since 2026-02-02",
}


def open_pair(
    pm_id: str,
    n: int,
    day: date,
    options: tuple[str, ...] = ("sell now", "hold", "add"),
    answer: str = "B",
) -> tuple[ProbeRow, ProbeRow]:
    """A trait MCQ and its open twin (ids n and n + 1) with the same question."""
    # probe_row only validates the current option as the answer, so set the real one by copy.
    mcq = probe_row(pm_id, n, day, options=options).model_copy(update={"answer": answer})
    twin = probe_row(
        pm_id, n + 1, day, form=ProbeForm.OPEN, options=(), answer=options[_LETTERS.index(answer)]
    )
    return mcq, twin.model_copy(update={"question": mcq.question})


def _open_row(
    pm_id: str,
    n: int,
    day: date,
    probe_type: ProbeType,
    answer: str,
    trait_id: str | None = "t_01",
) -> ProbeRow:
    return probe_row(
        pm_id,
        n,
        day,
        form=ProbeForm.OPEN,
        probe_type=probe_type,
        options=(),
        answer=answer,
        trait_id=trait_id,
    )


def in_situ_row(n: int, case: str, pm_id: str = "pm_001", day: date = date(2026, 2, 2)) -> ProbeRow:
    """An in-situ probe whose rubric is the answer text of `case`."""
    return _open_row(pm_id, n, day, ProbeType.IN_SITU, IN_SITU_ANSWERS[case])


def governance_row(
    n: int, kind: str, pm_id: str = "pm_001", day: date = date(2026, 2, 2)
) -> ProbeRow:
    """A governance probe whose rubric is the answer text of `kind`."""
    return _open_row(pm_id, n, day, ProbeType.GOVERNANCE, GOVERNANCE_ANSWERS[kind])


def routine_row(
    n: int,
    held: tuple[tuple[str, str], ...],
    pm_id: str = "pm_001",
    day: date = date(2026, 2, 2),
) -> ProbeRow:
    """A routine question holding the given (param, value) preferences."""
    body = "; ".join(f"{p}={v}" for p, v in held) or "none"
    answer = f"format: {body}; intrusion: none"
    return _open_row(pm_id, n, day, ProbeType.ROUTINE_QUESTION, answer, trait_id=None)


def presence_row(n: int, pm_id: str = "pm_001", day: date = date(2026, 2, 2)) -> ProbeRow:
    """A presence probe, which no judge grades."""
    return probe_row(pm_id, n, day).model_copy(
        update={
            "probe_type": ProbeType.TRAIT_PRESENCE,
            "option_a": "yes",
            "option_b": "no",
            "option_c": None,
            "source_b": None,
            "source_c": None,
        }
    )


def traits_for(pm_id: str) -> list[Trait]:
    """Two active biases (t_01, t_02) and one preference (t_03)."""
    return [
        trait(pm_id, "t_01", "loss_aversion_lambda", Kind.BIAS, 2.0),
        trait(pm_id, "t_02", "disposition_ratio", Kind.BIAS, 2.0),
        trait(pm_id, "t_03", "response_format", Kind.PREFERENCE, "short bullets"),
    ]


def dormant_event(pm_id: str, trait_id: str, day: date) -> DriftEvent:
    """A dormant event on `trait_id`."""
    return drift(pm_id, day, DriftEventType.DORMANT, trait_id)


def revive_event(pm_id: str, trait_id: str, day: date) -> DriftEvent:
    """A revive event on `trait_id`."""
    return drift(pm_id, day, DriftEventType.REVIVE, trait_id)


def bank() -> ProbeBank:
    """The packaged probe bank."""
    return load_catalogue().probes


def verdict_reply(judge: Judge, **fields: object) -> dict:
    """A fake model message whose text is a verdict JSON object."""
    payload = {"rationale": "r", **fields}
    return fake_message([{"type": "text", "text": json.dumps(payload)}])
