"""Shared gate 2 test fixtures: `Session`, `Trait` and `Signal` builders, and recovery
reply and request-routing helpers used across the gate 2 test modules.

Extended by later gate 2 test modules, so a row's shape only has to match
`schema.py` in one place.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from pm_traitbench.enums import Kind, Ownership, SessionKind, SignalMode, TurnRole, Valence
from pm_traitbench.signals.assemble import session_id
from pm_traitbench.tables.schema import Session, Signal, Trait, Turn
from tests.dialogue.fixtures import fake_message

PM_A = "pm_001"
PM_B = "pm_002"


def session_of(
    pm_id: str,
    day: date,
    pm_texts: Sequence[str],
    advisor_texts: Sequence[str] | None = None,
    kind: SessionKind = SessionKind.CHECK_IN,
    trade_idea_ids: tuple[str, ...] = (),
    letter: int = 0,
) -> Session:
    """A `Session` for `pm_id` on `day`, alternating pm and advisor turns from the given texts.

    Advisor texts default to `"noted"` for each pm turn. `letter` selects the
    session's index among same-day sessions, per `signals.assemble.session_id`.
    """
    if advisor_texts is None:
        advisor_texts = ["noted"] * len(pm_texts)
    turns: list[Turn] = []
    for pm_text, advisor_text in zip(pm_texts, advisor_texts, strict=True):
        turns.append(Turn(role=TurnRole.PM, text=pm_text))
        turns.append(Turn(role=TurnRole.ADVISOR, text=advisor_text))
    return Session(
        session_id=session_id(pm_id, day, letter),
        pm_id=pm_id,
        date=day,
        kind=kind,
        trade_idea_ids=tuple(trade_idea_ids),
        turns=tuple(turns),
    )


def trait(
    pm_id: str, trait_id: str, param: str, kind: Kind, value: float | str, active: bool = True
) -> Trait:
    """A `Trait` for `param`; a bias gets neutral positive multipliers, a preference none."""
    if kind == Kind.BIAS:
        return Trait(
            pm_id=pm_id,
            trait_id=trait_id,
            kind=kind,
            param=param,
            value=value,
            active=active,
            mult_range=1.0,
            mult_risk_off=1.0,
            mult_risk_on=1.0,
        )
    return Trait(
        pm_id=pm_id,
        trait_id=trait_id,
        kind=kind,
        param=param,
        value=value,
        active=True,
        mult_range=None,
        mult_risk_off=None,
        mult_risk_on=None,
    )


def signal(
    pm_id: str,
    session_id: str,
    day: date,
    trait_id: str,
    mode: SignalMode = SignalMode.STATED,
    valence: Valence = Valence.CONFIRM,
    ownership: Ownership = Ownership.SELF,
    signal_id: str | None = None,
) -> Signal:
    """A `Signal` on `trait_id`; `signal_id` defaults to one keyed off `trait_id`'s number."""
    return Signal(
        signal_id=signal_id or f"sg_{trait_id.removeprefix('t_').zfill(3)}",
        pm_id=pm_id,
        session_id=session_id,
        date=day,
        trait_id=trait_id,
        mode=mode,
        trade_idea_id=None,
        valence=valence,
        ownership=ownership,
        third_party_value=None,
        claim_session_id=None,
    )


def recovery_reply(
    biases: Mapping[str, tuple[bool, Sequence[str]]],
    preferences: Mapping[str, tuple[str | None, Sequence[str]]],
) -> dict:
    """A `fake_message` whose text is a recovery reply: `{param: (present/value, session_ids)}`."""
    payload = {
        "biases": [
            {"param": param, "present": present, "session_ids": list(session_ids)}
            for param, (present, session_ids) in biases.items()
        ],
        "preferences": [
            {"param": param, "value": value, "session_ids": list(session_ids)}
            for param, (value, session_ids) in preferences.items()
        ],
    }
    return fake_message([{"type": "text", "text": json.dumps(payload)}])


def is_recovery_request(request: Mapping[str, Any]) -> bool:
    """True when `request` is a gate 2 recovery request, by its schema's title."""
    schema = request.get("output_config", {}).get("format", {}).get("schema", {})
    return schema.get("title") == "gate2_recovery"
