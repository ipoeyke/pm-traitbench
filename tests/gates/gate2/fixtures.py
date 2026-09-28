"""Shared gate 2 test fixtures: `Session`, `Trait` and `Signal` builders, dialogue log and
skeleton builders, and recovery and classification reply and request-routing helpers used
across the gate 2 test modules.

Extended by later gate 2 test modules, so a row's shape only has to match
`schema.py` in one place.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import BIAS_PARAMS, DEFAULT_MODEL
from pm_traitbench.dialogue.usage import ZERO_USAGE
from pm_traitbench.enums import (
    Kind,
    Ownership,
    SessionKind,
    SignalMode,
    StanceEntry,
    TurnRole,
    Valence,
)
from pm_traitbench.gates.gate2.recover import compute_truth
from pm_traitbench.signals.assemble import session_id
from pm_traitbench.tables.schema import (
    DialogueLog,
    Session,
    Signal,
    Skeleton,
    Stance,
    Trait,
    Turn,
    TurnLog,
)
from pm_traitbench.tables.specs import (
    DIALOGUE_LOGS,
    DRIFT_EVENTS,
    PERSONAS,
    SESSIONS,
    SIGNALS,
    SKELETONS,
    TRAITS,
)
from pm_traitbench.tables.store import DataStore
from tests.dialogue.fixtures import default_responder, fake_message

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


_REQUEST_HASH = "0" * 64


def _turn_log(role: TurnRole, text: str, directive: str | None) -> TurnLog:
    """A turn log entry with one request hash, zero usage and no tool calls or mentions."""
    return TurnLog(
        role=role,
        text=text,
        mentions=(),
        directive=directive,
        scripted_violation=False,
        tool_calls=(),
        model=DEFAULT_MODEL,
        request_hashes=(_REQUEST_HASH,),
        usage=ZERO_USAGE,
    )


def log_with_directives(session: Session, directives: Sequence[str | None]) -> DialogueLog:
    """A `DialogueLog` mirroring `session`'s turns; `directives` fills its PM turns in order,
    one per PM turn, and every advisor turn carries none.
    """
    pm_positions = [i for i, turn in enumerate(session.turns) if turn.role == TurnRole.PM]
    directive_by_position = dict(zip(pm_positions, directives, strict=True))
    turns = tuple(
        _turn_log(turn.role, turn.text, directive_by_position.get(i))
        for i, turn in enumerate(session.turns)
    )
    return DialogueLog(
        session_id=session.session_id, pm_id=session.pm_id, voice_id="v_01", turns=turns
    )


def skeleton_with_stances(session: Session, stances: Sequence[Stance]) -> Skeleton:
    """A skeleton sitting on `session`'s own id, date and ideas, with no advisor violation."""
    return Skeleton(
        session_id=session.session_id,
        pm_id=session.pm_id,
        date=session.date,
        kind=session.kind,
        trade_idea_ids=session.trade_idea_ids,
        stances=tuple(stances),
        advisor_violation=None,
        forbidden_trait_ids=(),
        forbidden_pref_params=(),
    )


def classify_reply(statements: Sequence[tuple[str, str]]) -> dict:
    """A `fake_message` whose text is a classification reply: `(quote, kind)` pairs."""
    payload = {"statements": [{"quote": quote, "kind": kind} for quote, kind in statements]}
    return fake_message([{"type": "text", "text": json.dumps(payload)}])


def is_classify_request(request: Mapping[str, Any]) -> bool:
    """True when `request` is a gate 2 classification request, by its schema's title."""
    schema = request.get("output_config", {}).get("format", {}).get("schema", {})
    return schema.get("title") == "gate2_classification"


# A recovery transcript's or a classify session's own "Session s_pmNNN_..." header line.
_SESSION_HEADER_RE = re.compile(r"Session (s_pm(\d+)_\S+),")
_FIND_N_RE = re.compile(r"Find (\d+) statements")


def truthful_responder(store: DataStore) -> Callable[[dict], dict]:
    """A responder that answers every gate 2 request with the store's own ground truth.

    A recovery request names no PM, so the PM is read off the transcript's first
    session header; every active bias comes back present, citing every session
    carrying a signal of its trait, and every held preference comes back with its
    true value, citing likewise. A classification request is answered with one
    statement per classifiable signal, quoting the first words of its directive
    turn with the trait's own kind. Any other request falls back to
    `default_responder`.
    """
    catalogue = load_catalogue()
    personas_by_id = {p.pm_id: p for p in store.read(PERSONAS)}
    traits_by_pm: dict[str, list[Trait]] = {}
    for t in store.read(TRAITS):
        traits_by_pm.setdefault(t.pm_id, []).append(t)
    drift_by_pm: dict[str, list] = {}
    for d in store.read(DRIFT_EVENTS):
        drift_by_pm.setdefault(d.pm_id, []).append(d)
    signals_by_pm: dict[str, list[Signal]] = {}
    for sig in store.read(SIGNALS):
        signals_by_pm.setdefault(sig.pm_id, []).append(sig)
    sessions_by_pm: dict[str, list[Session]] = {}
    for s in store.read(SESSIONS):
        sessions_by_pm.setdefault(s.pm_id, []).append(s)
    skeletons_by_session = {sk.session_id: sk for sk in store.read(SKELETONS)}
    logs_by_session = {log.session_id: log for log in store.read(DIALOGUE_LOGS)}

    def _citations(pm_id: str, trait_id: str | None) -> list[str]:
        if trait_id is None:
            return []
        return sorted(
            sig.session_id for sig in signals_by_pm.get(pm_id, ()) if sig.trait_id == trait_id
        )

    def recovery(request: Mapping[str, Any]) -> dict:
        transcript = request["messages"][0]["content"]
        match = _SESSION_HEADER_RE.search(transcript)
        assert match is not None, "no session header found in recovery transcript"
        pm_id = f"pm_{match.group(2)}"
        persona = personas_by_id[pm_id]
        pm_traits = traits_by_pm.get(pm_id, [])
        pm_drift = drift_by_pm.get(pm_id, [])
        pm_sessions = sessions_by_pm[pm_id]
        last_date = max(s.date for s in pm_sessions)
        entries = catalogue.preferences_for(persona.mandate.asset_class)
        truth = compute_truth(pm_traits, pm_drift, last_date, entries)
        pref_trait_by_param = {t.param: t for t in pm_traits if t.kind == Kind.PREFERENCE}

        biases = {}
        for param in BIAS_PARAMS:
            present = bool(truth[param].truth_active)
            biases[param] = (present, _citations(pm_id, truth[param].trait_id) if present else [])

        preferences = {}
        for entry in entries:
            value = truth[entry.param].truth_value
            held = pref_trait_by_param.get(entry.param)
            citing = _citations(pm_id, held.trait_id) if held is not None and value else []
            preferences[entry.param] = (value, citing)

        return recovery_reply(biases, preferences)

    def classification(request: Mapping[str, Any]) -> dict:
        user_content = request["messages"][0]["content"]
        match = _SESSION_HEADER_RE.search(user_content)
        assert match is not None, "no session header found in classify request"
        classify_session_id = match.group(1)
        log = logs_by_session[classify_session_id]
        skeleton = skeletons_by_session[classify_session_id]
        pm_traits_kind = {t.trait_id: t.kind for t in traits_by_pm.get(log.pm_id, [])}
        directive_text = {
            turn.directive: turn.text
            for turn in log.turns
            if turn.role == TurnRole.PM and turn.directive is not None
        }

        statements: list[tuple[str, str]] = []
        for stance in skeleton.stances:
            if stance.entry == StanceEntry.CLAIM:
                continue
            text = directive_text.get(stance.stance)
            kind = pm_traits_kind.get(stance.trait_id)
            if text is None or kind is None:
                continue
            quote = " ".join(text.split()[:3])
            statements.append((quote, kind.value))

        n_match = _FIND_N_RE.search(user_content)
        n = int(n_match.group(1)) if n_match else len(statements)
        return classify_reply(statements[:n])

    def responder(request: Mapping[str, Any]) -> dict:
        if is_recovery_request(request):
            return recovery(request)
        if is_classify_request(request):
            return classification(request)
        return default_responder(request)

    return responder


def wrong_responder(store: DataStore) -> Callable[[dict], dict]:
    """Like `truthful_responder`, but every recovery bias flipped and every value nulled."""
    truthful = truthful_responder(store)

    def responder(request: Mapping[str, Any]) -> dict:
        if not is_recovery_request(request):
            return truthful(request)
        reply = truthful(request)
        payload = json.loads(reply["content"][0]["text"])
        for bias in payload["biases"]:
            bias["present"] = not bias["present"]
        for pref in payload["preferences"]:
            pref["value"] = None
        return fake_message([{"type": "text", "text": json.dumps(payload)}])

    return responder
