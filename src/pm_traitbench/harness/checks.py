"""Deterministic format checks for routine-question replies and the check map."""

import re
from importlib import resources
from pathlib import Path
from typing import assert_never

import yaml

from pm_traitbench.catalogues.loader import Catalogue
from pm_traitbench.enums import CheckKind, FormatOutcome
from pm_traitbench.errors import HarnessError

CHECKED_PARAMS: tuple[str, ...] = (
    "response_format",
    "number_language",
    "length_on_routine_questions",
    "hedging_language",
)

_LIST_MARKER = re.compile(r"^\s*([-*•]|\d+[.)])\s+")
_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_HEADER = re.compile(r"^#{1,6}\s+\S")
# No leading \b on "bp" so a glued "12bp" matches; letters before it still do not.
_BP = re.compile(r"(?<![a-z])bps?\b|basis points?", re.IGNORECASE)
_PERCENT = re.compile(r"%|\bper ?cent\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEDGES = re.compile(r"\b(might|could|perhaps|possibly|likely|unlikely|uncertain)\b", re.IGNORECASE)
# Case-sensitive: a capitalised "May" is usually the month.
_MAY = re.compile(r"\bmay\b")
# A bare percentage is a quoted move, so a level must sit next to confidence or conviction.
_LEVEL = r"(high|medium|low|\d+(\.\d+)?\s*%)"
_CONFIDENCE = re.compile(
    rf"\b{_LEVEL}\s+(confidence|conviction)\b"
    rf"|\b(confidence|conviction)(\s+level)?\s*(of|at|is|:)?\s*{_LEVEL}(?!\w)",
    re.IGNORECASE,
)


def load_check_map(
    catalogue: Catalogue, path: Path | None = None
) -> dict[tuple[str, str], CheckKind]:
    """Map each (param, value) of the checked preferences to its check kind.

    Every catalogue value of a checked param must be mapped and every mapped
    value must exist in the catalogue, so a new value is never silently unscored.
    """
    if path is None:
        text = resources.files("pm_traitbench.harness").joinpath("checks.yaml").read_text("utf-8")
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HarnessError(f"cannot read check map {path}: {exc}") from exc
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise HarnessError("check map must be a mapping of param to value to check kind")
    catalogue_values = {p.param: p.values for p in catalogue.preferences}
    extra = sorted(set(raw) - set(CHECKED_PARAMS), key=str)
    if extra:
        raise HarnessError(f"check map has unchecked params: {', '.join(map(str, extra))}")
    result: dict[tuple[str, str], CheckKind] = {}
    for param in CHECKED_PARAMS:
        entries = raw.get(param)
        if not isinstance(entries, dict):
            raise HarnessError(f"check map is missing param {param}")
        known = set(catalogue_values.get(param, ()))
        for value in sorted(known - set(entries), key=str):
            raise HarnessError(f"check map has no entry for {param}: {value}")
        for value, kind in sorted(entries.items(), key=str):
            if value not in known:
                raise HarnessError(f"check map names {param} value not in catalogue: {value}")
            try:
                result[(param, value)] = CheckKind(kind)
            except ValueError as exc:
                raise HarnessError(f"unknown check kind for {param}: {value}: {kind}") from exc
    return result


def _lines(reply: str) -> list[str]:
    return [line.rstrip() for line in reply.strip().splitlines() if line.strip()]


def _is_table_row(line: str) -> bool:
    return line.count("|") >= 2


def _bullets(reply: str) -> bool:
    lines = _lines(reply)
    return len(lines) >= 2 and all(_LIST_MARKER.match(line) for line in lines)


def _prose_paragraph(reply: str) -> bool:
    if re.search(r"\n\s*\n", reply.strip()):
        return False
    return not any(
        _LIST_MARKER.match(line) or _is_table_row(line) or _HEADER.match(line)
        for line in _lines(reply)
    )


def _table(reply: str) -> bool:
    lines = _lines(reply)
    return any(
        _is_table_row(current) and _SEPARATOR_ROW.match(following)
        for current, following in zip(lines, lines[1:], strict=False)
    )


def _headers(reply: str) -> bool:
    return sum(1 for line in _lines(reply) if _HEADER.match(line)) >= 2


def _units(reply: str, want_bp: bool, want_percent: bool) -> FormatOutcome:
    has_bp = bool(_BP.search(reply))
    has_percent = bool(_PERCENT.search(reply))
    if not (has_bp or has_percent):
        return FormatOutcome.NOT_APPLICABLE
    return _outcome(has_bp == want_bp and has_percent == want_percent)


def _sentence_count(reply: str) -> int:
    return sum(1 for piece in _SENTENCE_SPLIT.split(reply.strip()) if piece)


def _outcome(passed: bool) -> FormatOutcome:
    return FormatOutcome.PASS if passed else FormatOutcome.FAIL


def run_check(kind: CheckKind, reply: str, short_page_words: int) -> FormatOutcome:
    """Run one deterministic check; judge values are filtered out by callers."""
    match kind:
        case CheckKind.BULLETS:
            return _outcome(_bullets(reply))
        case CheckKind.PROSE_PARAGRAPH:
            return _outcome(_prose_paragraph(reply))
        case CheckKind.TABLE:
            return _outcome(_table(reply))
        case CheckKind.HEADERS:
            return _outcome(_headers(reply))
        case CheckKind.UNITS_BP:
            return _units(reply, want_bp=True, want_percent=False)
        case CheckKind.UNITS_PERCENT:
            return _units(reply, want_bp=False, want_percent=True)
        case CheckKind.UNITS_BOTH:
            return _units(reply, want_bp=True, want_percent=True)
        case CheckKind.ONE_SENTENCE:
            return _outcome(_sentence_count(reply) == 1)
        case CheckKind.TWO_TO_THREE_SENTENCES:
            return _outcome(2 <= _sentence_count(reply) <= 3)
        case CheckKind.SHORT_PAGE:
            return _outcome(len(reply.split()) <= short_page_words)
        case CheckKind.NO_HEDGES:
            return _outcome(not (_HEDGES.search(reply) or _MAY.search(reply)))
        case CheckKind.CONFIDENCE_LEVEL:
            return _outcome(bool(_CONFIDENCE.search(reply)))
        case CheckKind.JUDGE:
            raise ValueError("judge values have no deterministic check; filter them first")
        case _:
            assert_never(kind)


def parse_routine_answer(answer: str) -> tuple[tuple[str, str], ...]:
    """Parse a routine-question answer key into (param, value) pairs."""
    prefix, suffix = "format: ", "; intrusion: none"
    if not (answer.startswith(prefix) and answer.endswith(suffix)):
        raise HarnessError(f"malformed routine answer: {answer!r}")
    body = answer[len(prefix) : len(answer) - len(suffix)]
    if body == "none":
        return ()
    pairs = []
    for item in body.split("; "):
        param, sep, value = item.partition("=")
        if not sep or not param or not value or ";" in item or "=" in value:
            raise HarnessError(f"malformed routine answer item: {item!r}")
        pairs.append((param, value))
    return tuple(pairs)
