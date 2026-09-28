"""Tests for the leakage and forbidden-trait judge requests, parsers and label mapping."""

import asyncio
import json

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import CachedClient
from pm_traitbench.dialogue.validate.judge import (
    FORBIDDEN_SCHEMA,
    LEAK_SCHEMA,
    LeakVerdict,
    Violation,
    forbidden_request,
    leak_request,
    map_label,
    parse_forbidden,
    parse_leak,
    send_judged,
    transcript_text,
)
from pm_traitbench.errors import ValidateError
from tests.dialogue.fixtures import FakeClient, fake_message
from tests.dialogue.validate.fixtures import (
    advisor_turn,
    forbidden_reply,
    is_forbidden_request,
    is_leak_request,
    leak_reply,
    log_of,
    pm_turn,
)

_CONFIG = Config().validation


def _log():
    return log_of(
        "s_pm001_2026-01-05_a",
        "pm_001",
        [pm_turn("I bought the dip"), advisor_turn("noted, anything else")],
    )


def test_transcript_text_labels_roles():
    log = _log()

    assert transcript_text(log) == "PM: I bought the dip\n\nADVISOR: noted, anything else"


def test_leak_request_carries_only_transcript_and_allowed_keys():
    log = _log()

    request = leak_request(log, _CONFIG)

    assert set(request.keys()) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert "tools" not in request
    assert "cache_control" not in request
    assert "thinking" not in request
    assert request["messages"] == [{"role": "user", "content": transcript_text(log)}]
    assert request["model"] == _CONFIG.judge_model
    assert request["max_tokens"] == _CONFIG.max_output_tokens
    assert request["output_config"]["format"]["schema"] is LEAK_SCHEMA
    assert is_leak_request(request)
    assert not is_forbidden_request(request)


def test_forbidden_request_numbers_avoid_lines_from_one():
    log = _log()
    avoid_lines = ["never mention position size", "never predict a specific price"]

    request = forbidden_request(log, avoid_lines, _CONFIG)

    assert set(request.keys()) == {"model", "max_tokens", "system", "messages", "output_config"}
    content = request["messages"][0]["content"]
    assert content == (
        f"{transcript_text(log)}\n\nThe PM must not:\n"
        "1. never mention position size\n"
        "2. never predict a specific price"
    )
    assert request["output_config"]["format"]["schema"] is FORBIDDEN_SCHEMA
    assert is_forbidden_request(request)
    assert not is_leak_request(request)


def test_judge_requests_never_carry_a_turn_directive():
    """A PM turn's `directive` (its stance line) must never reach a judge request: only its
    narrated `text` does.
    """
    directive = "push to run this at twice the usual size"
    log = log_of(
        "s_pm001_2026-01-05_a",
        "pm_001",
        [pm_turn("I bought the dip", directive=directive), advisor_turn("noted")],
    )

    leak_dump = json.dumps(leak_request(log, _CONFIG))
    forbidden_dump = json.dumps(forbidden_request(log, ["never mention position size"], _CONFIG))

    assert directive not in leak_dump
    assert directive not in forbidden_dump


def test_parse_leak_accepts_a_valid_reply_and_rejects_bad_shapes():
    valid = leak_reply(True, "loss aversion", "I always average down")
    assert parse_leak(valid) == LeakVerdict(
        explicit=True, label="loss aversion", quote="I always average down"
    )

    string_explicit = fake_message(
        [{"type": "text", "text": '{"explicit": "yes", "label": null, "quote": ""}'}]
    )
    assert parse_leak(string_explicit) is None

    missing_quote = fake_message([{"type": "text", "text": '{"explicit": false, "label": null}'}])
    assert parse_leak(missing_quote) is None

    hit_max_tokens = leak_reply(True, "loss aversion", "quote")
    hit_max_tokens = {**hit_max_tokens, "stop_reason": "max_tokens"}
    assert parse_leak(hit_max_tokens) is None


def test_parse_forbidden_accepts_empty_and_filled_lists_and_rejects_non_integer_index():
    empty = forbidden_reply([])
    assert parse_forbidden(empty) == ()

    filled = forbidden_reply([(1, "quote one"), (2, "quote two")])
    assert parse_forbidden(filled) == (
        Violation(index=1, quote="quote one"),
        Violation(index=2, quote="quote two"),
    )

    bad_index = fake_message(
        [{"type": "text", "text": '{"violations": [{"index": "1", "quote": "q"}]}'}]
    )
    assert parse_forbidden(bad_index) is None


def test_map_label_uses_catalogue_phrases_case_insensitively():
    labels = load_catalogue().bias_labels

    assert map_label("Disposition effect", labels) == "disposition_ratio"
    assert map_label("I like coffee", labels) is None
    assert map_label(None, labels) is None
    assert map_label("   ", labels) is None
    # Matches both loss_aversion_lambda ("loss aversion") and disposition_ratio
    # ("disposition"); the earlier param in BIAS_PARAMS order must win.
    assert map_label("loss aversion and disposition too", labels) == "loss_aversion_lambda"


def _cached_client(responder, tmp_path):
    fake = FakeClient(responder)
    return fake, CachedClient(lambda: fake, tmp_path, token_budget=None)


def test_send_judged_retries_then_commits(tmp_path):
    request = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return fake_message([{"type": "text", "text": "not json"}])
        return leak_reply(True, "loss aversion", "I always average down")

    fake, client = _cached_client(responder, tmp_path)

    parsed, rejected = asyncio.run(
        send_judged(client, request, parse_leak, "s_test", max_retries=2)
    )

    assert rejected == 1
    assert parsed == LeakVerdict(
        explicit=True, label="loss aversion", quote="I always average down"
    )
    assert calls["n"] == 2

    parsed_again, rejected_again = asyncio.run(
        send_judged(client, request, parse_leak, "s_test", max_retries=2)
    )
    assert rejected_again == 0
    assert parsed_again == parsed
    assert client.totals.cache_hits == 1


def test_send_judged_raises_validate_error_at_the_cap(tmp_path):
    request = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}

    def responder(req):
        return fake_message([{"type": "text", "text": "not json"}])

    fake, client = _cached_client(responder, tmp_path)

    with pytest.raises(
        ValidateError, match=r"session s_test: the judge reply was unparsable or schema-invalid"
    ):
        asyncio.run(send_judged(client, request, parse_leak, "s_test", max_retries=1))

    assert len(fake.requests) == 2  # 1 + max_retries


def test_send_judged_bypasses_a_cached_reply_that_fails_the_real_parser(tmp_path):
    """A reply already committed to the cache under a permissive parser must not be replayed
    forever once a stricter parser rejects it: the retry must fetch a fresh reply instead.
    """
    request = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return fake_message([{"type": "text", "text": "not json"}])
        return leak_reply(True, "loss aversion", "I always average down")

    fake, client = _cached_client(responder, tmp_path)

    # Prime the cache with an unparsable reply, via a parser that accepts anything.
    asyncio.run(send_judged(client, request, lambda response: response, "s_test", max_retries=0))
    assert len(fake.requests) == 1

    parsed, rejected = asyncio.run(
        send_judged(client, request, parse_leak, "s_test", max_retries=1)
    )

    assert len(fake.requests) == 2  # the stale cache hit is bypassed by exactly one fresh call
    assert rejected == 1
    assert parsed == LeakVerdict(
        explicit=True, label="loss aversion", quote="I always average down"
    )
    assert client.totals.cache_hits == 1
