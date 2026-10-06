"""Tests for the leakage, forbidden-trait and stance judge requests, parsers, label mapping
and refusal fallback."""

import asyncio
import json

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.dialogue.client import CachedClient
from pm_traitbench.dialogue.validate.judge import (
    FORBIDDEN_SCHEMA,
    LEAK_LABELS,
    LEAK_SCHEMA,
    STANCE_SCHEMA,
    STANCE_SYSTEM,
    LeakVerdict,
    StanceVerdict,
    Violation,
    forbidden_request,
    leak_request,
    parse_forbidden,
    parse_leak,
    parse_stance,
    send_judged,
    stance_request,
    transcript_text,
)
from pm_traitbench.errors import ValidateError
from tests.dialogue.fixtures import SESSION_ID, FakeClient, fake_message
from tests.dialogue.validate.fixtures import (
    advisor_turn,
    forbidden_reply,
    is_forbidden_request,
    is_leak_request,
    is_stance_request,
    leak_reply,
    log_of,
    pm_turn,
    stance_reply,
)

_CONFIG = Config().validation


def _log():
    return log_of(
        SESSION_ID,
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
        SESSION_ID,
        "pm_001",
        [pm_turn("I bought the dip", directive=directive), advisor_turn("noted")],
    )

    leak_dump = json.dumps(leak_request(log, _CONFIG))
    forbidden_dump = json.dumps(forbidden_request(log, ["never mention position size"], _CONFIG))

    assert directive not in leak_dump
    assert directive not in forbidden_dump


def test_stance_request_tags_the_message_before_the_instruction():
    """The PM message comes first and both parts sit in XML tags, which keeps the API from
    reading the request as a reasoning-extraction attempt and refusing it.
    """
    pm_text = "Holding the full size, the thesis has not changed."
    stance = "refuse to trim the position"

    request = stance_request(pm_text, stance, _CONFIG)

    assert set(request.keys()) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert request["system"] == STANCE_SYSTEM
    assert request["messages"] == [
        {
            "role": "user",
            "content": (
                f"<pm_message>{pm_text}</pm_message>\n\n<instruction>{stance}</instruction>"
            ),
        }
    ]
    assert request["model"] == _CONFIG.judge_model
    assert request["output_config"]["format"]["schema"] is STANCE_SCHEMA
    assert is_stance_request(request)
    assert not is_leak_request(request)
    assert not is_forbidden_request(request)


def test_stance_request_keeps_tag_like_text_inside_its_tags():
    """Angle brackets in the PM text or stance line are passed through verbatim, each still
    wrapped in exactly one pair of its own tags.
    """
    content = stance_request("buy <b>now</b>", "say <no>", _CONFIG)["messages"][0]["content"]

    assert content.startswith("<pm_message>buy <b>now</b></pm_message>")
    assert content.endswith("<instruction>say <no></instruction>")
    assert content.count("<pm_message>") == content.count("<instruction>") == 1


def test_parse_stance_accepts_a_valid_reply_and_rejects_bad_shapes():
    assert parse_stance(stance_reply(True, "held firm")) == StanceVerdict(
        carried_out=True, reason="held firm"
    )

    string_flag = fake_message([{"type": "text", "text": '{"carried_out": "yes", "reason": "r"}'}])
    assert parse_stance(string_flag) is None

    refused = {**stance_reply(True, "held firm"), "stop_reason": "refusal"}
    assert parse_stance(refused) is None


def test_parse_leak_accepts_a_valid_reply_and_rejects_bad_shapes():
    valid = leak_reply(True, "loss_aversion_lambda", "I always average down")
    assert parse_leak(valid) == LeakVerdict(
        explicit=True, label="loss_aversion_lambda", quote="I always average down"
    )
    assert parse_leak(leak_reply(True, "other", "I like coffee")).label == "other"
    assert parse_leak(leak_reply(True, "loss aversion", "quote")) is None

    string_explicit = fake_message(
        [{"type": "text", "text": '{"explicit": "yes", "label": null, "quote": ""}'}]
    )
    assert parse_leak(string_explicit) is None

    missing_quote = fake_message([{"type": "text", "text": '{"explicit": false, "label": null}'}])
    assert parse_leak(missing_quote) is None

    hit_max_tokens = leak_reply(True, "loss_aversion_lambda", "quote")
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


_REQUEST = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
_UNPARSABLE = fake_message([{"type": "text", "text": "not json"}])
_VERDICT = LeakVerdict(explicit=True, label="loss_aversion_lambda", quote="I always average down")


def _cached_client(responder, tmp_path):
    fake = FakeClient(responder)
    return fake, CachedClient(lambda: fake, tmp_path, token_budget=None)


def _unparsable_then_valid(tmp_path):
    """A client whose first reply is unparsable and every later one a valid leak verdict."""

    def responder(_request):
        if len(fake.requests) == 1:
            return _UNPARSABLE
        return leak_reply(_VERDICT.explicit, _VERDICT.label, _VERDICT.quote)

    fake, client = _cached_client(responder, tmp_path)
    return fake, client


def test_send_judged_retries_then_commits(tmp_path):
    fake, client = _unparsable_then_valid(tmp_path)

    parsed, rejected, fell_back = asyncio.run(
        send_judged(client, _REQUEST, parse_leak, "s_test", max_retries=2)
    )

    assert rejected == 1
    assert parsed == _VERDICT
    assert fell_back is False
    assert len(fake.requests) == 2

    parsed_again, rejected_again, _ = asyncio.run(
        send_judged(client, _REQUEST, parse_leak, "s_test", max_retries=2)
    )
    assert rejected_again == 0
    assert parsed_again == parsed
    assert client.totals.cache_hits == 1


def test_send_judged_raises_validate_error_at_the_cap(tmp_path):
    fake, client = _cached_client(lambda _request: _UNPARSABLE, tmp_path)

    with pytest.raises(
        ValidateError, match=r"session s_test: the judge reply was unparsable or schema-invalid"
    ):
        asyncio.run(send_judged(client, _REQUEST, parse_leak, "s_test", max_retries=1))

    assert len(fake.requests) == 2  # 1 + max_retries


def test_send_judged_bypasses_a_cached_reply_that_fails_the_real_parser(tmp_path):
    """A reply already committed to the cache under a permissive parser must not be replayed
    forever once a stricter parser rejects it: the retry must fetch a fresh reply instead.
    """
    fake, client = _unparsable_then_valid(tmp_path)

    # Prime the cache with an unparsable reply, via a parser that accepts anything.
    asyncio.run(send_judged(client, _REQUEST, lambda response: response, "s_test", max_retries=0))
    assert len(fake.requests) == 1

    parsed, rejected, fell_back = asyncio.run(
        send_judged(client, _REQUEST, parse_leak, "s_test", max_retries=1)
    )

    assert len(fake.requests) == 2  # the stale cache hit is bypassed by exactly one fresh call
    assert rejected == 1
    assert parsed == _VERDICT
    assert client.totals.cache_hits == 1


_REFUSAL = fake_message(
    [{"type": "text", "text": "I can't help with that."}],
    stop_reason="refusal",
)
_FALLBACK = "claude-sonnet-5-5"


def _leak_ok(_request=None):
    return leak_reply(_VERDICT.explicit, _VERDICT.label, _VERDICT.quote)


def _refuse_primary(request):
    return _REFUSAL if request["model"] == _REQUEST["model"] else _leak_ok()


def test_send_judged_falls_back_to_another_model_on_refusal(tmp_path):
    fake, client = _cached_client(_refuse_primary, tmp_path)

    parsed, rejected, fell_back = asyncio.run(
        send_judged(client, _REQUEST, parse_leak, "s_test", max_retries=2, fallback_model=_FALLBACK)
    )

    assert parsed == _VERDICT
    assert rejected == 1
    assert fell_back is True
    assert [r["model"] for r in fake.requests] == [_REQUEST["model"], _FALLBACK]
    assert fake.requests[1] == {**_REQUEST, "model": _FALLBACK}


def test_send_judged_does_not_fall_back_on_an_unparsable_reply(tmp_path):
    fake, client = _unparsable_then_valid(tmp_path)

    parsed, rejected, fell_back = asyncio.run(
        send_judged(client, _REQUEST, parse_leak, "s_test", max_retries=2, fallback_model=_FALLBACK)
    )

    assert parsed == _VERDICT
    assert rejected == 1
    assert fell_back is False
    assert {r["model"] for r in fake.requests} == {_REQUEST["model"]}


def test_send_judged_replays_a_cached_refusal_straight_to_the_fallback(tmp_path):
    """A committed refusal lets a rerun skip the primary call and reuse the cached fallback."""
    fake, client = _cached_client(_refuse_primary, tmp_path)
    send = send_judged(
        client, _REQUEST, parse_leak, "s_test", max_retries=2, fallback_model=_FALLBACK
    )
    asyncio.run(send)

    parsed, rejected, fell_back = asyncio.run(
        send_judged(client, _REQUEST, parse_leak, "s_test", max_retries=2, fallback_model=_FALLBACK)
    )

    assert parsed == _VERDICT
    assert rejected == 1
    assert fell_back is True
    assert len(fake.requests) == 2  # both replies of the rerun came from the cache
    assert client.totals.cache_hits == 2


def test_send_judged_without_fallback_raises_on_repeated_refusal(tmp_path):
    fake, client = _cached_client(lambda _request: _REFUSAL, tmp_path)

    with pytest.raises(ValidateError, match=r"session s_test: the judge reply was refused"):
        asyncio.run(send_judged(client, _REQUEST, parse_leak, "s_test", max_retries=1))

    assert len(fake.requests) == 2
    assert {r["model"] for r in fake.requests} == {_REQUEST["model"]}


def test_send_judged_raises_when_the_fallback_also_refuses(tmp_path):
    fake, client = _cached_client(lambda _request: _REFUSAL, tmp_path)

    with pytest.raises(ValidateError, match=r"session s_test: the judge reply was refused"):
        asyncio.run(
            send_judged(
                client, _REQUEST, parse_leak, "s_test", max_retries=1, fallback_model=_FALLBACK
            )
        )

    # one primary refusal, then 1 + max_retries fallback attempts; no fallback of the fallback
    assert [r["model"] for r in fake.requests] == [_REQUEST["model"], _FALLBACK, _FALLBACK]


def test_leak_schema_label_enum_is_every_bias_param_other_or_null():
    enum = LEAK_SCHEMA["properties"]["label"]["enum"]
    assert enum == [*LEAK_LABELS, None]
    assert set(LEAK_LABELS) == set(BIAS_PARAMS) | {"other"}
