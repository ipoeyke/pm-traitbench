"""Tests for the parameter-name grep layer."""

import pytest

from pm_traitbench.catalogues.loader import leak_param_names, load_catalogue
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.dialogue.validate.grep import check_grep
from tests.dialogue.fixtures import SESSION_ID
from tests.dialogue.validate.fixtures import advisor_turn, log_of, pm_turn

PARAMS = ("pushback_style", "loss_aversion_lambda")


def _log(*turns):
    return log_of(SESSION_ID, "pm_001", turns)


def test_clean_transcript_passes():
    log = _log(pm_turn("I bought the dip"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ()


def test_param_name_with_underscores_fails():
    log = _log(pm_turn("my loss_aversion_lambda is high"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ("names a parameter: loss_aversion_lambda",)


def test_bias_param_as_phrase_fails():
    log = _log(pm_turn("my loss aversion lambda is high"), advisor_turn("noted"))

    # "loss aversion" is also a banned bias phrase, so both reasons fire.
    assert check_grep(log, PARAMS) == (
        "names a parameter: loss aversion",
        "names a parameter: loss_aversion_lambda",
    )


def test_preference_param_fails_only_as_its_raw_identifier():
    params = leak_param_names(load_catalogue())

    raw = _log(pm_turn("set my macro_commentary to none"), advisor_turn("noted"))
    assert check_grep(raw, params) == ("names a parameter: macro_commentary",)

    stated = _log(
        pm_turn("no macro commentary unless I ask, and keep the register blunt"),
        advisor_turn("noted"),
    )
    assert check_grep(stated, params) == ()


def test_banned_bias_phrase_fails():
    log = _log(pm_turn("that is my extrapolation bias talking"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ("names a parameter: extrapolation bias",)


def test_banned_phrase_stem_matches_its_inflections():
    log = _log(pm_turn("maybe I am overconfident here"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ("names a parameter: overconfiden",)


@pytest.mark.parametrize(
    "text",
    [
        "that is where the view is anchored",
        "I run a long bias in this book",
        "extrapolating the trend, the curve looks rich",
        "the disposition of the position is unchanged",
        "shepherd the order through the close",
    ],
)
def test_ordinary_desk_words_pass(text):
    log = _log(pm_turn(text), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ()


def test_advisor_turn_is_not_grepped():
    log = _log(
        pm_turn("all clear"),
        advisor_turn("my loss_aversion_lambda take is this, driven by loss aversion"),
    )

    assert check_grep(log, PARAMS) == ()


def test_turn_naming_two_params_reports_both():
    log = _log(pm_turn("loss_aversion_lambda and pushback_style both matter"), advisor_turn("ok"))

    assert check_grep(log, PARAMS) == (
        "names a parameter: loss_aversion_lambda",
        "names a parameter: pushback_style",
    )


def test_case_insensitive_for_params_and_banned_phrases():
    log = _log(
        pm_turn("PUSHBACK_STYLE is what I said, driven by Loss Aversion"), advisor_turn("ok")
    )

    assert check_grep(log, PARAMS) == (
        "names a parameter: loss aversion",
        "names a parameter: pushback_style",
    )


def test_whole_word_only():
    log = _log(pm_turn("my pushback_styles differ"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ()


def test_reasons_sorted_unique():
    log = _log(
        pm_turn("my loss_aversion_lambda again"),
        advisor_turn("noted"),
        pm_turn("my pushback_style is firm"),
        advisor_turn("noted again"),
        pm_turn("loss aversion lambda repeated"),
        advisor_turn("noted a third time"),
    )

    assert check_grep(log, PARAMS) == (
        "names a parameter: loss aversion",
        "names a parameter: loss_aversion_lambda",
        "names a parameter: pushback_style",
    )


def test_leak_param_names_is_biases_then_catalogue_preferences():
    catalogue = load_catalogue()
    params = leak_param_names(catalogue)

    assert params.index("register") > params.index(BIAS_PARAMS[-1])


def test_stating_any_catalogue_preference_value_passes():
    catalogue = load_catalogue()
    params = leak_param_names(catalogue)

    for entry in catalogue.preferences:
        for value in entry.values:
            log = _log(pm_turn(f"my rule: {value}"), advisor_turn("noted"))
            assert check_grep(log, params) == (), (entry.param, value)
