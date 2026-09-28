"""Tests for the parameter-name grep layer."""

from pm_traitbench.catalogues.loader import leak_param_names, load_catalogue
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.dialogue.validate.grep import check_grep
from tests.dialogue.validate.fixtures import advisor_turn, log_of, pm_turn

PARAMS = ("register", "loss_aversion_lambda")


def _log(*turns):
    return log_of("s_pm001_2026-01-05_a", "pm_001", turns)


def test_clean_transcript_passes():
    log = _log(pm_turn("I bought the dip"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ()


def test_param_name_with_underscores_fails():
    log = _log(pm_turn("my loss_aversion_lambda is high"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ("names a parameter: loss_aversion_lambda",)


def test_param_name_as_phrase_fails():
    log = _log(pm_turn("my loss aversion lambda is high"), advisor_turn("noted"))

    # "loss aversion" is also a banned stance word, so both reasons fire.
    assert check_grep(log, PARAMS) == (
        "names a parameter: loss aversion",
        "names a parameter: loss_aversion_lambda",
    )


def test_preference_param_fails():
    log = _log(pm_turn("I like a wide register"), advisor_turn("noted"))
    params = leak_param_names(load_catalogue())

    assert check_grep(log, params) == ("names a parameter: register",)


def test_banned_word_substring_fails():
    log = _log(pm_turn("I keep extrapolating the trend"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ("names a parameter: extrapolat",)


def test_advisor_turn_is_not_grepped():
    log = _log(
        pm_turn("all clear"),
        advisor_turn("my loss_aversion_lambda take is this, driven by loss aversion"),
    )

    assert check_grep(log, PARAMS) == ()


def test_turn_naming_two_params_reports_both():
    log = _log(pm_turn("register and pushback_style both matter"), advisor_turn("noted"))

    assert check_grep(log, ("register", "pushback_style")) == (
        "names a parameter: pushback_style",
        "names a parameter: register",
    )


def test_case_insensitive_for_params_and_banned_words():
    log = _log(pm_turn("REGISTER is what I said, driven by Loss Aversion"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == (
        "names a parameter: loss aversion",
        "names a parameter: register",
    )


def test_whole_word_only():
    log = _log(pm_turn("I registered the trade"), advisor_turn("noted"))

    assert check_grep(log, PARAMS) == ()


def test_reasons_sorted_unique():
    log = _log(
        pm_turn("my loss_aversion_lambda again"),
        advisor_turn("noted"),
        pm_turn("I like a wide register"),
        advisor_turn("noted again"),
        pm_turn("loss aversion lambda repeated"),
        advisor_turn("noted a third time"),
    )

    assert check_grep(log, PARAMS) == (
        "names a parameter: loss aversion",
        "names a parameter: loss_aversion_lambda",
        "names a parameter: register",
    )


def test_leak_param_names_is_biases_then_catalogue_preferences():
    catalogue = load_catalogue()
    params = leak_param_names(catalogue)

    assert params.index("register") > params.index(BIAS_PARAMS[-1])
