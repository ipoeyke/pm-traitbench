"""Tests for the stance bank: models, the loader's consistency checks and rendering."""

import pytest

from pm_traitbench.catalogues.loader import (
    REVEALED_PATTERNS,
    STANCE_SLOTS,
    check_catalogue,
    check_stances,
    render_stance,
)
from pm_traitbench.catalogues.models import (
    BiasStances,
    Catalogue,
    PreferenceGroup,
    PreferenceStances,
    StanceLines,
    Stances,
)
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import AssetClass, Kind, StanceEntry
from pm_traitbench.errors import CatalogueError, PlanError

_N_PREFERENCES_MAX = 8

# Which parts of a planted event a stance line for a given (kind of trait, kind of
# evidence) pair may quote.
_EXPECTED_STANCE_SLOTS: dict[tuple[Kind, StanceEntry], frozenset[str]] = {
    (Kind.BIAS, StanceEntry.REVEALED): frozenset(
        {"instrument", "entry", "target", "stop", "round_level"}
    ),
    (Kind.BIAS, StanceEntry.STATED): frozenset(),
    (Kind.BIAS, StanceEntry.CLAIM): frozenset(),
    (Kind.BIAS, StanceEntry.RETRACT): frozenset(),
    (Kind.BIAS, StanceEntry.THIRD_PARTY): frozenset({"who"}),
    (Kind.BIAS, StanceEntry.DRIFT_UPDATE): frozenset(),
    (Kind.BIAS, StanceEntry.DRIFT_DORMANT): frozenset(),
    (Kind.BIAS, StanceEntry.DRIFT_REVIVE): frozenset(),
    (Kind.PREFERENCE, StanceEntry.STATED): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.REVEALED): frozenset({"value", "instrument"}),
    (Kind.PREFERENCE, StanceEntry.REVEALED_REACTION): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.VIOLATION): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.RETRACT): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.THIRD_PARTY): frozenset({"value", "who"}),
    (Kind.PREFERENCE, StanceEntry.DRIFT_UPDATE): frozenset({"value", "old_value"}),
}


def _check(catalogue: Catalogue) -> None:
    check_catalogue(catalogue, list(AssetClass), n_preferences_max=_N_PREFERENCES_MAX)


def _lines(*values: str) -> StanceLines:
    return {"all": tuple(values)}


def _revealed_lines_for(patterns: tuple[str, ...]) -> dict[str, StanceLines]:
    """A minimal, generic "all" bank for each of a bias's engine action patterns."""
    return {
        pattern: _lines(
            "act on {instrument} near {entry}",
            "act on {instrument} at {target}",
        )
        for pattern in patterns
    }


def _bias_stances(param: str) -> BiasStances:
    return BiasStances(
        revealed=_revealed_lines_for(REVEALED_PATTERNS[param]),
        stated=_lines("say something about the habit", "tell the advisor about the habit"),
        claim=_lines("claim the opposite habit outright", "tell the advisor it never happens"),
        retract=_lines(
            "say the habit is true, then correct yourself: it is not",
            "state the habit, then take it back a moment later",
        ),
        third_party=_lines(
            "tell the advisor {who} shows the habit",
            "say {who} does this every time",
        ),
        drift_update=_lines(
            "tell the advisor you have been working on it",
            "say you have started to change the habit",
        ),
        drift_dormant=_lines(
            "tell the advisor you have stopped doing it",
            "say the habit has faded away",
        ),
        drift_revive=_lines(
            "tell the advisor it is back again",
            "say you have slipped back into the habit",
        ),
    )


def _pref_stances(revealed: StanceLines | None = None) -> PreferenceStances:
    return PreferenceStances(
        stated=_lines("tell the advisor: {value}", "say plainly: {value}"),
        revealed_reaction=_lines("push back when a reply drifts from {value}", "flag {value}"),
        violation=_lines(
            "answer anyway even though the PM prefers {value}",
            "ignore the rule even though the PM prefers {value}",
        ),
        retract=_lines(
            "state {value} as your rule, then correct yourself",
            "say you want {value}, then take it back",
        ),
        third_party=_lines(
            "tell the advisor {who} wants: {value}",
            "say {who}'s rule is: {value}",
        ),
        drift_update=_lines(
            "tell the advisor it changed from {old_value} to {value}",
            "say you now want {value} instead of {old_value}",
        ),
        revealed=revealed if revealed is not None else {},
    )


def _minimal_stances() -> Stances:
    biases = {param: _bias_stances(param) for param in BIAS_PARAMS}
    preferences: dict[PreferenceGroup, PreferenceStances] = {}
    for group in PreferenceGroup:
        if group == PreferenceGroup.EXPRESSION:
            preferences[group] = _pref_stances(
                revealed=_lines(
                    "build {instrument} the way your rule sets: {value}",
                    "shape {instrument} to match: {value}",
                )
            )
        else:
            preferences[group] = _pref_stances()
    return Stances(biases=biases, preferences=preferences)


def _catalogue_with(catalogue: Catalogue, stances: Stances) -> Catalogue:
    return catalogue.model_copy(update={"stances": stances})


# --- packaged bank ---


def test_packaged_stance_bank_loads_and_passes_checks(catalogue: Catalogue) -> None:
    check_stances(catalogue)


def test_check_catalogue_raises_for_a_bad_stance_bank(catalogue: Catalogue) -> None:
    # Proves check_catalogue actually calls check_stances: nothing else it runs would
    # notice a stance bank missing a whole bias key, so removing that call breaks this.
    stances = _minimal_stances()
    biases = dict(stances.biases)
    del biases["exit_deficiency"]
    stances = stances.model_copy(update={"biases": biases})
    with pytest.raises(CatalogueError, match="exit_deficiency"):
        _check(_catalogue_with(catalogue, stances))


# --- Stances.lines ---


def test_lines_returns_all_lines_when_no_asset_class_key_present() -> None:
    stances = _minimal_stances()
    assert stances.lines("loss_aversion_lambda", StanceEntry.STATED, AssetClass.EQUITIES) == (
        "say something about the habit",
        "tell the advisor about the habit",
    )


def test_lines_prefers_asset_class_lines_over_all() -> None:
    stances = _minimal_stances()
    bank = stances.biases["loss_aversion_lambda"]
    overridden = bank.model_copy(
        update={
            "stated": {
                "all": ("generic line one", "generic line two"),
                "equities": ("equities line one", "equities line two"),
            }
        }
    )
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "loss_aversion_lambda": overridden}}
    )
    assert stances.lines("loss_aversion_lambda", StanceEntry.STATED, AssetClass.EQUITIES) == (
        "equities line one",
        "equities line two",
    )
    assert stances.lines("loss_aversion_lambda", StanceEntry.STATED, AssetClass.COMMODITIES) == (
        "generic line one",
        "generic line two",
    )


def test_lines_raises_planerror_naming_key_and_entry_for_unknown_key() -> None:
    stances = _minimal_stances()
    with pytest.raises(PlanError, match="not_a_real_key.*stated"):
        stances.lines("not_a_real_key", StanceEntry.STATED, AssetClass.EQUITIES)


def test_lines_raises_planerror_for_entry_the_model_has_no_attribute_for() -> None:
    stances = _minimal_stances()
    with pytest.raises(PlanError, match="violation"):
        stances.lines("loss_aversion_lambda", StanceEntry.VIOLATION, AssetClass.EQUITIES)


def test_lines_raises_planerror_for_empty_stance_lines() -> None:
    stances = _minimal_stances()
    with pytest.raises(PlanError, match="revealed"):
        stances.lines(
            PreferenceGroup.COMMUNICATION.value, StanceEntry.REVEALED, AssetClass.EQUITIES
        )


def test_lines_resolves_preference_group_by_string_value() -> None:
    stances = _minimal_stances()
    result = stances.lines(
        PreferenceGroup.EXPRESSION.value, StanceEntry.REVEALED, AssetClass.EQUITIES
    )
    assert len(result) >= 2


def test_lines_on_a_bias_revealed_entry_tells_caller_to_use_revealed_lines() -> None:
    stances = _minimal_stances()
    with pytest.raises(PlanError, match="revealed_lines"):
        stances.lines("loss_aversion_lambda", StanceEntry.REVEALED, AssetClass.EQUITIES)


# --- Stances.revealed_lines ---


def test_revealed_lines_returns_lines_for_a_known_pattern() -> None:
    stances = _minimal_stances()
    result = stances.revealed_lines("loss_aversion_lambda", "hold", AssetClass.EQUITIES)
    assert len(result) >= 2


def test_revealed_lines_prefers_asset_class_lines_over_all() -> None:
    stances = _minimal_stances()
    bank = stances.biases["loss_aversion_lambda"]
    overridden_pattern = {
        "all": ("generic hold line one", "generic hold line two"),
        "equities": ("equities hold line one", "equities hold line two"),
    }
    overridden = bank.model_copy(update={"revealed": {**bank.revealed, "hold": overridden_pattern}})
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "loss_aversion_lambda": overridden}}
    )
    assert stances.revealed_lines("loss_aversion_lambda", "hold", AssetClass.EQUITIES) == (
        "equities hold line one",
        "equities hold line two",
    )
    assert stances.revealed_lines("loss_aversion_lambda", "hold", AssetClass.COMMODITIES) == (
        "generic hold line one",
        "generic hold line two",
    )


def test_revealed_lines_raises_planerror_for_unknown_param() -> None:
    stances = _minimal_stances()
    with pytest.raises(PlanError, match="not_a_real_param"):
        stances.revealed_lines("not_a_real_param", "hold", AssetClass.EQUITIES)


def test_revealed_lines_raises_planerror_for_unknown_pattern() -> None:
    stances = _minimal_stances()
    with pytest.raises(PlanError, match="loss_aversion_lambda.*bogus_pattern"):
        stances.revealed_lines("loss_aversion_lambda", "bogus_pattern", AssetClass.EQUITIES)


# --- check_stances: one test per rule ---


def test_check_stances_missing_bias_key_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    biases = dict(stances.biases)
    del biases["exit_deficiency"]
    stances = stances.model_copy(update={"biases": biases})
    with pytest.raises(CatalogueError, match="missing.*exit_deficiency"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_extra_bias_key_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    biases = dict(stances.biases)
    biases["not_a_bias_param"] = biases["exit_deficiency"]
    stances = stances.model_copy(update={"biases": biases})
    with pytest.raises(CatalogueError, match="extra.*not_a_bias_param"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_missing_preference_group_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    preferences = dict(stances.preferences)
    del preferences[PreferenceGroup.WORKFLOW]
    stances = stances.model_copy(update={"preferences": preferences})
    with pytest.raises(CatalogueError, match="workflow"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_missing_revealed_pattern_key_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.biases["loss_aversion_lambda"]
    revealed = dict(bank.revealed)
    del revealed["hold"]
    overridden = bank.model_copy(update={"revealed": revealed})
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "loss_aversion_lambda": overridden}}
    )
    with pytest.raises(CatalogueError, match="missing.*hold"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_extra_revealed_pattern_key_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.biases["loss_aversion_lambda"]
    revealed = dict(bank.revealed)
    revealed["not_a_real_pattern"] = revealed["hold"]
    overridden = bank.model_copy(update={"revealed": revealed})
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "loss_aversion_lambda": overridden}}
    )
    with pytest.raises(CatalogueError, match="extra.*not_a_real_pattern"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_entry_without_all_key_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.biases["exit_deficiency"]
    overridden = bank.model_copy(
        update={"stated": {"equities": ("only an equities line", "and another")}}
    )
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "exit_deficiency": overridden}}
    )
    with pytest.raises(CatalogueError, match="no 'all' key"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_unknown_key_in_stance_lines_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.biases["exit_deficiency"]
    overridden = bank.model_copy(
        update={
            "stated": {
                "all": ("a stated line", "another stated line"),
                "bogus_key": ("a stray line", "one more"),
            }
        }
    )
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "exit_deficiency": overridden}}
    )
    with pytest.raises(CatalogueError, match="bogus_key"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_fewer_than_2_lines_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.biases["exit_deficiency"]
    overridden = bank.model_copy(update={"stated": {"all": ("only one line here",)}})
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "exit_deficiency": overridden}}
    )
    with pytest.raises(CatalogueError, match="fewer than 2"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_revealed_non_empty_on_non_expression_group_raises(
    catalogue: Catalogue,
) -> None:
    stances = _minimal_stances()
    bank = stances.preferences[PreferenceGroup.COMMUNICATION]
    overridden = bank.model_copy(
        update={"revealed": {"all": ("build it as {value}", "shape it as {value}")}}
    )
    stances = stances.model_copy(
        update={"preferences": {**stances.preferences, PreferenceGroup.COMMUNICATION: overridden}}
    )
    with pytest.raises(CatalogueError, match="communication.*revealed.*must be empty"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_expression_revealed_empty_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.preferences[PreferenceGroup.EXPRESSION]
    overridden = bank.model_copy(update={"revealed": {}})
    stances = stances.model_copy(
        update={"preferences": {**stances.preferences, PreferenceGroup.EXPRESSION: overridden}}
    )
    with pytest.raises(CatalogueError, match="expression.*revealed.*must not be empty"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_unknown_slot_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.biases["exit_deficiency"]
    overridden = bank.model_copy(
        update={"stated": {"all": ("say something {bogus_slot}", "say another thing entirely")}}
    )
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "exit_deficiency": overridden}}
    )
    with pytest.raises(CatalogueError, match="bogus_slot"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_preference_line_missing_value_slot_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.preferences[PreferenceGroup.COMMUNICATION]
    overridden = bank.model_copy(
        update={"stated": {"all": ("say this has no slot at all", "say plainly: {value}")}}
    )
    stances = stances.model_copy(
        update={"preferences": {**stances.preferences, PreferenceGroup.COMMUNICATION: overridden}}
    )
    with pytest.raises(CatalogueError, match=r"does not use the \{value\} slot"):
        check_stances(_catalogue_with(catalogue, stances))


def test_check_stances_banned_word_raises(catalogue: Catalogue) -> None:
    stances = _minimal_stances()
    bank = stances.biases["exit_deficiency"]
    overridden = bank.model_copy(
        update={"stated": {"all": ("say the herding shows up here", "say another thing entirely")}}
    )
    stances = stances.model_copy(
        update={"biases": {**stances.biases, "exit_deficiency": overridden}}
    )
    with pytest.raises(CatalogueError, match="banned word"):
        check_stances(_catalogue_with(catalogue, stances))


def test_stance_slots_matches_the_specified_table() -> None:
    assert STANCE_SLOTS == _EXPECTED_STANCE_SLOTS
    assert len(STANCE_SLOTS) == 15


def test_revealed_patterns_covers_every_bias_param() -> None:
    assert set(REVEALED_PATTERNS) == set(BIAS_PARAMS)
    for patterns in REVEALED_PATTERNS.values():
        assert len(patterns) >= 1


# --- render_stance ---


def test_render_stance_fills_slots() -> None:
    slots = {"instrument": "Equity 0001", "entry": "21.84"}
    assert render_stance("add to {instrument} at {entry}", slots) == "add to Equity 0001 at 21.84"


def test_render_stance_raises_catalogue_error_on_missing_slot() -> None:
    with pytest.raises(CatalogueError, match="instrument"):
        render_stance("add to {instrument}", {})


def test_render_stance_raises_catalogue_error_on_positional_placeholder() -> None:
    with pytest.raises(CatalogueError, match="add to"):
        render_stance("add to {}", {"instrument": "x"})


def test_render_stance_raises_catalogue_error_on_malformed_line() -> None:
    with pytest.raises(CatalogueError, match="add to"):
        render_stance("add to {instrument!r", {"instrument": "x"})
