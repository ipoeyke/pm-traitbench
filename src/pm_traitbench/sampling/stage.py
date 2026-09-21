"""Ties every per-PM sampler into one deterministic pass over the population.

Draws follow a fixed order per PM, each on its own stream keyed by
``stream(config.seed.root, "pm", slot.index, purpose)``: mandate, biases,
preferences, rules, trait assembly, self-description, then drift.

The order is fixed, but not every step depends on the one before it. Rules
need the mandate. Preferences need the asset class from the population slot,
not the mandate. The self-description needs the two strongest biases. Drift
needs the final trait list, and only runs for PMs marked for drift.
"""

from dataclasses import dataclass

from numpy.random import Generator

from pm_traitbench.catalogues.loader import check_catalogue, load_catalogue
from pm_traitbench.catalogues.models import Catalogue, PreferenceGroup
from pm_traitbench.config import Config
from pm_traitbench.errors import ConfigError
from pm_traitbench.rng import stream
from pm_traitbench.sampling.biases import sample_biases
from pm_traitbench.sampling.drift import sample_drift
from pm_traitbench.sampling.mandate import sample_mandate
from pm_traitbench.sampling.population import PmSlot, build_population
from pm_traitbench.sampling.preferences import sample_preferences
from pm_traitbench.sampling.profile import sample_self_description
from pm_traitbench.sampling.rules import sample_rules
from pm_traitbench.sampling.traits import build_traits
from pm_traitbench.stages import Stage
from pm_traitbench.tables.schema import DriftEvent, Persona, Rule, StatedProfile, Trait
from pm_traitbench.tables.specs import DRIFT_EVENTS, PERSONAS, RULES, TRAITS
from pm_traitbench.tables.store import DataStore


@dataclass(frozen=True)
class SampleResult:
    """The four tables produced by one sampling pass over the population."""

    personas: list[Persona]
    traits: list[Trait]
    rules: list[Rule]
    drift_events: list[DriftEvent]


def _pm_stream(config: Config, slot: PmSlot, *purpose: str) -> Generator:
    return stream(config.seed.root, "pm", slot.index, *purpose)


def check_sampling_config(config: Config, catalogue: Catalogue) -> None:
    """Check config values that a sampler would otherwise accept and then fail on.

    These are cross-cutting invariants between config and catalogue size that
    pydantic's per-field validation cannot see.
    """
    n_groups = len(PreferenceGroup)
    if config.preferences.n_min < n_groups:
        raise ConfigError(
            f"preferences.n_min must be at least {n_groups} (one preference is always drawn "
            f"per group), got {config.preferences.n_min}"
        )

    if config.biases.min_active < 2:
        raise ConfigError(
            f"biases.min_active must be at least 2 (the self-description needs two active "
            f"biases), got {config.biases.min_active}"
        )

    for asset_class in config.population.asset_classes:
        entries = catalogue.rules.entries_for(asset_class)
        n_mandatory = sum(1 for entry in entries if entry.mandatory)
        mandatory_has_discipline = any(entry.mandatory and entry.discipline for entry in entries)
        min_required_max = n_mandatory + (0 if mandatory_has_discipline else 1)
        if config.rules.n_self_rules_max < min_required_max:
            raise ConfigError(
                f"rules.n_self_rules_max must be at least {min_required_max} for asset class "
                f"'{asset_class}' (the mandatory rule entries, plus a discipline rule when none "
                f"of them is one), got {config.rules.n_self_rules_max}"
            )

        if config.rules.n_self_rules_min > len(entries):
            raise ConfigError(
                f"rules.n_self_rules_min must not exceed the number of applicable rule "
                f"entries for asset class '{asset_class}' ({len(entries)}), got "
                f"{config.rules.n_self_rules_min}"
            )


def sample_all(config: Config, catalogue: Catalogue) -> SampleResult:
    """Sample personas, traits, rules and drift events for the whole population."""
    check_sampling_config(config, catalogue)
    timeline = config.timeline()
    personas: list[Persona] = []
    traits: list[Trait] = []
    rules: list[Rule] = []
    drift_events: list[DriftEvent] = []

    for slot in build_population(config):
        mandate = sample_mandate(slot, config, catalogue, _pm_stream(config, slot, "mandate"))
        biases = sample_biases(
            config,
            _pm_stream(config, slot, "activation"),
            _pm_stream(config, slot, "biases"),
            _pm_stream(config, slot, "regime"),
        )
        preferences = sample_preferences(
            slot.asset_class, config, catalogue, _pm_stream(config, slot, "preferences")
        )
        pm_rules = sample_rules(
            slot.pm_id, mandate, config, catalogue, _pm_stream(config, slot, "rules")
        )
        pm_traits = build_traits(slot.pm_id, biases, preferences)
        self_description = sample_self_description(
            biases, slot.typicality, catalogue, _pm_stream(config, slot, "profile")
        )

        personas.append(
            Persona(
                pm_id=slot.pm_id,
                market_seed=slot.market_seed,
                split=slot.split,
                mandate=mandate,
                stated_profile=StatedProfile(self_description=self_description),
                typicality=slot.typicality,
            )
        )
        traits.extend(pm_traits)
        rules.extend(pm_rules)

        if slot.drift:
            drift_events.extend(
                sample_drift(
                    slot.pm_id,
                    pm_traits,
                    config,
                    catalogue,
                    timeline,
                    _pm_stream(config, slot, "drift"),
                )
            )

    return SampleResult(personas=personas, traits=traits, rules=rules, drift_events=drift_events)


def run(config: Config, store: DataStore) -> None:
    """Load and check the catalogue, sample the population, then write the four tables."""
    catalogue = load_catalogue()
    check_catalogue(catalogue, config.population.asset_classes, config.preferences.n_max)
    check_sampling_config(config, catalogue)
    result = sample_all(config, catalogue)
    store.write(PERSONAS, result.personas)
    store.write(TRAITS, result.traits)
    store.write(RULES, result.rules)
    store.write(DRIFT_EVENTS, result.drift_events)


SAMPLE_STAGE = Stage(
    number=1,
    name="sample",
    help="sample personas, traits, rules and drift events",
    run=run,
    reads=(),
    writes=(PERSONAS, TRAITS, RULES, DRIFT_EVENTS),
)
