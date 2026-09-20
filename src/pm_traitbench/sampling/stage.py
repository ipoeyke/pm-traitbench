"""Ties every per-PM sampler into one deterministic pass over the population.

Draws follow a fixed order per PM, each on its own stream keyed by
``stream(config.seed.root, "pm", slot.index, purpose)``: mandate first, since
rules and preferences need it; then biases, preferences and rules; trait
assembly; self-description, which needs the PM's two strongest biases; and
drift last, since it needs the final trait list and only runs for PMs
marked for drift.
"""

from dataclasses import dataclass

from numpy.random import Generator

from pm_traitbench.catalogues.loader import check_catalogue, load_catalogue
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
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


def sample_all(config: Config, catalogue: Catalogue) -> SampleResult:
    """Sample personas, traits, rules and drift events for the whole population."""
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
