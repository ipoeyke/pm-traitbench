"""Self-description sampling: a PM's one-line statement about how they invest."""

from numpy.random import Generator

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import Typicality
from pm_traitbench.errors import SamplingError
from pm_traitbench.sampling.biases import BiasDraw
from pm_traitbench.sampling.picks import pick_index


def sample_self_description(
    biases: list[BiasDraw], typicality: Typicality, catalogue: Catalogue, rng: Generator
) -> str:
    """Sample a self-description from the PM's two strongest active biases.

    A typical PM's description agrees with each bias; an anti-typical PM's
    contradicts it. Ties in strength break by BIAS_PARAMS order.
    """
    active = [bias for bias in biases if bias.active]
    if len(active) < 2:
        raise SamplingError(f"self-description needs at least 2 active biases, got {len(active)}")

    ranked = sorted(active, key=lambda bias: (-bias.strength, BIAS_PARAMS.index(bias.param)))
    top_two = ranked[:2]

    phrases = []
    for bias in top_two:
        phrasings = catalogue.self_descriptions[bias.param]
        options = phrasings.agree if typicality == Typicality.TYPICAL else phrasings.contradict
        phrases.append(options[pick_index(rng, len(options), f"phrasings for '{bias.param}'")])

    return ", ".join(phrases)
