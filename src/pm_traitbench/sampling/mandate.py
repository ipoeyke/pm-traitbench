"""Mandate sampling: draws each PM's sub-style, risk unit, benchmark and book size."""

from math import exp, log

from numpy.random import Generator

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.sampling.picks import pick_index
from pm_traitbench.sampling.population import PmSlot
from pm_traitbench.tables.schema import Mandate

_BOOK_SIZE_ROUND_TO = 1e6


def sample_mandate(slot: PmSlot, config: Config, catalogue: Catalogue, rng: Generator) -> Mandate:
    """Draw a mandate for a PM slot: a uniform sub-style and a log-uniform book size."""
    sub_styles = catalogue.sub_styles[slot.asset_class]
    sub_style = sub_styles[pick_index(rng, len(sub_styles), "sub styles")]

    book_min = config.mandate.book_size_min
    book_max = config.mandate.book_size_max
    raw_book_size = exp(rng.uniform(log(book_min), log(book_max)))
    snapped = round(raw_book_size / _BOOK_SIZE_ROUND_TO) * _BOOK_SIZE_ROUND_TO
    book_size = round(min(max(snapped, book_min), book_max), 4)

    return Mandate(
        asset_class=slot.asset_class,
        sub_style=sub_style.name,
        book_size=book_size,
        risk_unit=sub_style.risk_unit,
        benchmark=sub_style.benchmark,
    )
