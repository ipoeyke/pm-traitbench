"""Shared test fixtures."""

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import Catalogue


@pytest.fixture(scope="session")
def catalogue() -> Catalogue:
    """The catalogue shipped inside the package, frozen for the whole run."""
    return load_catalogue()
