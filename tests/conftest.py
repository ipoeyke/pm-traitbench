"""Shared test fixtures."""

from pathlib import Path

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import Catalogue

_FIXTURE_CATALOGUE_DIR = Path(__file__).parent / "fixtures" / "catalogue"


@pytest.fixture(scope="session")
def fixture_catalogue() -> Catalogue:
    """The compact catalogue under tests/fixtures/catalogue, frozen for the whole run."""
    return load_catalogue(_FIXTURE_CATALOGUE_DIR)
