import math
import statistics

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass, Split, Typicality
from pm_traitbench.rng import stream
from pm_traitbench.sampling.mandate import sample_mandate
from pm_traitbench.sampling.population import PmSlot


def _slot(asset_class: AssetClass) -> PmSlot:
    return PmSlot(
        index=1,
        pm_id="pm_001",
        split=Split.FULL,
        asset_class=asset_class,
        market_seed="A",
        typicality=Typicality.TYPICAL,
        drift=False,
    )


def test_sub_style_risk_unit_and_benchmark_match_the_catalogue(fixture_catalogue: Catalogue):
    config = Config()
    for asset_class in AssetClass:
        entries = {entry.name: entry for entry in fixture_catalogue.sub_styles[asset_class]}
        for i in range(10):
            rng = stream(1, "t", i, "mandate")
            mandate = sample_mandate(_slot(asset_class), config, fixture_catalogue, rng)
            assert mandate.asset_class == asset_class
            assert mandate.sub_style in entries
            entry = entries[mandate.sub_style]
            assert mandate.risk_unit == entry.risk_unit
            assert mandate.benchmark == entry.benchmark


def test_book_size_is_within_bounds_and_a_multiple_of_one_million(fixture_catalogue: Catalogue):
    config = Config()
    for i in range(50):
        rng = stream(1, "t", i, "mandate")
        mandate = sample_mandate(_slot(AssetClass.EQUITIES), config, fixture_catalogue, rng)
        assert config.mandate.book_size_min <= mandate.book_size <= config.mandate.book_size_max
        assert mandate.book_size % 1e6 == 0


def test_over_many_draws_both_sub_styles_appear_and_median_book_size_is_plausible(
    fixture_catalogue: Catalogue,
):
    config = Config()
    slot = _slot(AssetClass.EQUITIES)
    sub_styles_seen = set()
    book_sizes = []
    for i in range(2000):
        rng = stream(1, "t", i, "mandate")
        mandate = sample_mandate(slot, config, fixture_catalogue, rng)
        sub_styles_seen.add(mandate.sub_style)
        book_sizes.append(mandate.book_size)

    equities_sub_styles = fixture_catalogue.sub_styles[AssetClass.EQUITIES]
    expected_sub_styles = {entry.name for entry in equities_sub_styles}
    assert sub_styles_seen == expected_sub_styles

    expected_median = math.sqrt(config.mandate.book_size_min * config.mandate.book_size_max)
    median = statistics.median(book_sizes)
    assert expected_median * 0.75 <= median <= expected_median * 1.25
