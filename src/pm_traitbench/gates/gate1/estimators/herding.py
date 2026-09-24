"""Herding: the share of entries on the street's side, among entries where the street took one.

This cannot read the hidden `conflict`/`followed_street` columns: a PM who
followed the street when the street agreed with them leaves no trace of a
conflict, so the only observable signal is whether the entered side matches
the street's overweight/underweight view. The agreement rate this recovers
rises with the herding weight, and the neutral baseline (entries the PM would
have taken anyway) absorbs the rate a PM with no herding would show by chance.
"""

from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import Side, StreetView
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs

_STREET_SIDE: dict[StreetView, Side] = {
    StreetView.OVERWEIGHT: Side.BUY,
    StreetView.UNDERWEIGHT: Side.SELL,
}


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Share of entries in `days` whose side matches the street's non-neutral view.

    Entries against an instrument with no consensus row, or a neutral street
    view, are skipped: neither gives a side to agree or disagree with.
    """
    n = 0
    agree = 0
    for idea in inputs.ideas:
        if idea.entry_date not in days:
            continue
        t = inputs.day_index[idea.entry_date]
        street = inputs.view.street_view(idea.instrument_id, t)
        street_side = _STREET_SIDE.get(street)
        if street_side is None:
            continue
        n += 1
        if idea.side == street_side:
            agree += 1
    if n == 0:
        return Estimate(value=None, n=0)
    return Estimate(value=agree / n, n=n)


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
