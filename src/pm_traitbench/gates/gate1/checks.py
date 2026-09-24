"""Cross-checks gate 1's recomputed opportunity counts against the engine's own counters.

Each estimator's opportunity set is meant to reproduce, from public and hidden
rows alike, a count the behaviour engine already tracked while it produced
those rows. Agreement is a check that the rebuild is faithful, not a
tolerance: any mismatch means gate 1 is reading the wrong rows for that PM.
"""

from pm_traitbench.errors import Gate1Error
from pm_traitbench.gates.gate1.estimators import disposition, extrapolation, loss_aversion
from pm_traitbench.gates.gate1.inputs import PmInputs


def check_counts(inputs: PmInputs) -> None:
    """Raise `Gate1Error` naming the PM, count and both numbers on the first mismatch."""
    days = frozenset(inputs.view.dates)
    checks = (
        ("loss_side_untriggered_days", len(loss_aversion.opportunities(inputs, days))),
        ("entries_after_run", extrapolation.after_run_count(inputs, days)),
        ("ideas", len(inputs.ideas)),
        ("sell_day_position_days", len(disposition.sell_day_rows(inputs, days))),
        ("conflict_entries", sum(1 for idea in inputs.ideas if idea.conflict)),
    )
    for name, recomputed in checks:
        engine_count = inputs.engine_counts[name]
        if recomputed != engine_count:
            raise Gate1Error(
                f"PM '{inputs.pm_id}': count '{name}' mismatch, "
                f"gate 1 recomputed {recomputed} but the engine counted {engine_count}"
            )
