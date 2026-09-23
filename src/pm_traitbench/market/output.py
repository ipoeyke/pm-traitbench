"""Per-family process output: the shared shape every market process returns."""

from dataclasses import dataclass, field

import numpy as np

from pm_traitbench.enums import Tenor


@dataclass
class ProcessOutput:
    """A family process's simulated series, merged across families by the generator."""

    prices: dict[str, np.ndarray] = field(default_factory=dict)
    spreads: dict[str, np.ndarray] = field(default_factory=dict)
    curves: dict[tuple[str, Tenor], np.ndarray] = field(default_factory=dict)

    def merge(self, other: "ProcessOutput") -> "ProcessOutput":
        """Combine two outputs; raise ValueError if either shares a key with the other."""
        merged = ProcessOutput()
        for name in ("prices", "spreads", "curves"):
            mine, theirs = getattr(self, name), getattr(other, name)
            repeated = set(mine) & set(theirs)
            if repeated:
                raise ValueError(f"duplicate {name} key(s): {sorted(repeated)}")
            getattr(merged, name).update(mine)
            getattr(merged, name).update(theirs)
        return merged
