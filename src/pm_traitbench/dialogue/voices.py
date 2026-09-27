"""Per-PM voice draw, independent of every trait, bias and preference."""

from collections.abc import Sequence

from pm_traitbench.catalogues.models import Voice
from pm_traitbench.rng import stream


def draw_voice(root: int, pm_id: str, voices: Sequence[Voice]) -> Voice:
    """Draw one voice uniformly at random for a PM, in catalogue order.

    Keyed only on the root seed and PM id, so a voice can never carry a signal
    for any trait, bias or preference the PM happens to have.
    """
    if not voices:
        raise ValueError("voices must not be empty")
    rng = stream(root, "dialogue", pm_id, "voice")
    index = int(rng.integers(len(voices)))
    return voices[index]
