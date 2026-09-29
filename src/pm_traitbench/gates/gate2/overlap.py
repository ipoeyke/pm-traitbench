"""Gate 2 cross-PM overlap: how much of one PM's transcript language reappears in another's.

A high containment flags PMs whose transcripts are too similar to give a
judge independent evidence for each PM's recovered biases.
"""

import math
import re
from collections.abc import Mapping, Sequence

_TOKEN_SPLIT = re.compile(r"[^a-z]+")


def ngrams(text: str, n: int) -> frozenset[tuple[str, ...]]:
    """The set of consecutive `n`-token windows in `text`; empty when fewer than `n` tokens."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    tokens = [token for token in _TOKEN_SPLIT.split(text.lower()) if token]
    return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def containment_by_pm(texts_by_pm: Mapping[str, str], n: int) -> dict[str, float]:
    """Each PM's largest n-gram containment in any other PM's text.

    Measured against the single nearest other PM, not the union of the
    rest, so a PM is not flagged as derivative merely because several
    peers each cover a different part of its language.
    """
    grams = {pm: ngrams(text, n) for pm, text in texts_by_pm.items()}
    result: dict[str, float] = {}
    for pm in sorted(grams):
        own = grams[pm]
        others = [grams[other] for other in grams if other != pm]
        if not own or not others:
            result[pm] = 0.0
            continue
        result[pm] = max(len(own & other) / len(own) for other in others)
    return result


def summarise(values: Sequence[float]) -> dict[str, float]:
    """Median, 90th percentile by nearest rank, and max of `values`; all 0.0 when empty."""
    if not values:
        return {"median": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    median = ordered[mid] if count % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    p90 = ordered[math.ceil(0.9 * count) - 1]
    return {"median": median, "p90": p90, "max": ordered[-1]}
