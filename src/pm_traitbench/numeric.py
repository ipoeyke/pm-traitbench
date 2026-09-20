"""Strict numeric-text parsing shared by table reading and catalogue checks."""

import math
import re

_NUMERIC_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def parse_number(text: str) -> float | None:
    """Parse plain decimal or scientific notation strictly; None otherwise.

    Rejects underscores, surrounding whitespace, and spellings of nan/inf
    (none of these match the pattern), plus any parse that overflows to a
    non-finite float (e.g. "1e999").
    """
    if not _NUMERIC_PATTERN.match(text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None
