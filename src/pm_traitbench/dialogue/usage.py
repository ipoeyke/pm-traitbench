"""Token usage read from a Messages API response."""

from collections.abc import Mapping
from typing import Any

from pm_traitbench.tables.schema import CallUsage

ZERO_USAGE = CallUsage(
    input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0
)


def usage_of(response: Mapping[str, Any]) -> CallUsage:
    """A response's token usage, counting an absent or null field as zero."""
    # `to_dict()` keeps explicit nulls for optional usage fields.
    usage = response.get("usage") or {}
    return CallUsage(
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_input_tokens=usage.get("cache_read_input_tokens") or 0,
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens") or 0,
    )
