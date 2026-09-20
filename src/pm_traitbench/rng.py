"""Keyed random streams: deterministic, stateless child generators.

Keying on a purpose-specific tuple means adding one PM or one extra draw
never perturbs any other sampled value. Python's `hash()` is salted per
process, so string keys are hashed with sha256 instead.
"""

import hashlib

import numpy as np


def _key_to_int(key: str | int) -> int:
    if isinstance(key, int):
        if key < 0:
            raise ValueError("integer keys must be non-negative")
        return key
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def stream(root_seed: int, *keys: str | int) -> np.random.Generator:
    """Return a fresh, independent generator for the given root seed and keys."""
    ss = np.random.SeedSequence(entropy=root_seed, spawn_key=tuple(_key_to_int(k) for k in keys))
    return np.random.default_rng(ss)
