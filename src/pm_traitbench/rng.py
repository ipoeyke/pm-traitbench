"""Keyed random streams: deterministic, stateless child generators.

Keying on a purpose-specific tuple means adding one PM or one extra draw
never perturbs any other sampled value. Python's `hash()` is salted per
process, so string keys are hashed with sha256 instead. Every key is encoded
as three fixed-width uint32 words (a type tag plus the high and low halves of
a 64-bit value) so distinct keys can never fold together in the spawn key.
"""

import hashlib

import numpy as np

_UINT64_MAX = 2**64 - 1


def _key_to_words(key: str | int) -> tuple[int, int, int]:
    """Encode one key as (tag, hi, lo): tag 0 for int, 1 for str."""
    if isinstance(key, bool):
        raise TypeError("stream keys must not be bool")
    if isinstance(key, int):
        if key < 0 or key > _UINT64_MAX:
            raise ValueError("integer keys must be non-negative and fit in 64 bits")
        tag, value = 0, key
    elif isinstance(key, str):
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        tag, value = 1, int.from_bytes(digest[:8], "big")
    else:
        raise TypeError(f"stream keys must be int or str, got {type(key).__name__}")
    return tag, (value >> 32) & 0xFFFFFFFF, value & 0xFFFFFFFF


def stream(root_seed: int, *keys: str | int) -> np.random.Generator:
    """Return a fresh, independent generator for the given root seed and keys."""
    words: list[int] = []
    for key in keys:
        words.extend(_key_to_words(key))
    ss = np.random.SeedSequence(entropy=root_seed, spawn_key=tuple(words))
    return np.random.default_rng(ss)
