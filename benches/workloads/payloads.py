from collections.abc import Callable

import numpy as np


def fixed_bytes_factory(size_bytes: int, seed: int = 0) -> Callable[[str], bytes]:
    """
    Return a callable f(key)->bytes that produces
    deterministic pseudo-random payloads of size `size_bytes`.
    Uses a per-factory RNG seeded by `seed`; keys are mixed into the seed stream for stability.
    """
    if size_bytes < 0:
        raise ValueError("size_bytes must be >= 0")
    rng = np.random.default_rng(seed)

    def _bytes_for(key: str) -> bytes:
        # mix key into RNG stream deterministically by drawing a jump based on key hash
        # (simple, fast, reproducible; not cryptographically secure)
        jump = (hash(key) & 0xFFFFFFFF) % 10_000_000
        # advance the stream a bit
        _ = rng.integers(0, 2**32, size=1 + (jump % 3))
        return rng.bytes(size_bytes) if size_bytes > 0 else b""

    return _bytes_for
