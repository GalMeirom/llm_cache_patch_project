from collections.abc import Generator

import numpy as np


def zipf_stream(
    *, total_keys: int, unique_ratio: float, theta: float, count: int, seed: int = 0
) -> Generator[str, None, None]:
    """
    Yield `count` keys following a Zipf(θ) over
    a hot-set of size H = max(1, floor(total_keys * unique_ratio)).
    Keys are "k{idx}", idx in [0, H-1]. Reproducible via `seed`.
    """
    if total_keys <= 0:
        raise ValueError("total_keys must be positive")
    if not (0 < unique_ratio <= 1):
        raise ValueError("unique_ratio must be in (0,1]")
    if theta <= 1.0:
        # Zipf requires theta > 1 for a proper heavy-tail; allow >1.0 only
        raise ValueError("theta must be > 1.0")
    if count < 0:
        raise ValueError("count must be >= 0")

    hot = max(1, int(total_keys * unique_ratio))
    rng = np.random.default_rng(seed)
    emitted = 0
    while emitted < count:
        # rng.zipf(theta) returns k in {1,2,...}; map to [0..hot-1] with modulo for simplicity
        k = int(rng.zipf(theta))
        idx = (k - 1) % hot
        yield f"k{idx}"
        emitted += 1


def uniform_stream(
    *, total_keys: int, unique_ratio: float, count: int, seed: int = 0
) -> Generator[str, None, None]:
    """
    Yield `count` keys uniformly over a domain of size D = max(1, floor(total_keys * unique_ratio)).
    Keys are "k{idx}", idx in [0, D-1]. Reproducible via `seed`.
    """
    if total_keys <= 0:
        raise ValueError("total_keys must be positive")
    if not (0 < unique_ratio <= 1):
        raise ValueError("unique_ratio must be in (0,1]")
    if count < 0:
        raise ValueError("count must be >= 0")

    dom = max(1, int(total_keys * unique_ratio))
    rng = np.random.default_rng(seed)
    for _ in range(count):
        idx = int(rng.integers(0, dom))
        yield f"k{idx}"
