# benches/workloads/generators.py
from collections.abc import Generator, Iterable

import numpy as np

__all__ = [
    "zipf_stream",
    "uniform_stream",
    "prefixed",
    "interleave_stream",
]


def zipf_stream(
    *, total_keys: int, unique_ratio: float, theta: float, count: int, seed: int = 0
) -> Generator[str, None, None]:
    """
    Emit `count` keys following Zipf(theta) over a hot set of size H=floor(total_keys*unique_ratio).
    Keys: "k{idx}" where idx ∈ [0, H-1]. Deterministic via seed.
    """
    if total_keys <= 0:
        raise ValueError("total_keys must be positive")
    if not (0 < unique_ratio <= 1):
        raise ValueError("unique_ratio must be in (0,1]")
    if theta <= 1.0:
        raise ValueError("theta must be > 1.0")
    if count < 0:
        raise ValueError("count must be >= 0")

    hot = max(1, int(total_keys * unique_ratio))
    rng = np.random.default_rng(seed)
    emitted = 0
    while emitted < count:
        k = int(rng.zipf(theta))  # 1..∞
        idx = (k - 1) % hot  # wrap into [0..hot-1]
        yield f"k{idx}"
        emitted += 1


def uniform_stream(
    *, total_keys: int, unique_ratio: float, count: int, seed: int = 0
) -> Generator[str, None, None]:
    """
    Emit `count` keys uniformly over domain D=floor(total_keys*unique_ratio).
    Keys: "k{idx}" where idx ∈ [0, D-1]. Deterministic via seed.
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


def prefixed(stream: Iterable[str], prefix: str) -> Generator[str, None, None]:
    """Prefix each key (e.g., 'S:' or 'N:') to avoid collisions between workloads."""
    p = prefix if prefix.endswith(":") else prefix + ":"
    for k in stream:
        yield p + k


def interleave_stream(
    a: Iterable[str], b: Iterable[str], *, ratio_a: float = 0.7, seed: int = 12345
) -> Generator[str, None, None]:
    """
    Interleave two streams at approximately ratio_a : (1-ratio_a).
    Drains whichever stream remains after the other is exhausted.
    """
    if not (0.0 < ratio_a < 1.0):
        raise ValueError("ratio_a must be in (0,1)")
    ita, itb = iter(a), iter(b)
    rng = np.random.default_rng(seed)
    a_prob = ratio_a
    while True:
        pick_a = rng.random() < a_prob
        try:
            yield next(ita if pick_a else itb)
        except StopIteration:
            # drain the other
            other = itb if pick_a else ita
            try:
                while True:
                    yield next(other)
            except StopIteration:
                break
