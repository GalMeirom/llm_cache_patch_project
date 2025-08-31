# benches/workloads/payloads.py
from collections.abc import Callable, Iterable

import numpy as np

__all__ = [
    "text_sentences_factory",
    "utf8_sizeof",
    "fixed_bytes_factory",  # keep for fallback
]


def utf8_sizeof(s: str) -> int:
    """Return UTF-8 byte length (use for cache getsizeof)."""
    return len(s.encode("utf-8"))


def _local_rng(seed: int, key: str) -> np.random.Generator:
    # Derive a stable per-key seed (no external I/O; reproducible)
    h = hash((key, "payload")) & 0xFFFFFFFF
    return np.random.default_rng(seed ^ h)


def _word_from_id(i: int) -> str:
    # Compact, human-ish tokens; deterministic and fast.
    # Example: token-000123, token-987654, ...
    return f"token-{i:06d}"


def _gen_words(rng: np.random.Generator, vocab_size: int, n: int) -> Iterable[str]:
    # Sample without replacement if n <= vocab_size, else with replacement.
    if n <= vocab_size:
        ids = rng.choice(vocab_size, size=n, replace=False)
    else:
        ids = rng.integers(0, vocab_size, size=n)
    for i in ids:
        yield _word_from_id(int(i))


def text_sentences_factory(
    *,
    vocab_size: int = 10_000,
    sentences_range: tuple[int, int] = (1, 1),
    words_per_sentence_range: tuple[int, int] = (3, 6),
    seed: int = 0,
) -> Callable[[str], str]:
    """
    Build a deterministic factory: key -> multi-sentence prompt text.

    Params:
      vocab_size: size of synthetic vocabulary (token-000000 .. token-(vocab_size-1))
      sentences_range: (min_sentences, max_sentences)
      words_per_sentence_range: (min_words, max_words)
      seed: base seed to keep runs reproducible

    Returns:
      f(key) -> "Sentence One. sentence two. ..."
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    a_s, b_s = sentences_range
    a_w, b_w = words_per_sentence_range
    if not (1 <= a_s <= b_s):
        raise ValueError("invalid sentences_range")
    if not (1 <= a_w <= b_w):
        raise ValueError("invalid words_per_sentence_range")

    def _mk_sentence(rng: np.random.Generator, n_words: int) -> str:
        words = list(_gen_words(rng, vocab_size, n_words))
        if not words:
            return ""
        # Capitalize first token; end with simple punctuation.
        words[0] = words[0].capitalize()
        return " ".join(words) + "."

    def _factory(key: str) -> str:
        rng = _local_rng(seed, key)
        s_count = int(rng.integers(a_s, b_s + 1))
        pieces = []
        for _ in range(s_count):
            w_count = int(rng.integers(a_w, b_w + 1))
            pieces.append(_mk_sentence(rng, w_count))
        return " ".join(pieces)

    return _factory


# --- legacy bytes payload kept for completeness (unused for text benches) ---


def fixed_bytes_factory(size_bytes: int, seed: int = 0) -> Callable[[str], bytes]:
    if size_bytes < 0:
        raise ValueError("size_bytes must be >= 0")
    rng = np.random.default_rng(seed)

    def _bytes_for(key: str) -> bytes:
        jump = (hash(key) & 0xFFFFFFFF) % 10_000_000
        _ = rng.integers(0, 2**32, size=1 + (jump % 3))
        return rng.bytes(size_bytes) if size_bytes > 0 else b""

    return _bytes_for
