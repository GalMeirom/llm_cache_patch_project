# benches/workloads/text_payloads.py
import hashlib
from collections.abc import Callable

from faker import Faker

__all__ = [
    "utf8_sizeof",
    "short_sentence_factory",
    "long_sentences_factory",
]


def utf8_sizeof(s: str) -> int:
    """UTF-8 byte length for cache getsizeof."""
    return len(s.encode("utf-8"))


def _seed_from_key(seed: int, key: str) -> int:
    """Stable per-key seed (independent of Python hash randomization)."""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    key32 = int.from_bytes(h[:4], "big")
    return (seed ^ key32) & 0xFFFFFFFF


def _sentence_with_words(fake: Faker, min_w: int, max_w: int) -> str:
    n = fake.random.randint(min_w, max_w)
    words = fake.words(nb=n, unique=False)  # lorem provider
    if not words:
        return ""
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def short_sentence_factory(
    *, seed: int = 1337, words_range: tuple[int, int] = (2, 6)
) -> Callable[[str], str]:
    """
    Build f(key)-> one short sentence (2–6 words by default), deterministic per key.
    """
    min_w, max_w = words_range
    if not (1 <= min_w <= max_w):
        raise ValueError("invalid words_range")

    def _make(key: str) -> str:
        s = _seed_from_key(seed, key)
        fake = Faker("en_US")
        fake.seed_instance(s)
        return _sentence_with_words(fake, min_w, max_w)

    return _make


def long_sentences_factory(
    *,
    seed: int = 4242,
    sentences_range: tuple[int, int] = (2, 5),
    words_range: tuple[int, int] = (12, 18),
) -> Callable[[str], str]:
    """
    Build f(key)-> 2–5 sentences, each 12–18 words by default, deterministic per key.
    """
    min_s, max_s = sentences_range
    min_w, max_w = words_range
    if not (1 <= min_s <= max_s):
        raise ValueError("invalid sentences_range")
    if not (1 <= min_w <= max_w):
        raise ValueError("invalid words_range")

    def _make(key: str) -> str:
        s = _seed_from_key(seed, key)
        fake = Faker("en_US")
        fake.seed_instance(s)
        count = fake.random.randint(min_s, max_s)
        return " ".join(_sentence_with_words(fake, min_w, max_w) for _ in range(count))

    return _make
