# benches/utils/utils.py
from __future__ import annotations

import random
from collections.abc import Iterable
from itertools import product
from typing import Any

from benches.workloads.generators import uniform_stream, zipf_stream
from benches.workloads.text_payloads import long_sentences_factory, short_sentence_factory


def _build_prompts(
    bench_type: str,
    *,
    count: int,
    total_keys: int,
    unique_ratio: float,
    theta: float,
    seed: int,
    short_portion: float = 0.7,  # portion of SHORT prompts when bench_type="mixed"
) -> Iterable[str]:
    """Yield prompt strings as requested."""
    if bench_type not in {"short", "novel", "mixed"}:
        raise ValueError("bench_type must be 'short', 'novel', or 'mixed'")

    sfn = short_sentence_factory(seed=seed)
    lfn = long_sentences_factory(seed=seed, sentences_range=(2, 5), words_range=(6, 16))

    if bench_type == "short":
        keys = uniform_stream(
            total_keys=total_keys, unique_ratio=unique_ratio, count=count, seed=seed
        )
        return (sfn(k) for k in keys)

    if bench_type == "novel":
        ur = max(0.9999, unique_ratio)
        keys = uniform_stream(total_keys=total_keys, unique_ratio=ur, count=count, seed=seed)
        return (lfn(k) for k in keys)

    # bench_type == "mixed": Bernoulli choice per item to match expected ratio

    p_short = min(max(float(short_portion), 0.0), 1.0)
    a = int(round(count * p_short))  # cap of shorts to keep totals bounded
    b = count - a

    ks_short = zipf_stream(
        total_keys=total_keys,
        unique_ratio=max(0.05, unique_ratio),
        theta=theta,
        count=a,
        seed=seed,
    )

    ks_long = uniform_stream(
        total_keys=total_keys, unique_ratio=unique_ratio, count=b, seed=seed + 1
    )

    rng = random.Random(seed)

    def _gen():
        it_s, it_l = iter(ks_short), iter(ks_long)
        ns = nl = 0
        while ns + nl < count:
            pick_short = rng.random() < p_short  # use "<" so E[short]=p_short
            if pick_short and ns < a:
                k = next(it_s)
                ns += 1
                yield sfn(k)
            elif nl < b:
                k = next(it_l)
                nl += 1
                yield lfn(k)
            else:
                # fallback if one iterator exhausted
                if ns < a:
                    k = next(it_s)
                    ns += 1
                    yield sfn(k)
                elif nl < b:
                    k = next(it_l)
                    nl += 1
                    yield lfn(k)
                else:
                    break

    return _gen()


def _pctl(values: list[float], p: float) -> float:
    """Percentile p∈[0,100] with linear interpolation."""
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _as_list(v: Any) -> list[Any]:
    """Normalize a flag value to a list; splits comma-separated strings."""
    if isinstance(v, list | tuple):
        return list(v)
    if isinstance(v, str) and "," in v:
        return [s.strip() for s in v.split(",")]
    return [v]


def _cartesian(dict_of_lists: dict[str, list[Any]]):
    """Yield dicts over the cartesian product of dict_of_lists' values."""
    keys = list(dict_of_lists.keys())
    pools = [dict_of_lists[k] for k in keys]
    for combo in product(*pools):
        yield dict(zip(keys, combo, strict=False))
