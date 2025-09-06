# bench_eviction.py
"""
Runs eviction benchmarks and records metrics.
- Uses your existing parse_flags() and write_dicts_to_csv().
- Policies run via MemoryCacheEviction.
- Prompts come from text_payloads.py + generators.py.
- Per configuration, call run_bench(...).
- For comma-separated flag values, runs the cartesian product.

Flags (examples):
  --policy LRU,LFU           # eviction policy/policies (default LRU)
  --maxsize 1000,5000        # cache sizes
  --clean_size 0             # optional clean size hint
  --miss_lag_s 0.1           # miss latency (seconds)
  --bench_type 1             # 1=short, 2=novel, 3=mixed
  --total_keys 1000          # size of key universe
  --unique_ratio 0.1         # 0..1; chance a request is new
  --theta 1.2                # Zipf theta (mixed bench)
  --count 1000               # total requests
  --seed 0                   # RNG seed for streams
  --csv_out results.csv      # optional CSV output (appends)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections.abc import Iterable
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# === your utilities ===
from gptcache.manager.eviction.memory_cache import MemoryCacheEviction

from benches.utils.flags_to_dict import parse_flags
from benches.utils.write_csv import write_dicts_to_csv
from benches.workloads.generators import uniform_stream, zipf_stream

# === workload + eviction (your provided files) ===
from benches.workloads.text_payloads import long_sentences_factory, short_sentence_factory

# optional system metrics
try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# ---------------- helpers ----------------


def _as_list(v: Any) -> list[Any]:
    """Turn flag value into a list (supports comma-separated strings)."""
    if isinstance(v, list | tuple):
        return list(v)
    if isinstance(v, str) and "," in v:
        return [s.strip() for s in v.split(",")]
    return [v]


def _pctl(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    if p <= 0:
        return min(vals)
    if p >= 100:
        return max(vals)
    xs = sorted(vals)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] if f == c else xs[f] * (c - k) + xs[c] * (k - f)


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
    import random

    p_short = min(max(float(short_portion), 0.0), 1.0)
    a = int(round(count * p_short))  # cap of shorts to keep totals bounded
    b = count - a

    ks_short = uniform_stream(total_keys=total_keys, unique_ratio=unique_ratio, count=a, seed=seed)
    ks_long = zipf_stream(
        total_keys=total_keys,
        unique_ratio=max(0.05, unique_ratio),
        theta=theta,
        count=b,
        seed=seed + 1,
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


# ---------------- core bench ----------------


def run_bench(
    *,
    policy: str = "LRU",
    maxsize: int = 1000,
    clean_size: int = 1,
    miss_lag_s: float = 0.1,
    bench_type: int = "mix",
    total_keys: int = 1000,
    unique_ratio: float = 0.1,
    theta: float = 1.2,
    count: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Single configuration run:
      - Key = prompt text (as-is).
      - Hit: no wait. Miss: sleep(miss_lag_s) then evict-base.put([key]).
    """
    ev = MemoryCacheEviction(
        policy=policy, maxsize=maxsize, clean_size=clean_size, on_evict=lambda ids: None
    )
    prompts = _build_prompts(
        bench_type,
        count=count,
        total_keys=total_keys,
        unique_ratio=unique_ratio,
        theta=theta,
        seed=seed,
    )

    lat_ms: list[float] = []
    hits = 0

    t0 = time.perf_counter()
    for _i, p in enumerate(prompts):

        ts = time.perf_counter()
        if ev.get(p):
            hits += 1
        else:
            time.sleep(float(miss_lag_s))
            ev.put([p])
        te = time.perf_counter()

        lat_ms.append((te - ts) * 1000.0)
    t1 = time.perf_counter()

    dur = max(t1 - t0, 1e-9)
    return {
        # params snapshot
        "policy": policy,
        "maxsize": maxsize,
        "clean_size": clean_size,
        "miss_lag_s": miss_lag_s,
        "bench_type": bench_type,
        "total_keys": total_keys,
        "unique_ratio": unique_ratio,
        "theta": theta,
        "count": count,
        "seed": seed,
        # metrics
        "duration_s": dur,
        "lat_ms_mean": statistics.fmean(lat_ms) if lat_ms else 0.0,
        "lat_ms_p95": _pctl(lat_ms, 95.0),
        "lat_ms_p99": _pctl(lat_ms, 99.0),
        "hit_rate": (hits / count) if count else 0.0,
        "qps": count / dur,
    }


# ---------------- driver ----------------


def _cartesian(dict_of_lists: dict[str, list[Any]]):
    from itertools import product

    keys = list(dict_of_lists.keys())
    pools = [dict_of_lists[k] for k in keys]
    for combo in product(*pools):
        yield dict(zip(keys, combo, strict=False))


def main(argv: list[str]) -> None:
    # Defaults
    defaults = {
        "policy": "LRU",
        "maxsize": 1000,
        "clean_size": 1,
        "miss_lag_s": 0.001,
        "bench_type": "mixed",
        "total_keys": 50000,
        "unique_ratio": 0.6,
        "theta": 1.2,
        "count": 20000,
        "seed": 0,
        "csv_out": None,
        "print": 0,
    }
    csv_prefix = "./benches/results/raw/eviction_bench/"
    flags = parse_flags(argv)
    cfg = {**defaults, **flags}

    # sweep keys support comma-separated values for multiple runs
    sweep_keys = [
        "policy",
        "maxsize",
        "clean_size",
        "miss_lag_s",
        "bench_type",
        "total_keys",
        "unique_ratio",
        "theta",
        "count",
        "seed",
    ]
    sweep = {k: _as_list(cfg.get(k)) for k in sweep_keys}

    results: list[dict[str, Any]] = []
    for params in _cartesian(sweep):
        # coerce numerics if they came in as strings
        for k, v in list(params.items()):
            if isinstance(v, str):
                try:
                    params[k] = float(v) if "." in v else int(v)
                except Exception:
                    pass
        res = run_bench(**params)
        results.append(res)

        if int(cfg.get("print", 0)) == 1:  # <-- per-run print
            print(json.dumps(res, indent=2))

    # optional CSV write
    if cfg.get("csv_out"):
        write_dicts_to_csv(results, csv_prefix + cfg["csv_out"])

    # stdout for quick inspection


if __name__ == "__main__":
    main(sys.argv[1:])
