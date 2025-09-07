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
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# === your utilities ===
from gptcache.manager.eviction.memory_cache import MemoryCacheEviction

import benches.utils.utils as utils
from benches.utils.flags_to_dict import parse_flags
from benches.utils.write_csv import write_dicts_to_csv

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
    prompts = utils._build_prompts(
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
        "lat_ms_p95": utils._pctl(lat_ms, 95.0),
        "lat_ms_p99": utils._pctl(lat_ms, 99.0),
        "hit_rate": (hits / count) if count else 0.0,
        "qps": count / dur,
    }


# ---------------- driver ----------------


def main(argv: list[str]) -> None:
    # Defaults
    defaults = {
        "policy": "LRU",
        "maxsize": 1000,
        "clean_size": 1,
        "miss_lag_s": 0.01,
        "bench_type": "mixed",
        "total_keys": 10000,
        "unique_ratio": 0.7,
        "theta": 1.2,
        "count": 10000,
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
    sweep = {k: utils._as_list(cfg.get(k)) for k in sweep_keys}

    results: list[dict[str, Any]] = []
    for params in utils._cartesian(sweep):
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
