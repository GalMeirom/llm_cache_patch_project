# bench_eviction.py
"""
Runs eviction benchmarks and records metrics. (Warm-up per-request capable)
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gptcache.manager.eviction.memory_cache import MemoryCacheEviction

import benches.utils.utils as utils
from benches.utils.flags_to_dict import parse_flags
from benches.utils.write_csv import write_dicts_to_csv


def run_bench(
    *,
    policy: str = "LRU",
    maxsize: int = 1000,
    clean_size: int = 1,
    miss_lag_s: float = 0.1,
    bench_type: str = "mixed",
    total_keys: int = 1000,
    unique_ratio: float = 0.1,
    theta: float = 1.2,
    count: int = 1000,
    seed: int = 0,
    per_writer: csv.writer | None = None,
) -> dict[str, Any]:
    """Single configuration run; if per_writer is set,
    write per-request rows: idx,policy,hit,lat_ms."""
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
    for idx, p in enumerate(prompts):
        ts = time.perf_counter()
        if ev.get(p):
            is_hit = 1
            hits += 1
        else:
            is_hit = 0
            time.sleep(float(miss_lag_s))
            ev.put([p])
        te = time.perf_counter()

        lat = (te - ts) * 1000.0
        lat_ms.append(lat)

        if per_writer is not None:
            per_writer.writerow([idx, policy, is_hit, float(lat)])

    t1 = time.perf_counter()
    dur = max(t1 - t0, 1e-9)

    return {
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
        "duration_s": dur,
        "lat_ms_mean": statistics.fmean(lat_ms) if lat_ms else 0.0,
        "lat_ms_p95": utils._pctl(lat_ms, 95.0),
        "lat_ms_p99": utils._pctl(lat_ms, 99.0),
        "hit_rate": (hits / count) if count else 0.0,
        "qps": count / dur,
    }


def _open_per_request_writer(path: str) -> tuple[csv.writer, Any]:
    """Append mode; write header if file is empty/nonexistent."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if not file_exists:
        w.writerow(["idx", "policy", "hit", "lat_ms"])
    return w, f


def main(argv: list[str]) -> None:
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
        "per_request_out": None,  # NEW
        "print": 0,
    }
    csv_prefix = "./benches/results/raw/eviction_bench/"
    flags = parse_flags(argv)
    cfg = {**defaults, **flags}

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

    per_writer = None
    per_file = None
    if cfg.get("per_request_out"):
        per_path = cfg["per_request_out"]
        # keep path as provided; create dirs if needed
        per_writer, per_file = _open_per_request_writer(per_path)

    results: list[dict[str, Any]] = []
    try:
        for params in utils._cartesian(sweep):
            for k, v in list(params.items()):
                if isinstance(v, str):
                    try:
                        params[k] = float(v) if "." in v else int(v)
                    except Exception:
                        pass
            res = run_bench(**params, per_writer=per_writer)
            results.append(res)
            if int(cfg.get("print", 0)) == 1:
                print(json.dumps(res, indent=2))
    finally:
        if per_file is not None:
            per_file.close()

    if cfg.get("csv_out"):
        out_path = csv_prefix + cfg["csv_out"]
        write_dicts_to_csv(results, out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
