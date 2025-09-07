# benches/runners/bench_cache.py
"""
Benchmark GPTCache.Cache + sDataManager using prompt generators.

Flow per run:
  1) Build sDataManager(max_size, policy) and Cache(); init with get_prompt.
  2) Wrap a SyntheticLLM(buffer=miss_lag_s) with LangChainLLMs.
  3) For each prompt from the chosen generator, call llm(prompt, cache_obj=cache).
     - On MISS: SyntheticLLM sleeps `buffer` seconds, so latency ~ miss_lag_s.
     - On HIT : Cache returns immediately (no sleep), so latency << miss_lag_s.
  4) Measure latency distribution, hit-rate (by time-threshold), throughput, CPU/RSS.

Flags (read via benches.utils.flags_to_dict.parse_flags):
  --policy LRU,LFU
  --maxsize 1000
  --clean_size 0
  --miss_lag_s 0.1
  --bench_type short|novel|mixed
  --short_portion 0.7           # only for mixed
  --total_keys 1000
  --unique_ratio 0.1
  --theta 1.2
  --count 1000
  --seed 0
  --csv_out results.csv
  --print 0|1                    # print each run's result
"""

from __future__ import annotations

import statistics
import sys
import time
from typing import Any

from gptcache import Cache  # type: ignore
from gptcache.adapter.langchain_models import LangChainLLMs  # type: ignore
from gptcache.processor.pre import get_prompt  # type: ignore

import benches.utils.utils as utils

# ---- use your utilities (run as module: python -m benches.runners.bench_cache)
from benches.utils.flags_to_dict import parse_flags
from benches.utils.write_csv import write_dicts_to_csv

# ---- workload & cache bits
from sDM.sDM import sDataManager
from sLLM.synthetic_llm import SyntheticLLM  # type: ignore

# ---------------- single-run bench ----------------


def run_bench(
    *,
    policy: str = "LRU",
    maxsize: int = 1000,
    clean_size: int = 0,
    miss_lag_s: float = 0.1,
    bench_type: str = "short",  # short|novel|mixed
    short_portion: float = 0.7,  # only used when mixed
    total_keys: int = 1000,
    unique_ratio: float = 0.1,
    theta: float = 1.2,
    count: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Create Cache + sDataManager + LangChainLLMs(SyntheticLLM) and run prompts through it.
    Hit detection heuristic: if latency < miss_lag_s * 0.5 -> treat as HIT (cache path).
    """
    # Data manager + cache
    dm = sDataManager(max_size=maxsize, clean_size=clean_size, policy=policy)
    cache = Cache()
    cache.init(pre_embedding_func=get_prompt, data_manager=dm)

    # LLM with synthetic time buffer for MISSes (cache bypass)
    llm = LangChainLLMs(llm=SyntheticLLM(buffer=float(miss_lag_s)))

    prompts = utils._build_prompts(
        bench_type,
        count=count,
        total_keys=total_keys,
        unique_ratio=unique_ratio,
        theta=theta,
        seed=seed,
        short_portion=short_portion,
    )

    lat_ms: list[float] = []
    hits = 0
    # Use a conservative threshold to classify hit vs miss
    hit_thresh_s = max(0.005, float(miss_lag_s) * 0.5)
    t0 = time.perf_counter()
    for _i, prompt in enumerate(prompts):

        ts = time.perf_counter()
        _ = llm(prompt, cache_obj=cache)  # response text unused; only timing
        te = time.perf_counter()

        dt = te - ts
        lat_ms.append(dt * 1000.0)
        if dt < hit_thresh_s:
            hits += 1

    t1 = time.perf_counter()
    dur = max(t1 - t0, 1e-9)

    return {
        # params
        "policy": policy,
        "maxsize": maxsize,
        "clean_size": clean_size,
        "miss_lag_s": miss_lag_s,
        "bench_type": bench_type,
        "short_portion": short_portion,
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
    defaults = {
        "policy": "LRU",
        "maxsize": 1000,
        "clean_size": 1,
        "miss_lag_s": 0.01,
        "bench_type": "short",
        "short_portion": 0.7,
        "total_keys": 10000,
        "unique_ratio": 0.7,
        "theta": 1.2,
        "count": 10000,
        "seed": 42,
        "csv_out": None,
        "print": 0,
    }
    csv_prefix = "./benches/results/raw/cache_bench/"
    flags = parse_flags(argv)
    cfg = {**defaults, **flags}

    sweep_keys = [
        "policy",
        "maxsize",
        "clean_size",
        "miss_lag_s",
        "bench_type",
        "short_portion",
        "total_keys",
        "unique_ratio",
        "theta",
        "count",
        "seed",
    ]
    sweep = {k: utils._as_list(cfg.get(k)) for k in sweep_keys}

    results: list[dict[str, Any]] = []
    for params in utils._cartesian(sweep):
        # coerce basic numerics/booleans from strings
        for k, v in list(params.items()):
            if isinstance(v, str):
                try:
                    if v.lower() in ("true", "false"):
                        params[k] = v.lower() == "true"
                    elif k in ("miss_lag_s", "unique_ratio", "theta", "short_portion"):
                        params[k] = float(v)
                    elif k in ("maxsize", "clean_size", "count", "seed"):
                        params[k] = int(v)
                except Exception:
                    pass

        res = run_bench(**params)
        results.append(res)

        if int(cfg.get("print", 0)) == 1:
            import json

            print(json.dumps(res, indent=2))

    if cfg.get("csv_out"):
        write_dicts_to_csv(results, csv_prefix + cfg["csv_out"])

    # final summary (optional)
    import json

    print(json.dumps({"runs": len(results)}, indent=2))


if __name__ == "__main__":
    # Run as: python -m benches.runners.bench_cache [flags...]
    main(sys.argv[1:])
