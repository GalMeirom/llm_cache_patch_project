# LLM Cache Patch Project — GPTCache + New Eviction Policies

This repository augments **GPTCache** with three eviction policies and ships a reproducible workflow: unit tests, two benchmark suites, plotting utilities, and an optional Docker setup.

- **New policies**
  - **SLRU** — Segmented LRU (probation/protected, two-touch promotion).
  - **TLFU_SLRU** — TinyLFU admission gate + SLRU residency (**best default** in our tests).
  - **GDSF** — GreedyDual-Size-Frequency; priority `H = L + freq * (cost/size)`.

- **Synthetic components** (fast, deterministic, API-free)
  - **sDataManager** — in-memory DataManager compatible with GPTCache.
  - **SyntheticLLM** — LangChain-compatible LLM; deterministic 5-word output.

- **Benchmarks**
  - **Cache-level** — full GPTCache path with `sDM + sLLM` (compact items).
  - **Eviction-only** — tests the eviction layer in isolation (stores full prompts).

> **One-time overlay step:** After cloning and installing requirements, run `bootstrap\apply_overlay.py` **exactly once**. Do **not** run it again. If you need to re-apply, prefer a fresh virtualenv or use `bootstrap\rollback.py` first.

---

## Repository Layout (trimmed)

```
llm_cache_patch_project
├─ benches/
│  ├─ runners/
│  │  ├─ bench_cache.py
│  │  ├─ bench_eviction.py
│  │  └─ bench_eviction_to_warmup.py
│  ├─ results/
│  │  ├─ raw/
│  │  │  ├─ eviction_bench\*.csv
│  │  │  └─ cache_bench\*.csv
│  │  └─ funcs\
│  │     ├─ plot_hit_rate_triplet.py
│  │     ├─ plot_hit_rate_vs_capacity_bars.py
│  │     ├─ plot_latency_bars.py
│  │     └─ plot_warmup_curve.py
│  ├─ workloads\
│  │  ├─ generators.py
│  │  └─ text_payloads.py
│  ├─ plots\
│  │  ├─ cache_bench\*.pdf
│  │  └─ eviction_bench\*.pdf
│  └─ utils\
│     ├─ flags_to_dict.py
│     └─ write_csv.py
├─ bootstrap\
│  ├─ apply_overlay.py
│  ├─ rollback.py
│  └─ install_and_verify.py
├─ overlays\gptcache\manager\eviction\
│  ├─ memory_cache.py
│  └─ policies\ (slru.py, tinylfu_slru.py, gdsf.py)
├─ sDM\   (sDataManager / sEvictionManager)
├─ sLLM\  (SyntheticLLM)
├─ tests\
│  ├─ conftest.py
│  └─ unit_tests\ (test_slru.py, test_tlfu_slru.py, test_gdsf.py)
├─ docker\entrypoint.py
├─ Dockerfile
├─ .dockerignore
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

---

## Prerequisites

- Windows + **PowerShell** (commands here use PowerShell).
- **Python 3.11** (recommended).
- **pip** 23+.
- (Optional) **Docker** for containerized runs.

---

## Quick Start (PowerShell)

```powershell
# 1) Clone
git clone <YOUR_REPO_URL> llm_cache_patch_project
cd .\llm_cache_patch_project

# 2) Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 3) Install deps
pip install --upgrade pip
pip install -r .\requirements.txt

# 4) Apply the overlay (ONE TIME ONLY)
python .\bootstrap\apply_overlay.py
```

If you need to **re-apply**, prefer a fresh venv or run `bootstrap\rollback.py` first. Re-running the overlay can corrupt patched files.

---

## Normal Use

### A) Vanilla GPTCache (no custom policy)

```python
# examples\vanilla_cache.py
from gptcache import Cache
from gptcache.processor.pre import get_prompt

cache = Cache()
# Minimal init; here we avoid embeddings by using the prompt as key
cache.init(pre_embedding_func=get_prompt)

# Use GPTCache adapters as usual…
```

### B) GPTCache with our policies (fast, API-free path)

This uses:
- `sDataManager` (in-memory, compact storage),
- `SyntheticLLM` (deterministic, no external API),
- `pre_embedding_func=get_prompt` (no embeddings).

```python
# examples\custom_policies.py
from gptcache import Cache
from gptcache.processor.pre import get_prompt
from gptcache.adapter.langchain_models import LangChainLLMs

from sDM import sDataManager          # our in-memory DataManager
from sLLM import SyntheticLLM         # our deterministic LLM

# Choose a policy: "LRU", "LFU", "FIFO", "RR", "SLRU", "TLFU_SLRU", or "GDSF"
dm = sDataManager(max_size=1000, policy="TLFU_SLRU")

cache = Cache()
cache.init(pre_embedding_func=get_prompt, data_manager=dm)

llm = LangChainLLMs(llm=SyntheticLLM())

q = "S: give me a tiny summary about caching"
resp = llm.generate([q])  # routed through GPTCache; miss populates via sDM
print(resp.generations[0][0].text)
```

**Notes**
- With `sDataManager + get_prompt`, entries are **compact prompt identifiers**.
- To use real embeddings/vector stores: configure GPTCache’s standard data managers and embedding functions; pass `policy="SLRU"` / `"TLFU_SLRU"` / `"GDSF"` wherever the manager accepts a policy flag.

---

## Running Tests (pytest)

```powershell
# From repo root, venv active, overlay applied
pytest -q

# Or specific files
pytest -q .\tests\unit_tests\test_slru.py
pytest -q .\tests\unit_tests\test_tlfu_slru.py
pytest -q .\tests\unit_tests\test_gdsf.py
```

**Tips**
- `tests\conftest.py` ensures the repo root is on `sys.path`.
- Imports in tests should be **absolute**, e.g.:
  ```python
  from sDM import sDataManager
  from sLLM import SyntheticLLM
  ```
- If you hit an import error, confirm `sDM\__init__.py` and `sLLM\__init__.py` exist.

---

## Running Benchmarks

Two complementary suites live under `benches\runners\`.

### Common CLI switches
- `--policies` e.g. `LRU,LFU,FIFO,RR,SLRU,TLFU_SLRU,GDSF`
- `--workload` `short | novel | mixed`
- `--count` total requests (e.g., `30000`)
- `--seed` RNG seed
- `--miss-lag-s` sleep on miss (simulated latency)
- `--csv-out` output CSV path
  *(cache-level only)*:
- `--capacities-kb` KB budgets (e.g., `1 2 4`) with **byte-accurate** accounting

### 1) Cache-level bench (full GPTCache path)

Uses **sDataManager + SyntheticLLM** and `pre_embedding_func=get_prompt`. Items are compact → higher throughput, moderate pressure.

```powershell
python .\benches\runners\bench_cache.py `
  --policies LRU,SLRU,TLFU_SLRU,GDSF `
  --workload mixed `
  --count 30000 `
  --seed 42 `
  --miss-lag-s 0.05 `
  --capacities-kb 1 2 4 `
  --csv-out .\benches\results\raw\cache_bench\mixed_cache.csv
```

### 2) Eviction-only bench (eviction layer in isolation)

Stores **full prompts** as values (larger footprint → stronger capacity pressure). No LLM path is involved.

```powershell
python .\benches\runners\bench_eviction.py `
  --policies LRU,SLRU,TLFU_SLRU,GDSF `
  --workload mixed `
  --count 30000 `
  --seed 42 `
  --miss-lag-s 0.05 `
  --csv-out .\benches\results\raw\eviction_bench\mixed_eviction.csv
```

### 3) Warmup curve helper (optional)

```powershell
python .\benches\runners\bench_eviction_to_warmup.py `
  --policies LRU,SLRU,TLFU_SLRU,GDSF `
  --workload short `
  --count 50000 `
  --seed 42 `
  --miss-lag-s 0.05 `
  --csv-out .\benches\results\raw\eviction_bench\warmup_per_request.csv
```

### Workloads & Generators

- **short** — Zipfian hot set; tiny sentences (few words).
- **novel** — large near-uniform keyspace; multi-sentence payloads; scan-heavy.
- **mixed** — interleaves `short` and `novel` streams (e.g., `--short-portion 0.7`).

Generators are deterministic (seeded). UTF-8 size accounting is available for KB budgets.

### CSV & Plots

- Raw CSVs → `benches\results\raw\{cache_bench,eviction_bench}\`
- Prebuilt PDFs → `benches\plots\{cache_bench,eviction_bench}\`

Regenerate plots from CSVs:

```powershell
python .\benches\results\funcs\plot_hit_rate_triplet.py
python .\benches\results\funcs\plot_latency_bars.py
python .\benches\results\funcs\plot_hit_rate_vs_capacity_bars.py
python .\benches\results\funcs\plot_warmup_curve.py
```

---

## Docker (optional)

A Dockerfile and entrypoint are provided for reproducible runs. The entrypoint **runs the overlay once** using a sentinel file.

### Build

```powershell
docker build -t gptcache-bench .
```

### Run tests (overlay will run once)

```powershell
docker run --rm -v "$((Get-Location).Path):/app" gptcache-bench
```

### Run a bench inside the container

```powershell
docker run --rm -v "$((Get-Location).Path):/app" gptcache-bench `
  python .\benches\runners\bench_cache.py `
    --policies LRU,SLRU,TLFU_SLRU,GDSF `
    --workload mixed --count 30000 --seed 42 `
    --csv-out .\benches\results\raw\cache_bench\mixed_cache.csv
```

**Environment variables**
- `SKIP_OVERLAY=1` — skip overlay if you already applied it locally and you’re just using the container as a runner.

---

## Design Notes (short)

- **TLFU_SLRU** is the most consistent top performer on mixed/novel workloads (admission + two-touch).
- **SLRU** and **LFU** achieve very similar hit-rate on our traces despite different mechanics.
- **GDSF** requires credible per-key cost signals; with flat costs it’s LFU-like.
- **Byte-accurate sizing** at small KB budgets changes outcomes materially; use UTF-8 byte counts for text.

---

## Troubleshooting

**`ImportError: attempted relative import with no known parent package`**
- Tests run with repo root on `sys.path` via `tests\conftest.py`.
- Use absolute imports:
  ```python
  from sDM import sDataManager
  from sLLM import SyntheticLLM
  ```
- Ensure `sDM\__init__.py` and `sLLM\__init__.py` exist.

**Overlay ran twice**
- Avoid reruns. If needed, recreate the venv or run `bootstrap\rollback.py`, then apply once.

**PowerShell activation blocked**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**Where are results/plots?**
- CSVs → `benches\results\raw\{cache_bench,eviction_bench}\`
- PDFs → `benches\plots\{cache_bench,eviction_bench}\`

---
