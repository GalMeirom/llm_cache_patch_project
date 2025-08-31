# File: tests/policies/test_gdsf.py
import importlib
import importlib.util

import numpy as np
import pytest

# ---------------------------
# Helpers to import targets
# ---------------------------


def _import_gdsf_module():
    return importlib.import_module("gptcache.manager.eviction.policies.gdsf")


def _GDSF():
    return _import_gdsf_module().GDSFCache


def _import_memory_cache_eviction():
    # GPTCache standard class that chooses policy via if/elif in __init__
    return importlib.import_module("gptcache.manager.eviction.memory_cache").MemoryCacheEviction


# ---------------------------
# 0) Smoke / module presence
# ---------------------------


def test_gdsf_module_is_importable_and_has_class():
    m = _import_gdsf_module()
    assert hasattr(m, "GDSFCache"), "GDSFCache class not found"


# ----------------------------------
# 1) Unit tests: core GDSF behavior
# ----------------------------------


def _build_gdsf(cost_fn=None, size_fn=None, getsizeof=None, maxsize=8):
    GDSF = _GDSF()
    return GDSF(maxsize=maxsize, getsizeof=getsizeof, cost_fn=cost_fn, size_fn=size_fn)


def _mapping_value(cost: float = 1.0, sz: int = 1, **extra) -> dict:
    # default cost path uses .get('miss_penalty_ms'|'latency_ms'|'cost', 1.0)
    v = {"cost": cost, "sz": sz}
    v.update(extra)
    return v


def _calc_priority(L: float, freq: int, cost: float, size: int) -> float:
    return L + (freq * (cost / size))


def test_gdsf_invalid_maxsize_raises():
    GDSF = _GDSF()
    with pytest.raises(ValueError):
        GDSF(maxsize=0)
    with pytest.raises(ValueError):
        GDSF(maxsize=-1)


def test_gdsf_basic_insert_hit_freq_and_priority_refresh():
    c = _build_gdsf(size_fn=lambda k, v: int(v["sz"]))
    # Insert a -> freq=1; L=0; H = 0 + 1*(10/5) = 2
    c["a"] = _mapping_value(cost=10.0, sz=5)
    assert c._freq["a"] == 1
    pa, cnt = c._handles["a"]
    assert pytest.approx(pa) == 2.0
    assert any(e[2] == "a" for e in c._heap)

    # Hit a -> freq=2; push new heap entry; new priority = 0 + 2*(10/5) = 4
    _ = c["a"]
    assert c._freq["a"] == 2
    pa2, cnt2 = c._handles["a"]
    assert pa2 > pa and pytest.approx(pa2) == 4.0
    # Multiple heap entries for 'a' exist (lazy invalidation)
    assert sum(1 for e in c._heap if e[2] == "a") >= 2
    # Latest handle matches a heap entry exactly (priority+counter)
    assert (pa2, cnt2, "a") in c._heap


def test_gdsf_default_cost_resolution_order():
    # miss_penalty_ms > latency_ms > cost > 1.0
    c = _build_gdsf(size_fn=lambda k, v: 1)
    c["mp"] = {"miss_penalty_ms": 5.0}
    c["lat"] = {"latency_ms": 4.0}
    c["cost_only"] = {"cost": 3.0}
    c["none"] = {}  # -> cost=1.0 by default
    assert (
        c._handles["mp"][0]
        > c._handles["lat"][0]
        > c._handles["cost_only"][0]
        > c._handles["none"][0]
    )


def test_gdsf_eviction_order_and_watermark_monotonicity():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"])
    c["a"] = _mapping_value(cost=10.0, sz=10)  # H=1
    c["b"] = _mapping_value(cost=10.0, sz=5)  # H=2
    c["c"] = _mapping_value(cost=15.0, sz=5)  # H=3

    # Evict in ascending priority; L should step 1 -> 2 -> 3
    k1, _ = c.popitem()
    assert k1 == "a"
    assert pytest.approx(c._L) == 1.0

    k2, _ = c.popitem()
    assert k2 == "b"
    assert pytest.approx(c._L) == 2.0

    k3, _ = c.popitem()
    assert k3 == "c"
    assert pytest.approx(c._L) == 3.0


def test_gdsf_size_bias_small_objects_survive():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"], maxsize=1)
    c["big"] = _mapping_value(cost=10.0, sz=10)  # H=1.0
    c["small"] = _mapping_value(cost=10.0, sz=1)  # triggers eviction; small has higher H
    assert "small" in c and "big" not in c


def test_gdsf_getsizeof_path_is_used_when_size_fn_missing():
    c = _build_gdsf(
        size_fn=None,
        getsizeof=lambda v: len(v) if isinstance(v, str) else 1,
        maxsize=10,  # must be >= max item size; 4 ("xxxx") + 2 ("yy") <= 10
    )
    # 'xxxx' size=4, default cost=1 => H=0.25
    c["k1"] = "xxxx"
    # 'yy' size=2, default cost=1 => H=0.5
    c["k2"] = "yy"

    # Lowest priority should be k1
    k, _ = c.popitem()
    assert k == "k1"
    assert "k2" in c


def test_gdsf_zero_or_negative_size_raises_valueerror():
    c0 = _build_gdsf(size_fn=lambda k, v: 0)
    with pytest.raises(ValueError):
        c0["x"] = _mapping_value(cost=1.0, sz=0)

    cneg = _build_gdsf(size_fn=lambda k, v: -5)
    with pytest.raises(ValueError):
        cneg["y"] = _mapping_value(cost=1.0, sz=-5)


def test_gdsf_negative_cost_is_allowed_and_affects_priority():
    c = _build_gdsf(cost_fn=lambda k, v: float(v["cost"]), size_fn=lambda k, v: v["sz"], maxsize=2)
    c["a"] = {"cost": -1.0, "sz": 1}  # H = -1
    c["b"] = {"cost": 1.0, "sz": 1}  # H = +1
    c["c"] = {"cost": 1.0, "sz": 1}  # Eviction happens; 'a' should be weakest
    assert "a" not in c and "b" in c and "c" in c


def test_gdsf_lazy_invalidation_and_handles_consistency():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"])
    c["a"] = _mapping_value(cost=5.0, sz=2)  # H=2.5
    for _ in range(5):
        _ = c["a"]  # bump frequency; many heap entries
    assert c._freq["a"] == 6
    pr, cnt = c._handles["a"]
    assert (pr, cnt, "a") in c._heap

    # Add clearly lower priorities, then keep evicting until 'a' goes
    c["b"] = _mapping_value(cost=0.1, sz=1)
    c["c"] = _mapping_value(cost=0.2, sz=1)
    removed = set()
    while len(c) > 0:
        k, _ = c.popitem()
        removed.add(k)
        if k == "a":
            break
    assert "a" in removed
    assert "a" not in c._handles
    assert "a" not in c._freq


def test_gdsf_clear_and_delitem_cleanup():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"])
    for k in ["a", "b", "c"]:
        c[k] = _mapping_value(cost=1.0, sz=1)

    del c["b"]
    assert "b" not in c
    assert "b" not in c._handles
    assert "b" not in c._freq

    # Pop should skip any stale heap entries
    k, _ = c.popitem()
    assert k in {"a", "c"}

    c.clear()
    assert len(c) == 0 and not c._heap and not c._handles and not c._freq
    assert c._L == 0.0


def test_gdsf_equal_priority_uses_counter_tiebreak_fifo_among_equals():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"], maxsize=2)
    c["a"] = _mapping_value(cost=1.0, sz=1)  # H=1
    c["b"] = _mapping_value(cost=1.0, sz=1)  # H=1
    # Insert 'c' -> equal H; earliest insertion should evict first ('a')
    c["c"] = _mapping_value(cost=1.0, sz=1)
    assert "a" not in c and "b" in c and "c" in c


def test_gdsf_popitem_fallback_when_heap_is_empty_keeps_L_unchanged():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"])
    c["x"] = _mapping_value(cost=1.0, sz=1)  # H=1
    c["y"] = _mapping_value(cost=0.5, sz=1)  # H=0.5 -> evicted first
    k, _ = c.popitem()
    assert k == "y" and pytest.approx(c._L) == 0.5

    # Force heap staleness and fallback
    c["a"] = _mapping_value(cost=2.0, sz=2)
    c["b"] = _mapping_value(cost=3.0, sz=3)
    c._heap.clear()  # empty heap to trigger fallback path
    prev_L = c._L
    k2, _ = c.popitem()  # fallback: data.popitem()
    assert k2 in {"a", "b"} and pytest.approx(c._L) == pytest.approx(prev_L)


def test_gdsf_priority_after_eviction_uses_updated_L_on_next_touch():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"])
    c["stay"] = _mapping_value(cost=10.0, sz=5)  # base H=2
    c["gone"] = _mapping_value(cost=1.0, sz=1)  # base H=1 -> evicted first
    k, _ = c.popitem()
    assert k == "gone"
    assert pytest.approx(c._L) == 1.0

    # Touch 'stay' now: freq -> 2; new H = L(=1) + 2*(10/5) = 5
    _ = c["stay"]
    pr, _ = c._handles["stay"]
    assert pytest.approx(pr) == 5.0


def test_gdsf_update_same_key_does_not_bump_freq_but_refreshes_priority():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"])
    c["k"] = _mapping_value(cost=2.0, sz=2)  # H = 1
    f1 = c._freq["k"]
    p1, _ = c._handles["k"]

    # Update value (same key), expect frequency unchanged, priority recomputed
    c["k"] = _mapping_value(cost=4.0, sz=2)  # H = 2
    f2 = c._freq["k"]
    p2, _ = c._handles["k"]
    assert f2 == f1  # no bump on set
    assert p2 > p1
    assert (p2, c._handles["k"][1], "k") in c._heap


def test_gdsf_segments_view_matches_keys():
    c = _build_gdsf(size_fn=lambda k, v: v["sz"])
    keys = ["k1", "k2", "k3"]
    for i, k in enumerate(keys):
        c[k] = _mapping_value(cost=1.0 + i, sz=1)
    seg = c.segments()
    assert isinstance(seg, dict) and "all" in seg
    assert set(seg["all"]) == set(keys)


# ----------------------------------------------------------
# 2) Wiring: MemoryCacheEviction integration (policy="GDSF")
# ----------------------------------------------------------


def test_memory_cache_eviction_with_gdsf_and_on_evict_callback():
    MemoryCacheEviction = _import_memory_cache_eviction()
    GDSF = _GDSF()

    evicted_keys = []

    def on_evict(keys):
        evicted_keys.extend(keys)

    ev = MemoryCacheEviction(
        policy="GDSF",
        maxsize=4,
        clean_size=2,
        on_evict=on_evict,
    )

    # Ensure underlying cache is GDSFCache
    assert isinstance(ev._cache, GDSF)

    # Insert 5 ids -> should trigger one cleanup of 2 keys
    for k in ["a", "b", "c", "d", "e"]:
        ev.put([k])

    assert len(ev._cache) <= 4
    assert len(evicted_keys) == 2
    assert len(set(evicted_keys)) == 2  # no duplicates


# ---------------------------------------------------------
# 3) End-to-end GPTCache pipeline (ONNX + vector backend)
# ---------------------------------------------------------


@pytest.mark.skipif(
    not (
        importlib.util.find_spec("onnxruntime")
        and (importlib.util.find_spec("hnswlib") or importlib.util.find_spec("faiss"))
    ),
    reason="Requires onnxruntime and a vector backend (hnswlib or faiss).",
)
def test_gdsf_with_full_gptcache_pipeline_onnx(tmp_path):
    from gptcache.core import Cache
    from gptcache.embedding import Onnx
    from gptcache.manager import CacheBase, VectorBase, get_data_manager
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

    onnx = Onnx()
    dim = onnx.dimension
    backend = "hnswlib" if importlib.util.find_spec("hnswlib") else "faiss"

    cb = CacheBase("sqlite", sql_url=f"sqlite:///{tmp_path/'sqlite.db'}")
    vb = VectorBase(backend, dimension=dim, index_path=str(tmp_path / f"{backend}.index"))

    dm = get_data_manager(
        cache_base=cb,
        vector_base=vb,
        object_base=None,
        eviction_base=None,  # use in-memory eviction
        max_size=5,
        clean_size=2,
        eviction="GDSF",
    )

    c = Cache()
    c.init(
        embedding_func=onnx.to_embeddings,
        data_manager=dm,
        similarity_evaluation=SearchDistanceEvaluation(),
    )

    # Verify MemoryCacheEviction chose GDSFCache
    ev = getattr(c.data_manager, "eviction_base", None)
    assert ev is not None
    assert isinstance(ev._cache, _GDSF())

    # Fill & exercise eviction
    for i in range(14):
        q, a = f"q{i}", f"a{i}"
        emb = np.asarray(onnx.to_embeddings(q), dtype="float32")
        c.data_manager.save(q, a, emb)

    assert len(c.data_manager.eviction_base._cache) <= 5

    # Search + hit callback (if any result is returned)
    res = c.data_manager.search(np.asarray(onnx.to_embeddings("q13"), dtype="float32"))
    assert isinstance(res, list)
    if res:
        c.data_manager.hit_cache_callback(res[0])
