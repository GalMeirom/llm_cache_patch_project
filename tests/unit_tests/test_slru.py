import importlib
import importlib.util

import pytest


def _import_slru():
    return importlib.import_module("gptcache.manager.eviction.policies.slru")


# 1) Smoke: SLRU package is present inside gptcache
def test_slru_module_is_importable():
    m = _import_slru()
    assert hasattr(
        m, "SLRUCache"
    ), "SLRUCache class not found in gptcache.manager.eviction.policies.slru"
    # overlay probe (optional)
    try:
        probe = importlib.import_module("gptcache._llm_patch_probe")
        assert getattr(probe, "LLM_PATCH_SENTINEL", False) is True
    except ModuleNotFoundError:
        pass


# Helper to assert segment ordering (MRU -> LRU)
def assert_segments(cache, expected_protected=None, expected_probation=None):
    prot, prob = cache.segments()
    if expected_protected is not None:
        assert tuple(prot) == tuple(expected_protected)
    if expected_probation is not None:
        assert tuple(prob) == tuple(expected_probation)


# 2) Core SLRU semantics (promotion, demotion, eviction)
def test_slru_promote_demote_and_evict_basic():
    SLRU = _import_slru().SLRUCache
    c = SLRU(maxsize=4, protected_ratio=0.5)  # protected target=2, probation target=2

    # Insert 3 -> all in probation (MRU->LRU: c,b,a)
    for k in ["a", "b", "c"]:
        c[k] = True
    assert set(c.keys()) == {"a", "b", "c"}
    assert_segments(c, expected_protected=(), expected_probation=("c", "b", "a"))

    # Hit 'b' -> promote to protected MRU
    _ = c["b"]
    assert_segments(c, expected_protected=("b",), expected_probation=("c", "a"))

    # Add 'd' then 'e' -> evict probation LRU ('a')
    c["d"] = True
    assert_segments(c, expected_protected=("b",), expected_probation=("d", "c", "a"))
    c["e"] = True
    assert "a" not in c
    assert_segments(c, expected_protected=("b",), expected_probation=("e", "d", "c"))

    # Hit 'c' -> promote to protected; no demotion since protected target not exceeded
    _ = c["c"]
    assert_segments(c, expected_protected=("c", "b"), expected_probation=("e", "d"))


# 3) Edge cases
def test_slru_edge_cases():
    SLRU = _import_slru().SLRUCache

    with pytest.raises((ValueError, AssertionError)):
        SLRU(maxsize=0)

    for r in (-0.1, 0.0, 1.0, 1.5):
        with pytest.raises((ValueError, AssertionError)):
            SLRU(maxsize=4, protected_ratio=r)

    # popitem prefers probation LRU, then protected LRU
    c = SLRU(maxsize=3, protected_ratio=1 / 3)  # protected target ~1
    for k in ["a", "b", "c"]:
        c[k] = True
    _ = c["b"]  # promote
    assert_segments(c, expected_protected=("b",), expected_probation=("c", "a"))

    k1, _ = c.popitem()
    k2, _ = c.popitem()
    k3, _ = c.popitem()
    assert (k1, k2, k3) == ("a", "c", "b")
    assert len(c) == 0


# 4) Integration with EvictionBase / memory eviction
def test_memory_cache_eviction_with_slru_on_evict_callback():
    from gptcache.manager.eviction.manager import EvictionBase
    from gptcache.manager.eviction.memory_cache import MemoryCacheEviction

    evicted_keys = []

    def on_evict(keys):
        evicted_keys.extend(keys)

    ev = EvictionBase.get(name="memory", policy="SLRU", maxsize=3, clean_size=2, on_evict=on_evict)
    assert isinstance(ev, MemoryCacheEviction)
    assert ev._policy == "SLRU"

    # Insert 3, then one more to trigger cleanup of 2 keys (clean_size=2)
    for k in ["a", "b", "c"]:
        ev.put([k])  # MemoryCacheEviction.put expects a list of ids
    ev.put(["d"])

    assert "d" in ev._cache and "c" in ev._cache
    assert "a" not in ev._cache and "b" not in ev._cache
    assert (
        len(evicted_keys) == 2
        and all(k in {"a", "b", "c"} for k in evicted_keys)
        and "d" not in evicted_keys
    )


# 5) End-to-end GPTCache instance using SLRU


def test_slru_with_full_gptcache_pipeline_onnx():
    from gptcache.core import Cache
    from gptcache.processor.pre import get_prompt

    from sDM import sDataManager

    dm = sDataManager(max_size=5, clean_size=2, policy="SLRU")

    c = Cache()
    c.init(
        pre_embedding_func=get_prompt,
        data_manager=dm,
    )

    # Verify SLRU is active
    eb = getattr(c.data_manager, "_eviction", None)
    assert eb is not None and getattr(eb, "_policy", "").upper() == "SLRU"

    # Fill & exercise eviction
    for i in range(12):
        q, a = f"q{i}", f"a{i}"
        c.data_manager.save(q, a, q)

    assert len(c.data_manager._eviction._cache) <= 5

    # Search + hit callback
    res = c.data_manager.search("q11")
    assert isinstance(res, list)
    if res:
        c.data_manager.hit_cache_callback(res[0])
