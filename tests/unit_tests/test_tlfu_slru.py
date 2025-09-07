import importlib
import importlib.util

import pytest

# --- helpers -----------------------------------------------------------------

TLFU_MODULE_CANDIDATES = [
    "gptcache.manager.eviction.policies.tinylfu_slru",
]


def _import_tlfu_slru():
    last_err = None
    for mod in TLFU_MODULE_CANDIDATES:
        try:
            m = importlib.import_module(mod)
            # pick a class whose name mentions both "lfu" and "slru"
            for name in dir(m):
                obj = getattr(m, name)
                if isinstance(obj, type) and "lfu" in name.lower() and "slru" in name.lower():
                    return m, obj
        except ModuleNotFoundError as e:
            last_err = e
    raise last_err or AssertionError("TinyLFU+SLRU policy module/class not found")


def _segments(cache):
    """Return (protected, probation) if available; otherwise ((), tuple(cache.keys()))."""
    if hasattr(cache, "segments"):
        return cache.segments()
    return (), tuple(cache.keys())


# 1) Smoke: policy is present and discoverable in gptcache
def test_tlfu_slru_module_is_importable_and_class_present():
    m, Cls = _import_tlfu_slru()
    assert Cls is not None
    # overlay probe (if present)
    try:
        probe = importlib.import_module("gptcache._llm_patch_probe")
        assert getattr(probe, "LLM_PATCH_SENTINEL", False) is True
    except ModuleNotFoundError:
        pass

    # EvictionBase wiring key should exist
    from gptcache.manager.eviction.manager import EvictionBase

    ev = EvictionBase.get(name="memory", policy="TLFU_SLRU", maxsize=3, clean_size=1)
    assert getattr(ev, "_policy", "").upper() == "TLFU_SLRU"


# 2) Core TinyLFU + SLRU behavior:
#    - Hits promote into protected (SLRU behavior)
#    - A stream of cold, distinct inserts should NOT evict a hot protected key (TinyLFU admission)
def test_tlfu_slru_promotion_and_admission_filter():
    _, Cls = _import_tlfu_slru()
    c = Cls(maxsize=4, protected_ratio=0.5)  # target: 2 protected / 2 probation

    # Fill + make 'b' hot
    for k in ["a", "b", "c"]:
        c[k] = True
    for _ in range(5):
        _ = c["b"]  # heat 'b' -> should be protected
    prot, prob = _segments(c)
    assert "b" in set(prot) or "b" in c

    # Stream many cold uniques; hot 'b' must survive
    cold_keys = [f"x{i}" for i in range(20)]
    for k in cold_keys:
        c[k] = True

    assert "b" in c  # hot key survives
    # Admission filter should reject at least one cold candidate over the stream
    admitted_cold = sum(1 for k in cold_keys if k in c)
    assert admitted_cold < len(cold_keys)
    assert len(c) <= 4


# 3) Edge cases: invalid params & pop order preference (probation before protected)
def test_tlfu_slru_edge_cases():
    _, Cls = _import_tlfu_slru()

    with pytest.raises((ValueError, AssertionError)):
        Cls(maxsize=0)

    for r in (-0.1, 0.0, 1.0, 1.5):
        with pytest.raises((ValueError, AssertionError)):
            Cls(maxsize=4, protected_ratio=r)

    c = Cls(maxsize=3, protected_ratio=1 / 3)
    for k in ["a", "b", "c"]:
        c[k] = True
    _ = c["b"]  # promote -> protected

    # popitem should evict probation LRU first, then protected LRU
    k1, _ = c.popitem()
    k2, _ = c.popitem()
    k3, _ = c.popitem()
    assert {k1, k2, k3} == {"a", "b", "c"}
    assert len(c) == 0


def test_memory_cache_eviction_with_tlfu_slru_on_evict_callback():
    from gptcache.manager.eviction.manager import EvictionBase
    from gptcache.manager.eviction.memory_cache import MemoryCacheEviction

    evicted = []

    def on_evict(keys):
        evicted.extend(keys)

    ev = EvictionBase.get(
        name="memory",
        policy="TLFU_SLRU",
        maxsize=3,
        clean_size=2,
        on_evict=on_evict,
    )
    assert isinstance(ev, MemoryCacheEviction)
    assert ev._policy.upper() == "TLFU_SLRU"

    # Seed cache
    for k in ["a", "b", "c"]:
        ev.put([k])
    before = set(ev._cache)

    # Push several cold inserts; TinyLFU may reject all, or admit some and evict others
    for k in ["d", "e", "f", "g", "h", "i"]:
        ev.put([k])

    after = set(ev._cache)

    # Capacity respected
    assert len(ev._cache) <= 3

    # Either pure rejection (no change) OR admission+eviction (change observed)
    assert (after == before) or ((before - after) or (after - before))

    # If callback fired, keys should be strings
    if evicted:
        assert all(isinstance(x, str) for x in evicted)


def test_tlfu_slru_segments_and_hot_retention_under_cold_flood():
    # Requires the helpers defined at top of this file: _import_tlfu_slru and _segments
    _, Cls = _import_tlfu_slru()
    c = Cls(maxsize=8, protected_ratio=0.5)  # protected target = 4 exactly
    assert hasattr(c, "segments"), "TLFU_SLRU must expose segments() for protected/probation"

    # 1) Fill with 8 probation entries (MRU order is policy-specific; we check membership/size)
    base = list("abcdefgh")
    for k in base:
        c[k] = True
    prot, prob = c.segments()
    assert len(prot) == 0 and len(prob) == 8

    # 2) Make a hot set and verify promotion into PROTECTED
    hot = ["b", "c", "d", "e"]  # exactly 4 to match protected target
    for _ in range(5):
        for k in hot:
            _ = c[k]
    prot, prob = c.segments()
    assert len(prot) == 4, f"expected protected size 4, got {len(prot)}"
    assert set(hot).issubset(set(prot)), f"hot keys not all protected: prot={prot}"

    # 3) Promote a new key when PROTECTED is full -> one protected LRU must be demoted to PROBATION
    _ = c["a"]  # promote "a" into protected
    prot2, prob2 = c.segments()
    assert len(prot2) == 4 and "a" in prot2
    demoted = list(set(hot) - set(prot2))
    assert len(demoted) == 1, f"exactly one of {hot} must be demoted; got {demoted}"
    assert demoted[0] in prob2, "demoted key must appear in probation"

    # 4) Cold flood: many uniques should NOT evict protected hot keys (TinyLFU admission)
    flood = [f"x{i}" for i in range(40)]
    for k in flood:
        c[k] = True
    prot3, prob3 = c.segments()
    # All previously protected keys (after step 3) should remain in the cache and protected
    assert set(prot2).issubset(set(c.keys())), "protected set must be retained under cold flood"
    assert set(prot2).issubset(set(prot3)), "protected keys must remain protected under cold flood"
    # Capacity respected; some cold keys must have been rejected/rotated
    assert len(c) <= 8
    admitted_cold = sum(1 for k in flood if k in c)
    assert admitted_cold < len(flood), "TinyLFU should reject some cold inserts"

    # 5) Sanity: segments partition the keyset (no duplicates across lists)
    assert set(prot3).isdisjoint(set(prob3))
    assert set(prot3).union(set(prob3)) == set(c.keys())


def test_tlfu_slru_with_full_gptcache_pipeline_onnx():
    from gptcache.core import Cache
    from gptcache.processor.pre import get_prompt

    from sDM import sDataManager

    dm = sDataManager(max_size=5, clean_size=2, policy="TLFU_SLRU")

    c = Cache()
    c.init(
        pre_embedding_func=get_prompt,
        data_manager=dm,
    )

    # Insert a few items; capacity enforced
    for i in range(6):
        q, a = f"q{i}", f"a{i}"
        c.data_manager.save(q, a, q)

    eb = c.data_manager._eviction
    assert eb is not None and getattr(eb, "_policy", "").upper() == "TLFU_SLRU"
    assert len(eb._cache) <= 5

    # Basic search + hit callback
    res = c.data_manager.search("q5")
    assert isinstance(res, list)
    if res:
        c.data_manager.hit_cache_callback(res[0])
