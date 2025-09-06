# tests/test_s_data_manager.py
# PyTest for sDataManager, plus an integration test that uses sDataManager *inside* GPTCache.Cache.

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from typing import Any

import numpy as np
import pytest
from gptcache.manager.scalar_data.base import DataType

# import the class under test
from sDM.sDM import sDataManager  # noqa: E402


class FakeSession:
    def __init__(self, name: str):
        self.name = name

    def check_hit_func(
        self,
        session_name: str,
        cache_session_ids: list[str | None],
        cache_questions: list[str | None],
        cache_answer: Any,
    ) -> bool:
        return session_name in (cache_session_ids or [])


def _vec(x: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed=x)
    return rng.random(dim, dtype=np.float32)


def test_import_and_search_basic():
    """Save two items, search by vector, ensure top hit and decrypted fields are correct."""
    dm = sDataManager(max_size=10, policy="LRU")

    q1, a1, v1 = "what is A?", "answer A", "what is A?"
    q2, a2, v2 = "what is B?", "answer B", "what is B?"

    dm.import_data([q1, q2], [a1, a2], [v1, v2], [None, None])

    res = dm.search(v1, top_k=1)
    assert res and len(res[0]) == 2
    score, id1 = res[0]
    assert 0.0 <= score <= 1.0

    cd = dm.get_scalar_data([score, id1])
    assert cd is not None
    assert isinstance(cd.question, str) and cd.question == q1
    assert cd.answers[0].answer_type == DataType.STR and cd.answers[0].answer == a1


def test_encryption_at_rest():
    """Verify strings are not stored in clear text inside the in-memory store."""
    dm = sDataManager(enc_key=b"k")
    q, a, v = "secret question", "secret answer", "secret question"
    dm.import_data([q], [a], [v], [None])

    id_ = next(iter(dm._scalar.keys()))
    stored = dm._scalar[id_]
    assert isinstance(stored.question, bytes | bytearray)
    if isinstance(stored.answers, list):
        first = stored.answers[0]
        assert not isinstance(first, str)
    else:
        assert isinstance(stored.answers, bytes | bytearray)


def test_session_gate_allows_and_blocks():
    """With a session provided: allowed if names match, blocked otherwise."""
    dm = sDataManager()
    q, a, v = "sessioned Q", "sessioned A", "sessioned Q"
    dm.import_data([q], [a], [v], ["S1"])

    res = dm.search(v, top_k=1)
    assert res
    score, id1 = res[0]

    s_ok = FakeSession("S1")
    cd_ok = dm.get_scalar_data([score, id1], session=s_ok)
    assert cd_ok is not None and cd_ok.answers[0].answer == a

    s_bad = FakeSession("S2")
    cd_bad = dm.get_scalar_data([score, id1], session=s_bad)
    assert cd_bad is None


def test_hit_cache_callback_and_eviction_callback():
    """Touching an id should not error; eviction callback should remove data."""
    dm = sDataManager(max_size=2, clean_size=1)
    q1, a1, v1 = "Q1", "A1", "Q1"
    q2, a2, v2 = "Q2", "A2", "Q2"
    dm.import_data([q1, q2], [a1, a2], [v1, v2], [None, None])

    r1 = dm.search(v1, top_k=1)[0]
    dm.hit_cache_callback(r1)  # should not raise

    r2 = dm.search(v2, top_k=1)[0]
    id2 = r2[1]
    q3, a3, v3 = "Q3", "A3", "Q3"
    dm.save(q3, a3, v3)  # should evict id2

    assert dm.get_scalar_data(r2) is None
    assert id2 not in dm._vectors


def test_add_list_delete_sessions():
    """Add a session to an id, list by session and by key,
    then delete session and auto-GC if last."""
    dm = sDataManager()
    q, a, v = "Q", "A", "Q"
    dm.import_data([q], [a], [v], [None])

    res = dm.search(v, top_k=1)
    score, id1 = res[0]

    dm.add_session([score, id1], "S9", None)
    assert id1 in dm.list_sessions(session_id="S9")
    records = dm.list_sessions(key=id1)
    assert records and records[0].session_id == "S9"

    dm.delete_session("S9")
    assert id1 not in dm._scalar
    assert id1 not in dm._vectors
    assert not dm.list_sessions(session_id="S9")


@pytest.mark.skipif(
    pytest.importorskip("gptcache", reason="gptcache not installed") is None,
    reason="gptcache not installed",
)
def test_cache_with_sdm_integration():
    """
    Use sDataManager *inside* GPTCache.Cache:
      - Create Cache()
      - Assign sDataManager as its data_manager
      - Insert data via cache.data_manager
      - Search and fetch via cache.data_manager
    """
    from gptcache import Cache  # type: ignore
    from gptcache.adapter.langchain_models import LangChainLLMs
    from gptcache.processor.pre import get_prompt

    from sLLM.synthetic_llm import SyntheticLLM  # type: ignore

    dm = sDataManager(max_size=2, policy="LRU", clean_size=1)
    cache = Cache()
    cache.init(pre_embedding_func=get_prompt, data_manager=dm)
    llm = LangChainLLMs(llm=SyntheticLLM())
    q1 = "What is A?"
    q2 = "What is B?"
    q3 = "What is C?"
    a1 = llm(q1, cache_obj=cache)
    llm(q2, cache_obj=cache)
    a3 = llm(q1, cache_obj=cache)
    assert a1 == a3
    llm(q3, cache_obj=cache)
    try:
        cache.data_manager._eviction._cache.__getitem__(2)
    except KeyError as e:
        assert e.args == (2,)
