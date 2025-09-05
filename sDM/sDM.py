"""
s_data_manager.py

A fast, local, in-memory DataManager implementation for GPTCache-like systems.

Design goals
------------
1) Use a real EvictionBase (SLRU / GDSF / etc.) to drive which keys stay hot.
2) Keep both "scalar" records and "vector" embeddings in lightweight local dicts
   to enable very fast, repeatable benchmarks without external I/O.
3) Provide a minimal "encryption" layer for string fields so payloads are not
   stored as plain text (XOR over UTF-8 bytes; reversible and cheap).
4) Implement the full DataManager API expected by higher layers.

Notes
-----
- Vector similarity is cosine (dot product on L2-normalized vectors).
- `get_scalar_data` returns decrypted strings; data at rest remains "encrypted".
- Session helpers mimic the behavior of SSDataManager enough for benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# If you keep DataManager in another module, import it:
# from gptcache.manager.data_manager_base import DataManager
# For the user's shared code, DataManager is defined alongside other managers.
from gptcache.manager.data_manager import DataManager  # adjust if needed

# === External contracts (from GPTCache) ===
# These imports assume you are within a project that exposes these symbols.
# If your paths differ, adjust imports accordingly.
from gptcache.manager.eviction import EvictionBase
from gptcache.manager.eviction.distributed_cache import NoOpEviction
from gptcache.manager.scalar_data.base import (
    Answer,
    CacheData,
    DataType,
    Question,
)
from gptcache.utils.error import ParamError

from .sEM import sEvictionManager

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalized float32 view of `vec`. If zero-norm, return original."""
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / norm) if norm > 0 else v


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """
    Very small, reversible transformation for strings (NOT cryptographically secure).
    Used here to avoid storing clear-text strings in memory during benchmarks.
    """
    if not key:
        return data
    klen = len(key)
    return bytes(b ^ key[i % klen] for i, b in enumerate(data))


def _on_evict(self, marked_keys):
    """EvictionBase callback: delete local state for the evicted key(s)."""
    if marked_keys is None:
        return

    # accept single key or iterable of keys
    try:
        keys = (
            list(marked_keys)
            if not isinstance(marked_keys, int | np.integer)
            else [int(marked_keys)]
        )
    except TypeError:
        keys = [marked_keys]

    for k in keys:
        self._scalar.pop(k, None)
        self._vectors.pop(k, None)
        self._sessions.pop(k, None)
        self._session_meta.pop(k, None)


@dataclass
class _SessionRecord:
    """Lightweight record to mirror session linkage similar to SSDataManager."""

    id: int
    session_id: str | None
    session_question: str | None


# ---------------------------------------------------------------------------
# sDataManager
# ---------------------------------------------------------------------------


class sDataManager(DataManager):
    """
    sDataManager(DataManager)

    A hybrid in-memory DataManager:
      • Eviction: controlled by EvictionBase (policy + callback on evict).
      • Scalar store: local dict (id -> CacheData) with XOR "encryption" for strings.
      • Vector store: local dict (id -> normalized float32 vector) with cosine search.

    Parameters
    ----------
    e : Optional[EvictionBase], default=None
        External eviction controller. If None, a memory EvictionBase is constructed
        with the provided `max_size`, `clean_size`, `policy`, wired to `_on_evict`.
    max_size : int, default=1000
        Target capacity hint for the eviction layer (number of ids).
    clean_size : Optional[int], default=None
        How many to evict on pressure (policy-dependent).
    policy : str, default="LRU"
        The policy name for the EvictionBase (e.g., "LRU", "SLRU", "GDSF"...).
    enc_key : bytes, default=b"gptcache"
        Key used by the simple XOR transform for string fields at rest.

    Rationale
    ---------
    This manager enables rapid benchmark loops: no DBs, no vector index services,
    deterministic similarity (cosine over normalized vectors), and a realistic
    eviction surface via EvictionBase.
    """

    # ----------------------------- construction -----------------------------------

    def __init__(
        self,
        e: EvictionBase | None = None,
        *,
        max_size: int = 1000,
        clean_size: int | None = None,
        policy: str = "LRU",
        enc_key: bytes = b"gptcache",
    ) -> None:
        # 1) Eviction surface: construct if not provided
        if e is None:
            e = EvictionBase(
                name="memory",
                maxsize=max_size,
                clean_size=clean_size,
                policy=policy,
                on_evict=self._clear,  # will be called with a list of ids to drop
            )
        self._eviction: EvictionBase = e

        # 2) Local, in-memory stores
        self._next_id: int = 1
        self._scalar: dict[int, CacheData] = {}  # id -> encrypted-at-rest CacheData
        self._vectors: dict[int, np.ndarray] = {}  # id -> normalized float32 vector
        self.eviction_manager = sEvictionManager(self._scalar, self._vectors)
        self._sessions: dict[int, set[str]] = {}  # id -> set(session_id)
        self._session_meta: dict[int, list[_SessionRecord]] = {}  # id -> session records
        self._enc_key = enc_key

        # If eviction comes pre-populated, we could hydrate here. For local runs,
        # simply ensure it is aware we currently own no ids (noop for NoOpEviction).
        if not isinstance(self._eviction, NoOpEviction):
            self._eviction.put([])

    # ----------------------------- encryption helpers ------------------------------

    def _enc_str(self, s: str) -> bytes:
        """Encode UTF-8, XOR with key → bytes."""
        return _xor_bytes(s.encode("utf-8"), self._enc_key)

    def _dec_str(self, b: bytes) -> str:
        """Reverse XOR and decode UTF-8 → str."""
        return _xor_bytes(b, self._enc_key).decode("utf-8")

    def _maybe_enc_question(self, q: str | Question) -> bytes | Question:
        """
        For plain string questions, store encrypted bytes at rest.
        If upstream gives a structured `Question`, keep it as-is.
        """
        if isinstance(q, Question):
            return q
        return self._enc_str(q)

    def _maybe_dec_question(self, q: bytes | Question) -> str | Question:
        """Return decrypted str for bytes; pass through Question."""
        if isinstance(q, Question):
            return q
        return self._dec_str(q)

    def _maybe_enc_answers(self, ans: str | Answer | list[str | Answer]) -> Answer | list[Answer]:
        """
        Always store string-ish answers as Answer(DataType.STR, <bytes>).
        Never return raw bytes to CacheData (avoids Pydantic iterating bytes→ints).
        """
        if isinstance(ans, list):
            out: list[Answer] = []
            for a in ans:
                if isinstance(a, Answer):
                    if a.answer_type == DataType.STR and isinstance(a.answer, str):
                        out.append(Answer(self._enc_str(a.answer), DataType.STR))
                    else:
                        out.append(a)
                elif isinstance(a, str):
                    out.append(Answer(self._enc_str(a), DataType.STR))
                else:
                    # best effort
                    out.append(Answer(a, DataType.STR))
            return out

        if isinstance(ans, Answer):
            if ans.answer_type == DataType.STR and isinstance(ans.answer, str):
                return Answer(self._enc_str(ans.answer), DataType.STR)
            return ans

        if isinstance(ans, str):
            return Answer(self._enc_str(ans), DataType.STR)

        # fallback
        return Answer(ans, DataType.STR)

    def _maybe_dec_answers(
        self, ans: Answer | list[Answer]
    ) -> Answer | list[Answer] | str | list[str]:
        """
        Reverse storage form:
          - Answer(STR, bytes) -> Answer(STR, str)
          - List[...] -> elementwise
        """
        if isinstance(ans, list):
            out: list[Answer | str] = []
            for a in ans:
                if (
                    isinstance(a, Answer)
                    and a.answer_type == DataType.STR
                    and isinstance(a.answer, bytes | bytearray)
                ):
                    out.append(Answer(self._dec_str(a.answer), DataType.STR))
                else:
                    out.append(a)
            return out
        if (
            isinstance(ans, Answer)
            and ans.answer_type == DataType.STR
            and isinstance(ans.answer, bytes | bytearray)
        ):
            return Answer(self._dec_str(ans.answer), DataType.STR)
        return ans

    def _alloc_ids(self, n: int) -> list[int]:
        """Allocate `n` monotonically increasing integer ids."""
        ids = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        return ids

    # ----------------------------- DataManager API ---------------------------------

    def _clear(self, marked_keys):
        self.eviction_manager.soft_evict(marked_keys)
        if self.eviction_manager.check_evict():
            self.eviction_manager.delete()

    def save(self, question, answer, embedding_data, **kwargs):
        """
        Insert a single (question, answer, embedding) triplet.

        Parameters
        ----------
        question : Union[str, Question]
        answer   : Union[str, Answer, List[Union[str, Answer]]]
        embedding_data : np.ndarray / Iterable[float]
        session : Optional[Session] in kwargs (with `.name`), default None
        """
        session = kwargs.get("session", None)
        session_id = session.name if session else None
        self.import_data([question], [answer], [embedding_data], [session_id])

    def import_data(
        self,
        questions: list[Any],
        answers: list[Any],
        embedding_datas: list[Any],
        session_ids: list[str | None],
    ):
        """
        Batch insert API.

        All input lists must be the same length. For each row:
          - Normalize embedding vector to float32, L2=1.
          - Encrypt string-ish fields for at-rest storage.
          - Create CacheData and store under a newly assigned id.
          - Register session metadata if provided.
          - Inform eviction policy of the new id.

        Raises
        ------
        ParamError : if input list lengths differ.
        """
        if not (len(questions) == len(answers) == len(embedding_datas) == len(session_ids)):
            raise ParamError("Make sure that all parameters have the same length")

        ids = self._alloc_ids(len(questions))

        for i, id_ in enumerate(ids):
            vec = _normalize(np.asarray(embedding_datas[i], dtype=np.float32))
            q_enc = self._maybe_enc_question(questions[i])
            a_enc = self._maybe_enc_answers(answers[i])

            # Persist in-memory
            self._vectors[id_] = vec
            self._scalar[id_] = CacheData(
                question=q_enc,  # bytes (encrypted) or Question
                answers=a_enc,  # bytes / Answer / list (encrypted when STR)
                embedding_data=vec,  # normalized float32
                session_id=session_ids[i],  # optional
            )

            # Session links
            if session_ids[i]:
                self._sessions.setdefault(id_, set()).add(session_ids[i])
                self._session_meta.setdefault(id_, []).append(
                    _SessionRecord(
                        id=id_,
                        session_id=session_ids[i],
                        session_question=questions[i] if isinstance(questions[i], str) else None,
                    )
                )

            # Eviction sees this id as present
            if not isinstance(self._eviction, NoOpEviction):
                self._eviction.put([id_])

    def get_scalar_data(self, res_data, **kwargs) -> CacheData | None:
        session = kwargs.get("session", None)
        if not res_data or len(res_data) < 2:
            return None
        id_ = res_data[1]
        cd = self._scalar.get(id_)
        if cd is None:
            return None

        if session:
            cache_session_ids = list(self._sessions.get(id_, set()))
            cache_questions = [r.session_question for r in self._session_meta.get(id_, [])]
            # representative textual answer (for gate)
            rep = None
            if isinstance(cd.answers, list) and cd.answers:
                a0 = cd.answers[0]
                if isinstance(a0, Answer) and a0.answer_type == DataType.STR:
                    rep = (
                        self._dec_str(a0.answer)
                        if isinstance(a0.answer, bytes | bytearray)
                        else a0.answer
                    )
            elif isinstance(cd.answers, Answer) and cd.answers.answer_type == DataType.STR:
                rep = (
                    self._dec_str(cd.answers.answer)
                    if isinstance(cd.answers.answer, bytes | bytearray)
                    else cd.answers.answer
                )
            if not session.check_hit_func(session.name, cache_session_ids, cache_questions, rep):
                return None

        # decrypt answers
        dec = self._maybe_dec_answers(cd.answers)

        # normalize to plain strings when possible
        def _norm(a):
            if isinstance(a, Answer) and a.answer_type == DataType.STR:
                return a.answer
            return a

        if isinstance(dec, Answer):
            out_answers = _norm(dec)
        elif isinstance(dec, list):
            if len(dec) == 1 and isinstance(dec[0], Answer) and dec[0].answer_type == DataType.STR:
                out_answers = dec[0].answer
            else:
                out_answers = [_norm(x) for x in dec]
        else:
            out_answers = dec

        return CacheData(
            question=self._maybe_dec_question(cd.question),
            answers=out_answers,
            embedding_data=cd.embedding_data,
            session_id=cd.session_id,
        )

    def hit_cache_callback(self, res_data, **kwargs):
        """
        Notify the eviction layer that an id has been accessed (for recency/frequency).
        `res_data` is expected as [score, id].
        """
        if res_data and len(res_data) >= 2 and not isinstance(self._eviction, NoOpEviction):
            self._eviction.get(res_data[1])

    def search(self, embedding_data, **kwargs):
        """
        Cosine similarity search over in-memory vectors.

        Parameters
        ----------
        embedding_data : np.ndarray / Iterable[float]
            Query vector; will be normalized to float32 internally.
        top_k : int, optional (kwargs)
            If > 0, return only the top k hits; otherwise return all.

        Returns
        -------
        List[List[float, int]]
            A list of [score, id] rows sorted by descending score.
        """
        if not self._vectors:
            return []

        top_k = kwargs.get("top_k", -1)
        q = _normalize(np.asarray(embedding_data, dtype=np.float32))

        # Build an array of ids and a matrix of vectors in the same order
        ids = np.fromiter(self._vectors.keys(), dtype=np.int64)
        if ids.size == 0:
            return []

        mat = np.stack([self._vectors[int(i)] for i in ids], axis=0)  # (N, D)
        scores = mat @ q  # cosine similarity because both sides are normalized

        order = np.argsort(scores)[::-1]
        if isinstance(top_k, int) and top_k > 0:
            order = order[:top_k]

        return [[float(scores[i]), int(ids[i])] for i in order]

    def flush(self):
        """
        No-op for in-memory manager.
        Provided for API symmetry with persistent managers.
        """
        pass

    def add_session(self, res_data, session_id, pre_embedding_data):
        """
        Attach a `session_id` to a specific result (id from `res_data`).

        Parameters
        ----------
        res_data : Sequence
            Expected form: [score, id]
        session_id : str
            Session identifier to associate with this id.
        pre_embedding_data : Any
            Kept for API compatibility; unused here.
        """
        if not res_data or len(res_data) < 2:
            return
        id_ = res_data[1]
        self._sessions.setdefault(id_, set()).add(session_id)
        self._session_meta.setdefault(id_, []).append(
            _SessionRecord(id=id_, session_id=session_id, session_question=None)
        )

    def list_sessions(self, session_id=None, key=None):
        """
        Flexible session enumeration.

        Parameters
        ----------
        session_id : Optional[str]
            If provided, return the list of ids that contain this session id.
        key : Optional[int]
            If provided, return the session records for a single id.

        Returns
        -------
        List[Any]
            - If `key` is provided: list of `_SessionRecord` for that id.
            - Else if `session_id` is provided: list of ids containing that session_id.
            - Else: list of all distinct session ids.
        """
        if key is not None:
            return list(self._session_meta.get(key, []))
        if session_id is not None:
            return [k for k, s in self._sessions.items() if session_id in s]
        # return all distinct session ids
        all_ids = set()
        for s in self._sessions.values():
            all_ids.update(s)
        return list(all_ids)

    def delete_session(self, session_id):
        """
        Remove `session_id` from all keys. If a key becomes session-empty,
        delete the underlying data as well (simulates GC for session-scoped cache).
        """
        to_delete: list[int] = []
        for k, s in self._sessions.items():
            if session_id in s:
                s.discard(session_id)
                # prune meta records for this session
                self._session_meta[k] = [
                    r for r in self._session_meta.get(k, []) if r.session_id != session_id
                ]
                # schedule full record deletion if no sessions remain
                if not s:
                    to_delete.append(k)

        for k in to_delete:
            self._sessions.pop(k, None)
            self._session_meta.pop(k, None)
            self._scalar.pop(k, None)
            self._vectors.pop(k, None)
            # Best-effort eviction side cleanup if policy supports it
            if not isinstance(self._eviction, NoOpEviction):
                try:
                    self._eviction.delete([k])  # may be a no-op depending on policy
                except Exception:
                    pass  # keep benchmarks resilient

    def close(self):
        """Release in-memory resources and shut down eviction (best-effort)."""
        # Collect ids before clearing (for eviction cleanup)
        ids = (
            set(self._scalar.keys())
            | set(self._vectors.keys())
            | set(self._sessions.keys())
            | set(self._session_meta.keys())
        )

        # Best-effort notify eviction
        try:
            if not isinstance(self._eviction, NoOpEviction) and ids:
                # remove all currently tracked keys from the eviction structure
                if hasattr(self._eviction, "delete"):
                    self._eviction.delete(list(ids))
            if hasattr(self._eviction, "close"):
                self._eviction.close()  # optional in some implementations
        except Exception:
            pass  # never raise on close

        # In-memory stores: clear everything
        self._scalar.clear()
        self._vectors.clear()
        self._sessions.clear()
        self._session_meta.clear()
        self._next_id = 1

        # Keep API symmetry
        try:
            self.flush()
        except Exception:
            pass
