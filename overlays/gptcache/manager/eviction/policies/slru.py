# overlays/gptcache/manager/eviction/policies/slru.py
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable, Iterable, Tuple, TypeVar, Generic, Optional

from cachetools import Cache

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class SLRUCache(Cache, Generic[K, V]):
    """
    Segmented LRU cache (SLRU), compatible with cachetools' Cache API.

    - Two segments:
        * probation : receives new items (LRU inside the segment)
        * protected : holds promoted/hot items (LRU inside the segment)
    - On hit in probation  -> promote key to protected MRU.
    - On hit in protected  -> refresh key to protected MRU.
    - Eviction: prefer the LRU of probation; if empty, evict the LRU of protected.

    Notes:
      * We inherit capacity/size handling from `cachetools.Cache`. Its `__setitem__`
        will call our `popitem()` when `currsize > maxsize`, so we only need to
        choose a victim and return `(key, value)` there.
      * Segment balancing: when promoting into protected and it exceeds its target
        capacity, we *demote* the protected-LRU back into probation MRU.
    """

    def __init__(
        self,
        maxsize: int,
        getsizeof=None,
        *,
        protected_ratio: float = 0.8,
        **kwargs: Any,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        if not (0.0 < protected_ratio < 1.0):
            raise ValueError("protected_ratio must be in (0, 1)")
        super().__init__(maxsize, getsizeof, **kwargs)

        # Segment targets (do not strictly evict; they guide promotion/demotion).
        prot_cap = max(1, int(maxsize * protected_ratio))
        prob_cap = max(1, maxsize - prot_cap)
        # Ensure sum does not exceed maxsize due to rounding.
        while prot_cap + prob_cap > maxsize and prot_cap > 1:
            prot_cap -= 1

        self._prot_cap = prot_cap
        self._prob_cap = prob_cap

        # OrderedDict with MRU at right, LRU at left.
        self._prot: "OrderedDict[K, None]" = OrderedDict()
        self._prob: "OrderedDict[K, None]" = OrderedDict()

    # ---------- mapping overrides ----------

    def __getitem__(self, key: K, cache_getitem=Cache.__getitem__) -> V:
        value = cache_getitem(self, key)  # may raise KeyError
        # refresh membership/recency by SLRU rules
        if key in self._prot:
            self._prot.move_to_end(key, last=True)  # MRU
        elif key in self._prob:
            # promote to protected MRU
            self._prob.pop(key, None)
            self._prot[key] = None
            self._prot.move_to_end(key, last=True)
            self._rebalance_protected()
        else:
            # Repair membership: if present in base mapping but lost in segments,
            # treat as new probation MRU.
            self._prob[key] = None
            self._prob.move_to_end(key, last=True)
        return value

    def __setitem__(self, key: K, value: V, cache_setitem=Cache.__setitem__) -> None:
        present = key in self
        # First, write to the underlying mapping (this may call popitem()).
        cache_setitem(self, key, value)

        if present:
            # Treat overwrite as access (promotion/refresh).
            if key in self._prot:
                self._prot.move_to_end(key, last=True)
            elif key in self._prob:
                self._prob.pop(key, None)
                self._prot[key] = None
                self._prot.move_to_end(key, last=True)
                self._rebalance_protected()
            else:
                # Repair: if base has it but not in segments, put in probation MRU.
                self._prob[key] = None
                self._prob.move_to_end(key, last=True)
        else:
            # New item starts in probation MRU.
            self._prob[key] = None
            self._prob.move_to_end(key, last=True)
            # No explicit trim here; base will call popitem() if over maxsize.

    def __delitem__(self, key: K, cache_delitem=Cache.__delitem__) -> None:
        # Remove from segments first (avoid dangling membership).
        self._prot.pop(key, None)
        self._prob.pop(key, None)
        cache_delitem(self, key)

    # ---------- eviction policy hook ----------

    def popitem(self) -> Tuple[K, V]:
        """
        Choose a victim according to SLRU and remove it from the cache.
        Returns: (key, value)
        """
        try:
            if self._prob:
                # Evict probation LRU first
                victim = next(iter(self._prob))
                self._prob.pop(victim, None)
            else:
                # If probation empty, evict protected LRU
                victim = next(iter(self._prot))
                self._prot.pop(victim, None)
        except StopIteration as exc:  # both segments empty
            raise KeyError("popitem(): cache is empty") from exc

        # Remove from base mapping and return value (updates currsize in base).
        value = super().pop(victim)
        return victim, value

    # ---------- helpers ----------

    def _rebalance_protected(self) -> None:
        """If protected exceeds its target size, demote its LRU into probation MRU."""
        while len(self._prot) > self._prot_cap:
            # Demote protected LRU to probation MRU (move-to-end).
            k, _ = self._prot.popitem(last=False)  # LRU of protected
            # Avoid duplicates in probation.
            self._prob.pop(k, None)
            self._prob[k] = None
            self._prob.move_to_end(k, last=True)

    # ---------- optional introspection (useful for tests) ----------

    def segments(self) -> Tuple[Tuple[K, ...], Tuple[K, ...]]:
        """
        Return (protected_MRU_to_LRU, probation_MRU_to_LRU) as tuples of keys.
        """
        prot = tuple(reversed(tuple(self._prot.keys())))  # MRU..LRU
        prob = tuple(reversed(tuple(self._prob.keys())))  # MRU..LRU
        return prot, prob
