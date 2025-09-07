from __future__ import annotations

from cachetools import Cache
from collections.abc import Mapping
from heapq import heappush, heappop
from typing import Any, Callable, Optional, Tuple


def _default_cost_fn(key: Any, value: Any) -> float:
    """
    Default cost:
      - If value is a Mapping, prefer 'miss_penalty_ms', else 'latency_ms', else 'cost'; fallback 1.0.
      - Otherwise 1.0.
    """
    if isinstance(value, Mapping):
        raw = value.get("miss_penalty_ms", value.get("latency_ms", value.get("cost", 1.0)))
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1.0
    return 1.0


class GDSFCache(Cache):
    """
    GreedyDual-Size-Frequency (GDSF)

    Priority:
        H = L + (freq * cost / size)

    Where:
      - L (watermark/clock) := priority of the last evicted item (non-decreasing).
      - freq := per-key hit count (first insert starts at 1; hits do freq += 1).
      - cost := cost_fn(key, value) (default uses mapping fields as above).
      - size := size_fn(key, value) OR getsizeof(value) OR 1; must be positive.

    Implementation:
      - Min-heap of (priority, counter, key) with lazy invalidation.
      - On set/get we push a fresh entry; validity is checked against _handles.
      - popitem() evicts the lowest valid priority and updates L.
      - segments() provided for test parity (single segment: "all").
    """

    def __init__(
        self,
        maxsize: int,
        getsizeof: Optional[Callable[[Any], int]] = None,
        cost_fn: Optional[Callable[[Any, Any], float]] = None,
        size_fn: Optional[Callable[[Any, Any], int]] = None,
    ):
        if maxsize is None or maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        super().__init__(maxsize, getsizeof)
        self._cost_fn = cost_fn or _default_cost_fn
        self._size_fn = size_fn
        self._heap: list[Tuple[float, int, Any]] = []  # (priority, counter, key)
        self._handles: dict[Any, Tuple[float, int]] = {}  # key -> (priority, counter)
        self._freq: dict[Any, int] = {}  # key -> freq
        self._counter: int = 0
        self._L: float = 0.0  # watermark

    # ---------- internals ----------

    def _kv_cost_size(self, key: Any, value: Any) -> Tuple[float, int]:
        cost = float(self._cost_fn(key, value))
        size = (
            int(self._size_fn(key, value))
            if self._size_fn
            else (self.getsizeof(value) if self.getsizeof else 1)
        )
        if size <= 0:
            raise ValueError("size_fn/getsizeof must return a positive size")
        return cost, size

    def _priority(self, key: Any, value: Any) -> float:
        fr = self._freq.get(key, 1)
        cost, size = self._kv_cost_size(key, value)
        return self._L + (fr * (cost / size))

    def _push(self, key: Any, value: Any) -> None:
        pr = self._priority(key, value)
        self._counter += 1
        entry = (pr, self._counter, key)
        self._handles[key] = (pr, self._counter)
        heappush(self._heap, entry)

    # ---------- cachetools overrides ----------

    def __setitem__(self, key: Any, value: Any) -> None:
        data = self._Cache__data
        if key not in data:
            self._freq[key] = 1  # new key starts at frequency 1
        super().__setitem__(key, value)  # may trigger popitem()
        # On set, we only refresh priority (no freq bump)
        self._push(key, value)

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        self._freq[key] = self._freq.get(key, 1) + 1  # hit increases frequency
        self._push(key, value)
        return value

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        self._handles.pop(key, None)
        self._freq.pop(key, None)

    def clear(self) -> None:
        super().clear()
        self._heap.clear()
        self._handles.clear()
        self._freq.clear()
        self._counter = 0
        self._L = 0.0

    def popitem(self) -> Tuple[Any, Any]:
        """
        Evict the item with the lowest valid priority and update L.
        """
        data = self._Cache__data
        while self._heap:
            pr, _, k = heappop(self._heap)
            if k in data:
                handle = self._handles.get(k)
                if handle is not None and handle[0] == pr:
                    v = data[k]
                    self._L = pr
                    super().__delitem__(k)
                    self._handles.pop(k, None)
                    self._freq.pop(k, None)
                    return k, v
            # stale entry -> continue
        # Safety valve (should be rare)
        k, v = data.popitem()
        self._handles.pop(k, None)
        self._freq.pop(k, None)
        return k, v

    # ---------- test-parity utility ----------

    def segments(self) -> dict[str, set]:
        """Single-segment view for tests."""
        return {"all": set(self._Cache__data.keys())}
