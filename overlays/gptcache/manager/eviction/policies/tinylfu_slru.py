from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b
from typing import Any, Iterable, Tuple

from cachetools import Cache


# ------------------------ TinyLFU frequency estimator ------------------------
#
# We use a Count-Min Sketch (CMS):
# - 'depth' independent hash rows; each row has 'width' counters.
# - To increment(key): for each row, hash(key) -> column and ++counter[row][col].
# - To estimate(key): min(counter[row][hash(key)]) over all rows (standard CMS).
# - We apply periodic DECAY by halving all counters every 'decay_window' operations,
#   so old traffic gradually loses weight (prevents frequencies from growing unbounded).
#
# Notes:
# - CMS is approximate: it may overestimate but not underestimate (w.h.p).
# - Using blake2b with small digest + different row seeds gives fast, stable hashes.
# - Parameters (width, depth, decay_window) trade accuracy vs. memory/CPU.
#


class _CountMinSketch:
    def __init__(self, width: int = 2048, depth: int = 4, decay_window: int = 20_000) -> None:
        self.width = int(width)
        self.depth = int(depth)
        self.tables = [[0] * self.width for _ in range(self.depth)]
        # Per-row random-ish seeds (fixed strings are fine for determinism here)
        self.seeds = [f"tlfu-{i}".encode("utf-8") for i in range(self.depth)]
        self.ops = 0
        self.decay_window = int(decay_window)

    def _indices(self, key: Any) -> Iterable[int]:
        # Hash the repr of the key (works for most Python objects deterministically)
        raw = repr(key).encode("utf-8")
        for s in self.seeds:
            h = blake2b(raw + s, digest_size=8).digest()
            yield int.from_bytes(h, "little") % self.width

    def increment(self, key: Any) -> None:
        # Bump counters in each row
        for row, col in enumerate(self._indices(key)):
            t = self.tables[row]
            if t[col] < 2**31 - 1:
                t[col] += 1
        # Trigger periodic decay
        self.ops += 1
        if self.ops >= self.decay_window:
            self._decay()

    def estimate(self, key: Any) -> int:
        # CMS estimate is the minimum across rows
        est = None
        for row, col in enumerate(self._indices(key)):
            v = self.tables[row][col]
            est = v if est is None else min(est, v)
        return est or 0

    def _decay(self) -> None:
        # Halve every counter to age out stale popularity
        for r in range(self.depth):
            t = self.tables[r]
            for c in range(self.width):
                t[c] >>= 1
        self.ops = 0


# -------------------------- TinyLFU + SLRU composite -------------------------
#
# This is ONE cache object that:
# - Subclasses cachetools.Cache for compatibility (capacity accounting + popitem()).
# - Uses TinyLFU (CMS) to decide WHETHER to admit a new key when at capacity.
# - Stores admitted keys in a two-segment LRU (SLRU) with probation & protected.
#
class TinyLFUSLRUCache(Cache):
    """
    TinyLFU admission + SLRU storage (probation/protected), cachetools-compatible.

    Admission (only when at capacity):
      - Compare CMS(freq(new)) vs CMS(freq(victim)); admit iff new > victim.
      - If rejected, do not evict or insert (prevents cache pollution).

    Storage (SLRU):
      - New keys start in probation MRU.
      - Hit in probation -> promote to protected MRU.
      - Hit in protected -> refresh protected MRU.
      - Eviction target: probation LRU; if empty, protected LRU.

    We override __getitem__/__setitem__/__delitem__/popitem(), letting Cache.__setitem__
    handle size accounting and call our popitem() if over capacity.
    """

    def __init__(
        self,
        maxsize: int,
        getsizeof=None,
        *,
        protected_ratio: float = 0.8,
        cms_width: int = 2048,
        cms_depth: int = 4,
        cms_decay_window: int = 20_000,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        if not (0.0 < protected_ratio < 1.0):
            raise ValueError("protected_ratio must be in (0, 1)")

        super().__init__(maxsize, getsizeof)

        # Segment targets (guide promotion/demotion; actual eviction uses popitem()).
        prot_cap = max(1, int(maxsize * protected_ratio))
        prob_cap = max(1, maxsize - prot_cap)
        while prot_cap + prob_cap > maxsize and prot_cap > 1:
            prot_cap -= 1
        self._prot_cap = prot_cap
        self._prob_cap = prob_cap

        # Segment orderings (OrderedDict: LRU at left, MRU at right).
        self._prot: "OrderedDict[Any, None]" = OrderedDict()
        self._prob: "OrderedDict[Any, None]" = OrderedDict()

        # TinyLFU estimator (global popularity with decay).
        self._cms = _CountMinSketch(cms_width, cms_depth, cms_decay_window)

    # --------------------------- mapping overrides ---------------------------

    def __getitem__(self, key, cache_getitem=Cache.__getitem__):
        # Base mapping may raise KeyError; on hit we update recency & CMS.
        value = cache_getitem(self, key)
        self._cms.increment(key)  # count the access

        if key in self._prot:
            # Protected hit: refresh MRU
            self._prot.move_to_end(key, last=True)
        elif key in self._prob:
            # Probation hit: promote to protected MRU
            self._prob.pop(key, None)
            self._prot[key] = None
            self._prot.move_to_end(key, last=True)
            self._rebalance_protected()
        else:
            # Repair: mapping had key but segments lost it; reinsert into probation MRU
            self._prob[key] = None
            self._prob.move_to_end(key, last=True)
        return value

    def __setitem__(self, key, value, cache_setitem=Cache.__setitem__):
        # Count every write attempt to reflect traffic.
        self._cms.increment(key)

        present = key in self
        if present:
            # Overwrite behaves like access/promotion
            cache_setitem(self, key, value)
            if key in self._prot:
                self._prot.move_to_end(key, last=True)
            elif key in self._prob:
                self._prob.pop(key, None)
                self._prot[key] = None
                self._prot.move_to_end(key, last=True)
                self._rebalance_protected()
            else:
                self._prob[key] = None
                self._prob.move_to_end(key, last=True)
            return

        # New key
        if self.currsize < self.maxsize:
            # Not full: admit directly to probation MRU
            cache_setitem(self, key, value)
            self._prob[key] = None
            self._prob.move_to_end(key, last=True)
            return

        # At capacity: TinyLFU admission vs. would-be victim
        victim = self._choose_victim_key()
        if victim is not None:
            if self._cms.estimate(key) <= self._cms.estimate(victim):
                # Reject insert; keep current content (no eviction)
                return

        # Admit: delegate to base (may call our popitem() to evict the same victim)
        cache_setitem(self, key, value)
        self._prob[key] = None
        self._prob.move_to_end(key, last=True)

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        # Keep segments consistent with base mapping.
        self._prot.pop(key, None)
        self._prob.pop(key, None)
        cache_delitem(self, key)

    def popitem(self) -> Tuple[Any, Any]:
        # Choose SLRU victim: probation LRU else protected LRU.
        try:
            if self._prob:
                victim = next(iter(self._prob))
                self._prob.pop(victim, None)
            else:
                victim = next(iter(self._prot))
                self._prot.pop(victim, None)
        except StopIteration as exc:
            raise KeyError("popitem(): cache is empty") from exc

        value = super().pop(victim)  # updates Cache.currsize
        return victim, value

    # ------------------------------ helpers ----------------------------------

    def _choose_victim_key(self) -> Any | None:
        if self._prob:
            return next(iter(self._prob))
        if self._prot:
            return next(iter(self._prot))
        return None

    def _rebalance_protected(self) -> None:
        # If protected exceeds target, demote its LRU to probation MRU.
        while len(self._prot) > self._prot_cap:
            k, _ = self._prot.popitem(last=False)  # protected LRU
            self._prob.pop(k, None)
            self._prob[k] = None
            self._prob.move_to_end(k, last=True)

    # (optional) for tests / observability
    def segments(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        prot = tuple(reversed(tuple(self._prot.keys())))  # MRU..LRU
        prob = tuple(reversed(tuple(self._prob.keys())))  # MRU..LRU
        return prot, prob
