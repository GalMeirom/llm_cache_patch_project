class sEvictionManager:
    """
    EvictionManager for in-memory dict-backed stores.

    Assumptions:
      - scalar_storage: dict[id -> CacheData-like payload]
      - vector_storage: dict[id -> np.ndarray (or any vector payload)]
    Behavior mirrors the original API: mark (soft evict), check_evict, delete (hard evict), rebuild.
    """

    MAX_MARK_COUNT = 5000
    MAX_MARK_RATE = 0.1
    BATCH_SIZE = 100000
    REBUILD_CONDITION = 5

    def __init__(self, scalar_storage: dict, vector_storage: dict):
        self._scalar_storage = scalar_storage
        self._vector_storage = vector_storage
        self._deleted: set[int] = set()  # ids marked for deletion (soft-evicted)
        self.delete_count = 0

    # ---- policy helpers ----

    def check_evict(self) -> bool:
        """Return True if the amount/rate of soft-evicted ids justifies a hard delete."""
        mark_count = len(self._deleted)
        all_count = len(self._scalar_storage)
        if all_count == 0:
            return False
        return (mark_count > self.MAX_MARK_COUNT) or (mark_count / all_count > self.MAX_MARK_RATE)

    # ---- lifecycle operations ----

    def soft_evict(self, marked_keys) -> None:
        """Mark ids for deletion; does not remove them yet."""
        if marked_keys is None:
            return
        try:
            iterable = (
                marked_keys
                if hasattr(marked_keys, "__iter__") and not isinstance(marked_keys, str | bytes)
                else [marked_keys]
            )
        except TypeError:
            iterable = [marked_keys]
        for k in iterable:
            if k in self._scalar_storage:
                self._deleted.add(int(k))

    def delete(self) -> None:
        """Hard-delete all currently marked ids from both storages; maybe trigger rebuild."""
        if not self._deleted:
            return

        # hard remove from scalar + vector
        for k in list(self._deleted):
            self._scalar_storage.pop(k, None)
            self._vector_storage.pop(k, None)
        self._deleted.clear()

        # bookkeeping + optional rebuild
        self.delete_count += 1
        if self.delete_count >= self.REBUILD_CONDITION:
            self.rebuild()

    def rebuild(self) -> None:
        """
        Rebuild vector index from remaining scalar ids.
        With dicts, this is effectively a no-op except to sync keys and reset counters.
        """
        # Keep only vectors whose ids still exist in scalar storage.
        valid_ids = set(self._scalar_storage.keys())
        for k in list(self._vector_storage.keys()):
            if k not in valid_ids:
                self._vector_storage.pop(k, None)

        # reset delete counter after a rebuild
        self.delete_count = 0
