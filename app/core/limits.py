"""In-memory protection for the messages endpoint. Per process: with several workers each
one keeps its own counters (fine for this project's single worker)."""

import math
import time
from collections import defaultdict, deque
from collections.abc import Callable, Hashable


class RateLimiter:
    """Sliding window: at most `limit` hits per key within `window_s` seconds."""

    def __init__(
        self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: defaultdict[Hashable, deque[float]] = defaultdict(deque)

    def hit(self, key: Hashable) -> int | None:
        """Record a hit. Returns None if allowed, or the seconds to wait (Retry-After)."""
        now = self._clock()
        hits = self._hits[key]
        while hits and hits[0] <= now - self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(1, math.ceil(hits[0] + self.window_s - now))
        hits.append(now)
        return None


class RunLocks:
    """At most one active run per key (conversation).

    A lock is released when its run ends. As a safety net a lock older than `stale_after`
    seconds is treated as free, so a run whose cleanup never happened (e.g. the client
    disconnected before the stream started) cannot block a conversation forever.
    """

    def __init__(self, stale_after: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.stale_after = stale_after
        self._clock = clock
        self._active: dict[Hashable, float] = {}

    def try_acquire(self, key: Hashable) -> bool:
        # No await in here, so check-and-set is atomic on the event loop.
        now = self._clock()
        started = self._active.get(key)
        if started is not None and now - started < self.stale_after:
            return False
        self._active[key] = now
        return True

    def release(self, key: Hashable) -> None:
        self._active.pop(key, None)
