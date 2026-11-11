import time
from collections import OrderedDict
from collections.abc import Callable


class TTLCache[K, V]:
    """A tiny in-memory cache whose entries expire after `ttl` seconds.

    Bounded to `maxsize` entries; the oldest entry is evicted first. Per process only.
    """

    def __init__(
        self, ttl: float, maxsize: int = 10_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        self._clock = clock
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()

    def get(self, key: K) -> V | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._data[key]
            return None
        return value

    def set(self, key: K, value: V) -> None:
        self._data.pop(key, None)
        self._data[key] = (self._clock() + self.ttl, value)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)
