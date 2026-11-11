from app.core.ttl_cache import TTLCache


def test_expiry_and_eviction(clock):
    cache = TTLCache(ttl=10, maxsize=2, clock=clock)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None  # evicted (oldest)
    assert cache.get("b") == 2
    clock.now += 10
    assert cache.get("c") is None  # expired
