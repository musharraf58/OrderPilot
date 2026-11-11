import pytest

from tests.conftest import AUTH, TOKEN
from tests.factories import CUSTOMER_ID, me


async def test_login_forwards_to_store_and_returns_token(api, store):
    route = store.post("/auth/login").respond(
        200, json={"access_token": "store-access", "refresh_token": "store-refresh"}
    )
    r = await api.post("/auth/login", json={"email": "a@b.com", "password": "pw"})
    assert r.status_code == 200
    assert r.json() == {"access_token": "store-access", "token_type": "bearer"}
    assert route.calls.last.request.read() == b'{"email":"a@b.com","password":"pw"}'


async def test_login_bad_credentials_is_401(api, store):
    store.post("/auth/login").respond(401, json={"detail": "Incorrect email or password"})
    r = await api.post("/auth/login", json={"email": "a@b.com", "password": "nope"})
    assert r.status_code == 401


async def test_login_store_down_is_503(api, store):
    store.post("/auth/login").respond(503)
    r = await api.post("/auth/login", json={"email": "a@b.com", "password": "pw"})
    assert r.status_code == 503


async def test_missing_token_is_401(api):
    r = await api.get("/auth/me")
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("status", [401, 403, 422])
async def test_invalid_token_is_401_and_not_cached(api, store, status):
    route = store.get("/auth/me").respond(status)
    for _ in range(2):
        r = await api.get("/auth/me", headers={"Authorization": "Bearer bad"})
        assert r.status_code == 401
    assert route.call_count == 2


async def test_store_down_during_auth_is_503(api, store):
    store.get("/auth/me").respond(503)
    r = await api.get("/auth/me", headers=AUTH)
    assert r.status_code == 503


async def test_valid_token_builds_context_from_auth_me(api, store):
    route = store.get("/auth/me").respond(200, json=me())
    r = await api.get("/auth/me", headers=AUTH)
    assert r.json() == {"customer_id": str(CUSTOMER_ID)}
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"


async def test_context_is_cached_until_ttl_expires(api, store, clock):
    route = store.get("/auth/me").respond(200, json=me())
    for _ in range(3):
        assert (await api.get("/auth/me", headers=AUTH)).status_code == 200
    assert route.call_count == 1
    clock.now += 61
    assert (await api.get("/auth/me", headers=AUTH)).status_code == 200
    assert route.call_count == 2


async def test_cache_is_per_token(api, store):
    route = store.get("/auth/me").respond(200, json=me())
    await api.get("/auth/me", headers=AUTH)
    await api.get("/auth/me", headers={"Authorization": "Bearer other-token"})
    assert route.call_count == 2
