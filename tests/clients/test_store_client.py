from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import respx

from app.clients.store import (
    StoreClient,
    StoreConflict,
    StoreError,
    StoreForbidden,
    StoreNotFound,
    StoreUnauthorized,
    StoreUnavailable,
)
from tests.factories import BASE_URL, me, order


@pytest.fixture
def store():
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
def sleep():
    return AsyncMock()


@pytest.fixture
async def client(sleep):
    async with StoreClient(BASE_URL, sleep=sleep) as c:
        yield c


async def test_sends_bearer_token_per_call(store, client):
    route = store.get("/auth/me").respond(200, json=me())
    await client.get_me("tok-a")
    await client.get_me("tok-b")
    assert [c.request.headers["Authorization"] for c in route.calls] == [
        "Bearer tok-a",
        "Bearer tok-b",
    ]


async def test_retries_503_then_succeeds(store, client, sleep):
    route = store.get("/orders").mock(
        side_effect=[httpx.Response(503), httpx.Response(503), httpx.Response(200, json=[order()])]
    )
    orders = await client.list_orders("tok")
    assert len(orders) == 1
    assert route.call_count == 3
    # Exponential backoff between attempts.
    assert [c.args[0] for c in sleep.await_args_list] == [0.5, 1.0]


async def test_retries_connect_error_then_succeeds(store, client):
    route = store.get("/orders").mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json=[])]
    )
    assert await client.list_orders("tok") == []
    assert route.call_count == 2


async def test_gives_up_after_max_attempts(store, client):
    route = store.get("/orders").respond(502)
    with pytest.raises(StoreUnavailable):
        await client.list_orders("tok")
    assert route.call_count == 3


@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (401, StoreUnauthorized),
        (403, StoreForbidden),
        (404, StoreNotFound),
        (409, StoreConflict),
        (422, StoreError),
    ],
)
async def test_4xx_maps_to_typed_error_without_retry(store, client, status, exc):
    route = store.get("/orders").respond(status, json={"detail": "nope"})
    with pytest.raises(exc) as info:
        await client.list_orders("tok")
    assert type(info.value) is exc
    assert info.value.message == "nope"
    assert route.call_count == 1


async def test_500_is_unavailable_without_retry(store, client):
    route = store.get("/orders").respond(500)
    with pytest.raises(StoreUnavailable):
        await client.list_orders("tok")
    assert route.call_count == 1


async def test_write_is_not_retried_after_read_timeout(store, client):
    order_id = uuid4()
    route = store.post(f"/orders/{order_id}/cancel").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(StoreUnavailable):
        await client.cancel_order("tok", order_id)
    assert route.call_count == 1


async def test_get_is_retried_after_read_timeout(store, client):
    route = store.get("/orders").mock(
        side_effect=[httpx.ReadTimeout("slow"), httpx.Response(200, json=[])]
    )
    assert await client.list_orders("tok") == []
    assert route.call_count == 2


async def test_unexpected_shape_is_store_error(store, client):
    store.get("/orders").respond(200, json={"items": []})
    with pytest.raises(StoreError):
        await client.list_orders("tok")


async def test_first_call_uses_longer_timeout_then_shortens(store):
    seen = []
    async with StoreClient(BASE_URL, first_call_timeout=30, timeout=10) as c:
        original = c._http.request

        async def spy(*args, **kwargs):
            seen.append(kwargs["timeout"].read)
            return await original(*args, **kwargs)

        c._http.request = spy
        store.get("/orders").respond(200, json=[])
        await c.list_orders("tok")
        await c.list_orders("tok")
    assert seen == [30, 10]
