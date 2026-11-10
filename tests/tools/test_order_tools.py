from uuid import uuid4

import pytest

from tests.conftest import TOKEN
from tests.factories import order

FOREIGN_ORDER_ID = str(uuid4())


@pytest.fixture
def my_orders():
    return [
        order(status="pending", created_at="2026-11-01T10:00:00Z"),
        order(status="paid", created_at="2026-11-03T10:00:00Z"),
        order(status="shipped", created_at="2026-10-01T10:00:00Z"),
        order(status="cancelled", created_at="2026-09-01T10:00:00Z"),
    ]


# --- list_my_orders ---------------------------------------------------------


async def test_list_my_orders_success_sorted_and_compact(store, registry, ctx, my_orders):
    route = store.get("/orders").respond(200, json=my_orders)
    result = await registry.execute("list_my_orders", {}, ctx)
    assert result.ok
    assert result.data["total_orders"] == 4
    assert [o["created_at"][:10] for o in result.data["orders"]] == [
        "2026-11-03",
        "2026-11-01",
        "2026-10-01",
        "2026-09-01",
    ]
    assert result.data["orders"][0].keys() == {"id", "status", "total", "created_at", "item_count"}
    assert result.data["orders"][0]["item_count"] == 2
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"


async def test_list_my_orders_status_filter_and_limit(store, registry, ctx, my_orders):
    store.get("/orders").respond(200, json=my_orders)
    paid = await registry.execute("list_my_orders", {"status": "paid"}, ctx)
    assert [o["status"] for o in paid.data["orders"]] == ["paid"]
    limited = await registry.execute("list_my_orders", {"limit": 2}, ctx)
    assert len(limited.data["orders"]) == 2
    assert limited.data["total_orders"] == 4


async def test_list_my_orders_empty(store, registry, ctx):
    store.get("/orders").respond(200, json=[])
    result = await registry.execute("list_my_orders", {}, ctx)
    assert result.data == {"total_orders": 0, "orders": []}


async def test_list_my_orders_unauthorized(store, registry, ctx):
    store.get("/orders").respond(401, json={"detail": "expired"})
    result = await registry.execute("list_my_orders", {}, ctx)
    assert (result.ok, result.error.code) == (False, "unauthorized")


async def test_list_my_orders_store_5xx(store, registry, ctx):
    store.get("/orders").respond(502)
    result = await registry.execute("list_my_orders", {}, ctx)
    assert (result.ok, result.error.code) == (False, "store_unavailable")


async def test_list_my_orders_invalid_input(registry, ctx):
    for args in ({"status": "lost"}, {"limit": 0}, {"limit": 51}, {"customer_id": "x"}):
        result = await registry.execute("list_my_orders", args, ctx)
        assert result.error.code == "invalid_arguments", args


# --- get_order --------------------------------------------------------------


async def test_get_order_success(store, registry, ctx, my_orders):
    target = my_orders[1]
    store.get("/orders").respond(200, json=my_orders)
    store.get(f"/orders/{target['id']}").respond(200, json=target)
    result = await registry.execute("get_order", {"order_id": target["id"]}, ctx)
    assert result.ok
    assert result.data["id"] == target["id"]
    assert result.data["items"][0].keys() == {"product_id", "name", "quantity", "unit_price"}
    assert "sku" not in str(result.data)


async def test_get_order_not_found_when_store_404s(store, registry, ctx, my_orders):
    target = my_orders[0]
    store.get("/orders").respond(200, json=my_orders)
    store.get(f"/orders/{target['id']}").respond(404, json={"detail": "gone"})
    result = await registry.execute("get_order", {"order_id": target["id"]}, ctx)
    assert (result.ok, result.error.code) == (False, "not_found")


async def test_get_order_store_5xx(store, registry, ctx):
    store.get("/orders").respond(503)
    result = await registry.execute("get_order", {"order_id": str(uuid4())}, ctx)
    assert (result.ok, result.error.code) == (False, "store_unavailable")


async def test_get_order_invalid_input(registry, ctx):
    for args in ({}, {"order_id": "123"}):
        result = await registry.execute("get_order", args, ctx)
        assert result.error.code == "invalid_arguments", args


async def test_get_order_of_another_customer_is_not_found_and_never_fetched(
    store, registry, ctx, my_orders
):
    store.get("/orders").respond(200, json=my_orders)
    # Even if the store would happily return it, we must never ask for it.
    detail = store.get(f"/orders/{FOREIGN_ORDER_ID}").respond(200, json=order(id=FOREIGN_ORDER_ID))
    result = await registry.execute("get_order", {"order_id": FOREIGN_ORDER_ID}, ctx)
    assert (result.ok, result.error.code) == (False, "not_found")
    assert not detail.called


async def test_foreign_and_nonexistent_orders_are_indistinguishable(
    store, registry, ctx, my_orders
):
    store.get("/orders").respond(200, json=my_orders)
    foreign = await registry.execute("get_order", {"order_id": FOREIGN_ORDER_ID}, ctx)
    missing = await registry.execute("get_order", {"order_id": str(uuid4())}, ctx)
    assert foreign.model_dump() == missing.model_dump()
    cancel_foreign = await registry.execute("cancel_order", {"order_id": FOREIGN_ORDER_ID}, ctx)
    cancel_missing = await registry.execute("cancel_order", {"order_id": str(uuid4())}, ctx)
    assert cancel_foreign.model_dump() == cancel_missing.model_dump()


# --- cancel_order -----------------------------------------------------------


async def test_cancel_order_success(store, registry, ctx, my_orders):
    target = my_orders[0]
    store.get("/orders").respond(200, json=my_orders)
    cancel = store.post(f"/orders/{target['id']}/cancel").respond(
        200, json=order(id=target["id"], status="cancelled")
    )
    result = await registry.execute("cancel_order", {"order_id": target["id"]}, ctx)
    assert result.ok
    assert result.data == {"id": target["id"], "status": "cancelled"}
    assert cancel.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"


async def test_cancel_order_of_another_customer_never_calls_store_cancel(
    store, registry, ctx, my_orders
):
    store.get("/orders").respond(200, json=my_orders)
    cancel = store.post(f"/orders/{FOREIGN_ORDER_ID}/cancel").respond(200, json=order())
    result = await registry.execute("cancel_order", {"order_id": FOREIGN_ORDER_ID}, ctx)
    assert (result.ok, result.error.code) == (False, "not_found")
    assert not cancel.called


@pytest.mark.parametrize("index", [1, 2, 3], ids=["paid", "shipped", "cancelled"])
async def test_cancel_order_not_cancellable_statuses_skip_store_cancel(
    store, registry, ctx, my_orders, index
):
    target = my_orders[index]
    store.get("/orders").respond(200, json=my_orders)
    cancel = store.post(f"/orders/{target['id']}/cancel").respond(200, json=target)
    result = await registry.execute("cancel_order", {"order_id": target["id"]}, ctx)
    assert (result.ok, result.error.code) == (False, "not_cancellable")
    assert not cancel.called


async def test_cancel_paid_order_message_mentions_refunds(store, registry, ctx, my_orders):
    target = my_orders[1]
    store.get("/orders").respond(200, json=my_orders)
    result = await registry.execute("cancel_order", {"order_id": target["id"]}, ctx)
    assert "refund" in result.error.message.lower()


async def test_cancel_order_store_409_is_not_cancellable(store, registry, ctx, my_orders):
    target = my_orders[0]
    store.get("/orders").respond(200, json=my_orders)
    store.post(f"/orders/{target['id']}/cancel").respond(409, json={"detail": "not pending"})
    result = await registry.execute("cancel_order", {"order_id": target["id"]}, ctx)
    assert (result.ok, result.error.code) == (False, "not_cancellable")


async def test_cancel_order_store_404_is_not_found(store, registry, ctx, my_orders):
    target = my_orders[0]
    store.get("/orders").respond(200, json=my_orders)
    store.post(f"/orders/{target['id']}/cancel").respond(404)
    result = await registry.execute("cancel_order", {"order_id": target["id"]}, ctx)
    assert (result.ok, result.error.code) == (False, "not_found")


async def test_cancel_order_store_5xx_is_not_retried_into_double_cancel(
    store, registry, ctx, my_orders
):
    target = my_orders[0]
    store.get("/orders").respond(200, json=my_orders)
    cancel = store.post(f"/orders/{target['id']}/cancel").respond(500)
    result = await registry.execute("cancel_order", {"order_id": target["id"]}, ctx)
    assert (result.ok, result.error.code) == (False, "store_unavailable")
    assert cancel.call_count == 1


async def test_cancel_order_invalid_input(registry, ctx):
    for args in ({}, {"order_id": "abc"}, {"order_id": str(uuid4()), "force": True}):
        result = await registry.execute("cancel_order", args, ctx)
        assert result.error.code == "invalid_arguments", args


# --- registry wiring --------------------------------------------------------


def test_default_registry_kinds(registry):
    assert {t.name: t.kind for t in registry.list_tools()} == {
        "search_products": "read",
        "get_product": "read",
        "list_my_orders": "read",
        "get_order": "read",
        "cancel_order": "write",
    }


def test_default_schemas_never_expose_identity(registry):
    for schema in registry.schemas():
        props = schema["parameters"].get("properties", {})
        assert not {"ctx", "customer_id", "user_id", "token", "store_token"} & set(props)
