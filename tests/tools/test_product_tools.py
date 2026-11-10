from uuid import uuid4

from tests.conftest import TOKEN
from tests.factories import product


async def test_search_products_success_is_compact(store, registry, ctx):
    route = store.get("/products").respond(
        200,
        json={
            "total": 3,
            "items": [
                product(name="Blue Mug", price="12.50", stock_quantity=3),
                product(name="Red Mug", price="9.00", stock_quantity=0),
                product(name="Hidden Mug", is_active=False),
            ],
        },
    )
    result = await registry.execute(
        "search_products", {"query": "mug", "max_price": 20, "limit": 5}, ctx
    )
    assert result.ok
    assert result.data["total_matches"] == 3
    assert [p["name"] for p in result.data["products"]] == ["Blue Mug", "Red Mug"]
    assert result.data["products"][0].keys() == {"id", "name", "price", "in_stock"}
    assert [p["in_stock"] for p in result.data["products"]] == [True, False]
    request = route.calls.last.request
    assert request.url.params["search"] == "mug"
    assert request.url.params["max_price"] == "20.0"
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"


async def test_search_products_empty(store, registry, ctx):
    store.get("/products").respond(200, json={"total": 0, "items": []})
    result = await registry.execute("search_products", {"query": "unicorn"}, ctx)
    assert result.ok
    assert result.data == {"total_matches": 0, "products": []}


async def test_search_products_store_5xx(store, registry, ctx):
    store.get("/products").respond(503)
    result = await registry.execute("search_products", {"query": "mug"}, ctx)
    assert (result.ok, result.error.code) == (False, "store_unavailable")


async def test_search_products_invalid_input(registry, ctx):
    for args in (
        {"limit": 0},
        {"limit": 500},
        {"min_price": 10, "max_price": 5},
        {"query": ""},
        {"customer_id": "someone-else"},
    ):
        result = await registry.execute("search_products", args, ctx)
        assert (result.ok, result.error.code) == (False, "invalid_arguments"), args


async def test_get_product_success(store, registry, ctx):
    p = product(description="x" * 1000)
    store.get(f"/products/{p['id']}").respond(200, json=p)
    result = await registry.execute("get_product", {"product_id": p["id"]}, ctx)
    assert result.ok
    assert result.data["name"] == "Blue Mug"
    assert result.data["sku"] == "MUG-BLUE"
    assert len(result.data["description"]) <= 400


async def test_get_product_not_found(store, registry, ctx):
    pid = uuid4()
    store.get(f"/products/{pid}").respond(404, json={"detail": "Product not found"})
    result = await registry.execute("get_product", {"product_id": str(pid)}, ctx)
    assert (result.ok, result.error.code) == (False, "not_found")


async def test_get_product_store_5xx(store, registry, ctx):
    pid = uuid4()
    store.get(f"/products/{pid}").respond(500)
    result = await registry.execute("get_product", {"product_id": str(pid)}, ctx)
    assert (result.ok, result.error.code) == (False, "store_unavailable")


async def test_get_product_invalid_input(registry, ctx):
    for args in ({}, {"product_id": "not-a-uuid"}, {"product_id": 7}):
        result = await registry.execute("get_product", args, ctx)
        assert result.error.code == "invalid_arguments", args
