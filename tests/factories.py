"""Builders for store API payloads, shaped like docs/store-openapi.json."""

from uuid import UUID, uuid4

BASE_URL = "https://store.test"
CUSTOMER_ID = UUID("11111111-1111-1111-1111-111111111111")


def product(**overrides) -> dict:
    data = {
        "id": str(uuid4()),
        "name": "Blue Mug",
        "description": "A sturdy blue mug.",
        "price": "12.50",
        "sku": "MUG-BLUE",
        "category_id": None,
        "stock_quantity": 7,
        "is_active": True,
    }
    return data | overrides


def order(**overrides) -> dict:
    data = {
        "id": str(uuid4()),
        "status": "pending",
        "total_amount": "25.00",
        "created_at": "2026-11-01T10:00:00Z",
        "items": [
            {
                "id": str(uuid4()),
                "product_id": str(uuid4()),
                "quantity": 2,
                "price_at_purchase": "12.50",
                "product": product(),
            }
        ],
    }
    return data | overrides


def me(**overrides) -> dict:
    data = {
        "id": str(CUSTOMER_ID),
        "email": "demo@example.com",
        "role": "customer",
        "created_at": "2026-01-01T00:00:00Z",
    }
    return data | overrides
