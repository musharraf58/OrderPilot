from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.clients.store import Product
from app.tools.base import Tool
from app.tools.context import UserContext

_DESCRIPTION_LIMIT = 400


class SearchProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Words to look for in product names/descriptions, e.g. 'blue mug'.",
    )
    min_price: float | None = Field(default=None, ge=0, description="Lowest price to include.")
    max_price: float | None = Field(default=None, ge=0, description="Highest price to include.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of products to return.")

    @model_validator(mode="after")
    def _price_range_is_sane(self) -> "SearchProductsInput":
        if None not in (self.min_price, self.max_price) and self.min_price > self.max_price:
            raise ValueError("min_price must not be greater than max_price")
        return self


class SearchProducts(Tool):
    name = "search_products"
    description = (
        "Search the store catalog. Use this when the customer asks what the store sells, "
        "wants to find a product, or wants options within a price range. Returns a short list "
        "of matching products with id, name, price and whether they are in stock. Use "
        "get_product with an id for full details."
    )
    input_model = SearchProductsInput
    kind = "read"

    async def _execute(self, args: SearchProductsInput, ctx: UserContext) -> dict[str, Any]:
        result = await self.client.list_products(
            ctx.store_token.get_secret_value(),
            search=args.query,
            min_price=args.min_price,
            max_price=args.max_price,
            page_size=args.limit,
        )
        products = [p for p in result.items if p.is_active][: args.limit]
        return {
            "total_matches": result.total,
            "products": [
                {
                    "id": str(p.id),
                    "name": p.name,
                    "price": p.price,
                    "in_stock": p.stock_quantity > 0,
                }
                for p in products
            ],
        }


class GetProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: UUID = Field(description="The product's id, as returned by search_products.")


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def product_details(p: Product) -> dict[str, Any]:
    return {
        "id": str(p.id),
        "name": p.name,
        "description": _truncate(p.description, _DESCRIPTION_LIMIT),
        "price": p.price,
        "sku": p.sku,
        "in_stock": p.stock_quantity > 0,
        "stock_quantity": p.stock_quantity,
        "available": p.is_active,
    }


class GetProduct(Tool):
    name = "get_product"
    description = (
        "Get the full details of one product (description, price, SKU, stock level). Use this "
        "when the customer asks about a specific product and you already have its id, for "
        "example from search_products or from an order."
    )
    input_model = GetProductInput
    kind = "read"

    async def _execute(self, args: GetProductInput, ctx: UserContext) -> dict[str, Any]:
        product = await self.client.get_product(ctx.store_token.get_secret_value(), args.product_id)
        return product_details(product)
