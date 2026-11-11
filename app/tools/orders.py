from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.clients.store import Order, StoreConflict
from app.tools.base import Tool, ToolFailure
from app.tools.context import UserContext

OrderStatus = Literal["pending", "paid", "shipped", "cancelled"]

# One message for "does not exist" and "belongs to someone else", so the answer never
# reveals whether another customer's order exists.
_ORDER_NOT_FOUND = ToolFailure("not_found", "No order with that id was found for this customer.")

_NOT_CANCELLABLE_REASONS = {
    "cancelled": "This order is already cancelled.",
    "paid": (
        "This order has already been paid and can no longer be cancelled. "
        "Refunds are not supported here."
    ),
    "shipped": "This order has already shipped and can no longer be cancelled.",
}


class OrderIdInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: UUID = Field(description="The order's id, as returned by list_my_orders.")


class ListMyOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: OrderStatus | None = Field(
        default=None, description="Only return orders with this status. Omit for all orders."
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of orders to return, newest first (1 = most recent order).",
    )


async def _find_owned_order(tool: Tool, order_id: UUID, ctx: UserContext) -> Order:
    """Return the order only if it is one of this customer's orders.

    The store's order objects carry no owner field, so ownership is proven by membership
    in GET /orders, which the store scopes to the token's owner (and ctx.customer_id was
    derived from that same token via /auth/me). Anything else is reported as not found.
    """
    orders = await tool.client.list_orders(ctx.store_token.get_secret_value())
    for order in orders:
        if order.id == order_id:
            return order
    raise _ORDER_NOT_FOUND


def order_summary(order: Order) -> dict[str, Any]:
    return {
        "id": str(order.id),
        "status": order.status,
        "total": order.total_amount,
        "created_at": order.created_at,
        "item_count": sum(i.quantity for i in order.items),
    }


def order_details(order: Order) -> dict[str, Any]:
    return {
        **{k: v for k, v in order_summary(order).items() if k != "item_count"},
        "items": [
            {
                "product_id": str(i.product_id),
                "name": i.product.name,
                "quantity": i.quantity,
                "unit_price": i.price_at_purchase,
            }
            for i in order.items
        ],
    }


class ListMyOrders(Tool):
    name = "list_my_orders"
    description = (
        "List the current customer's orders with id, status, total, date and number of items. "
        "Orders are always returned newest first, so limit=1 gives the most recent order. "
        "Use this when the customer asks about their orders, order history or "
        "'where is my order', or when you need an order id. You never need to ask for the "
        "customer's identity: it is already known. Use get_order for the items in one order."
    )
    input_model = ListMyOrdersInput
    kind = "read"

    async def _execute(self, args: ListMyOrdersInput, ctx: UserContext) -> dict[str, Any]:
        orders = await self.client.list_orders(ctx.store_token.get_secret_value())
        if args.status:
            orders = [o for o in orders if o.status == args.status]
        orders.sort(key=lambda o: datetime.fromisoformat(o.created_at), reverse=True)
        return {
            "total_orders": len(orders),
            "orders": [order_summary(o) for o in orders[: args.limit]],
        }


class GetOrder(Tool):
    name = "get_order"
    description = (
        "Get one of the current customer's orders in detail: status, total, date and each item "
        "with quantity and price. Use this when the customer asks about a specific order. Only "
        "the customer's own orders can be found; any other id returns not_found."
    )
    input_model = OrderIdInput
    kind = "read"

    async def _execute(self, args: OrderIdInput, ctx: UserContext) -> dict[str, Any]:
        await _find_owned_order(self, args.order_id, ctx)
        order = await self.client.get_order(ctx.store_token.get_secret_value(), args.order_id)
        return order_details(order)


class CancelOrder(Tool):
    name = "cancel_order"
    description = (
        "Cancel one of the current customer's orders. This changes data and cannot be undone, "
        "so only call it after the customer has clearly asked to cancel that specific order. "
        "Only orders with status 'pending' can be cancelled; paid or shipped orders return "
        "not_cancellable (refunds are not supported). Returns the order's new status."
    )
    input_model = OrderIdInput
    kind = "write"

    async def _execute(self, args: OrderIdInput, ctx: UserContext) -> dict[str, Any]:
        order = await _find_owned_order(self, args.order_id, ctx)
        if order.status != "pending":
            reason = _NOT_CANCELLABLE_REASONS.get(order.status, "This order cannot be cancelled.")
            raise ToolFailure("not_cancellable", reason)
        try:
            cancelled = await self.client.cancel_order(
                ctx.store_token.get_secret_value(), args.order_id
            )
        except StoreConflict as exc:
            # The status changed between our check and the store's: the store has the final say.
            raise ToolFailure(
                "not_cancellable", "The store says this order can no longer be cancelled."
            ) from exc
        return {"id": str(cancelled.id), "status": cancelled.status}
