import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Self
from uuid import UUID

import httpx
from pydantic import BaseModel, TypeAdapter

logger = logging.getLogger(__name__)

# Errors raised before the request reached the server: always safe to retry.
_RETRY_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
_RETRY_STATUSES = frozenset({502, 503, 504})


class StoreError(Exception):
    """Base class for every error the store client raises."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class StoreUnauthorized(StoreError): ...


class StoreForbidden(StoreError): ...


class StoreNotFound(StoreError): ...


class StoreConflict(StoreError): ...


class StoreUnavailable(StoreError):
    """Network failure or 5xx after retries were exhausted."""


_STATUS_TO_ERROR: dict[int, type[StoreError]] = {
    401: StoreUnauthorized,
    403: StoreForbidden,
    404: StoreNotFound,
    409: StoreConflict,
}


class StoreModel(BaseModel):
    # Ignore fields we do not model, so the store can add fields without breaking us.
    model_config = {"extra": "ignore"}


class User(StoreModel):
    id: UUID
    email: str
    role: str


class Token(StoreModel):
    access_token: str
    refresh_token: str


class Product(StoreModel):
    id: UUID
    name: str
    description: str | None
    price: str
    sku: str
    category_id: UUID | None
    stock_quantity: int
    is_active: bool


class ProductList(StoreModel):
    total: int
    items: list[Product]


class OrderItem(StoreModel):
    id: UUID
    product_id: UUID
    quantity: int
    price_at_purchase: str
    product: Product


class Order(StoreModel):
    id: UUID
    status: str
    total_amount: str
    created_at: str
    items: list[OrderItem]


_ORDER_LIST = TypeAdapter(list[Order])


class StoreClient:
    """Async HTTP client for the e-commerce backend.

    The end user's token is passed on every call and never stored on the client,
    so one client instance can safely serve many users.
    """

    def __init__(
        self,
        base_url: str,
        *,
        first_call_timeout: float = 30.0,
        timeout: float = 10.0,
        max_attempts: int = 3,
        backoff: float = 0.5,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"))
        self._first_call_timeout = first_call_timeout
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._backoff = backoff
        self._sleep = sleep
        # The free-tier server can take a while to wake up, so until we have seen one
        # healthy response we allow a much longer timeout.
        self._warm = False

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        retryable: tuple[type[Exception], ...] = _RETRY_CONNECT_ERRORS
        if method == "GET":
            # A read that timed out can be repeated; a write may already have been applied.
            retryable += (httpx.ReadTimeout,)

        last_error = "no response"
        for attempt in range(self._max_attempts):
            if attempt:
                await self._sleep(self._backoff * 2 ** (attempt - 1))
            seconds = self._timeout if self._warm else self._first_call_timeout
            try:
                response = await self._http.request(
                    method,
                    path,
                    headers=headers,
                    params=params,
                    json=json,
                    timeout=httpx.Timeout(seconds, connect=min(seconds, 15.0)),
                )
            except retryable as exc:
                last_error = type(exc).__name__
                logger.warning("store %s %s failed: %s", method, path, last_error)
                continue
            except httpx.HTTPError as exc:
                raise StoreUnavailable(f"store request failed: {type(exc).__name__}") from exc

            if response.status_code in _RETRY_STATUSES:
                last_error = f"HTTP {response.status_code}"
                logger.warning("store %s %s -> %s", method, path, response.status_code)
                continue

            self._warm = True
            self._raise_for_status(response)
            return response

        raise StoreUnavailable(f"store unavailable ({last_error})")

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        status = response.status_code
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict) and isinstance(body.get("detail"), str):
                detail = body["detail"]
        except ValueError:
            pass
        message = detail or f"store returned HTTP {status}"
        if status >= 500:
            raise StoreUnavailable(message, status)
        raise _STATUS_TO_ERROR.get(status, StoreError)(message, status)

    @staticmethod
    def _parse[T: BaseModel](model: type[T], response: httpx.Response) -> T:
        try:
            return model.model_validate(response.json())
        except ValueError as exc:
            raise StoreError("store returned an unexpected response shape") from exc

    # --- auth -------------------------------------------------------------

    async def login(self, email: str, password: str) -> Token:
        response = await self._request(
            "POST", "/auth/login", json={"email": email, "password": password}
        )
        return self._parse(Token, response)

    async def get_me(self, token: str) -> User:
        return self._parse(User, await self._request("GET", "/auth/me", token=token))

    # --- products ---------------------------------------------------------

    async def list_products(
        self,
        token: str,
        *,
        search: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        page_size: int = 20,
    ) -> ProductList:
        params: dict[str, Any] = {"page_size": page_size}
        if search is not None:
            params["search"] = search
        if min_price is not None:
            params["min_price"] = min_price
        if max_price is not None:
            params["max_price"] = max_price
        response = await self._request("GET", "/products", token=token, params=params)
        return self._parse(ProductList, response)

    async def get_product(self, token: str, product_id: UUID) -> Product:
        response = await self._request("GET", f"/products/{product_id}", token=token)
        return self._parse(Product, response)

    # --- orders -----------------------------------------------------------

    async def list_orders(self, token: str) -> list[Order]:
        """All of the token owner's orders.

        The store's GET /orders takes no paging parameters and returns a plain array, so
        one call sees every order. If the store ever paginates it, the response shape
        changes and parsing fails loudly instead of silently returning a partial list.
        """
        response = await self._request("GET", "/orders", token=token)
        try:
            return _ORDER_LIST.validate_python(response.json())
        except ValueError as exc:
            raise StoreError("store returned an unexpected response shape") from exc

    async def get_order(self, token: str, order_id: UUID) -> Order:
        return self._parse(Order, await self._request("GET", f"/orders/{order_id}", token=token))

    async def cancel_order(self, token: str, order_id: UUID) -> Order:
        response = await self._request("POST", f"/orders/{order_id}/cancel", token=token)
        return self._parse(Order, response)
