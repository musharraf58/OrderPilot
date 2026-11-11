"""Shared FastAPI dependencies. Long-lived objects live on app.state (set in the lifespan,
or directly by tests)."""

import hashlib
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.agent.llm import LLMClient
from app.clients.store import StoreClient, StoreError, StoreUnavailable
from app.core.ttl_cache import TTLCache
from app.tools.context import UserContext
from app.tools.registry import ToolRegistry

_bearer = HTTPBearer(auto_error=False)


def get_store(request: Request) -> StoreClient:
    return request.app.state.store


def get_registry(request: Request) -> ToolRegistry:
    return request.app.state.registry


def get_llm(request: Request) -> LLMClient:
    llm = request.app.state.llm
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The assistant is not configured.")
    return llm


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


async def get_user_context(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    store: Annotated[StoreClient, Depends(get_store)],
) -> UserContext:
    """Turn the Bearer token into a UserContext (customer_id from the store's /auth/me).

    Successful lookups are cached briefly, keyed by a hash of the token, so we don't call
    /auth/me on every request. Failures are never cached.
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Missing bearer token.")
    token = credentials.credentials
    cache: TTLCache[str, UserContext] = request.app.state.auth_cache
    key = hashlib.sha256(token.encode()).hexdigest()

    ctx = cache.get(key)
    if ctx is not None:
        return ctx
    try:
        ctx = await UserContext.from_token(store, token)
    except StoreUnavailable as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The store is unavailable. Try again later."
        ) from exc
    except StoreError as exc:
        # 401 and anything else the store rejects the token with.
        raise _unauthorized("Invalid or expired token.") from exc
    cache.set(key, ctx)
    return ctx


CurrentUser = Annotated[UserContext, Depends(get_user_context)]
