import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import actions, auth, conversations, health
from app.clients.store import StoreClient
from app.core.config import get_settings
from app.core.limits import RateLimiter, RunLocks
from app.core.logging import setup_logging
from app.core.middleware import RequestIDMiddleware
from app.core.ttl_cache import TTLCache
from app.db.session import get_engine, get_sessionmaker
from app.tools.registry import build_default_registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the long-lived clients once, share them through app.state, close on exit."""
    settings = get_settings()
    store = StoreClient(settings.STORE_API_URL)
    llm = None
    if settings.GEMINI_API_KEY.get_secret_value():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # google-genai emits SyntaxWarnings on import
            from app.agent.gemini import GeminiClient
        llm = GeminiClient(settings.GEMINI_API_KEY.get_secret_value(), settings.GEMINI_MODEL)

    app.state.store = store
    app.state.registry = build_default_registry(store)
    app.state.llm = llm
    app.state.sessionmaker = get_sessionmaker()
    app.state.auth_cache = TTLCache(ttl=settings.AUTH_CACHE_TTL_S)
    app.state.rate_limiter = RateLimiter(settings.RATE_LIMIT_MESSAGES, settings.RATE_LIMIT_WINDOW_S)
    app.state.action_rate_limiter = RateLimiter(
        settings.ACTION_RATE_LIMIT, settings.RATE_LIMIT_WINDOW_S
    )
    app.state.run_locks = RunLocks(stale_after=settings.AGENT_TIMEOUT_S + 30)
    try:
        yield
    finally:
        await store.aclose()
        await get_engine().dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    app = FastAPI(title="OrderPilot", lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(conversations.router)
    app.include_router(actions.router)
    return app


app = create_app()
