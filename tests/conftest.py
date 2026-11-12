from unittest.mock import AsyncMock

import pytest
import respx
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.clients.documind import DocuMindClient
from app.clients.store import StoreClient
from app.core.limits import RateLimiter, RunLocks
from app.core.ttl_cache import TTLCache
from app.main import create_app
from app.tools.context import UserContext
from app.tools.registry import build_default_registry
from tests.database import migrate_test_database
from tests.factories import (
    BASE_URL,
    CUSTOMER_ID,
    DOCUMIND_URL,
    POLICY_COLLECTION,
    collection,
)

TOKEN = "customer-token"
POLICY_COLLECTION_ROW = collection()


@pytest.fixture
def store():
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
async def store_client(store):
    async with StoreClient(BASE_URL, sleep=AsyncMock()) as client:
        yield client


@pytest.fixture
def documind():
    """Mocked DocuMind: login and the policy collection lookup always work.

    Tests add a search route with documind_search(...)."""
    with respx.mock(base_url=DOCUMIND_URL, assert_all_called=False) as mock:
        mock.post("/auth/login").respond(200, json={"access_token": "dm-token"})
        mock.get("/collections").respond(200, json=[POLICY_COLLECTION_ROW])
        yield mock


@pytest.fixture
def documind_search(documind):
    def _respond(*results, status=200):
        route = documind.post(f"/collections/{POLICY_COLLECTION_ROW['id']}/search")
        return route.respond(status, json=list(results)) if status == 200 else route.respond(status)

    return _respond


@pytest.fixture
async def documind_client(documind):
    async with DocuMindClient(
        DOCUMIND_URL,
        "svc@example.com",
        SecretStr("pw"),
        collection_name=POLICY_COLLECTION,
        sleep=AsyncMock(),
    ) as client:
        yield client


@pytest.fixture
def registry(store_client, documind_client):
    return build_default_registry(store_client, documind_client, policy_min_relevance=0.4)


@pytest.fixture
def ctx():
    return UserContext(customer_id=CUSTOMER_ID, store_token=SecretStr(TOKEN))


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(scope="session")
def database_url():
    return migrate_test_database()


@pytest.fixture
async def sessionmaker(database_url):
    """A clean database for each test (tables truncated, schema kept)."""
    engine = create_async_engine(database_url, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE conversations, messages, pending_actions CASCADE"))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def api_app(store_client, registry, clock):
    """The real app, with app.state filled by test doubles instead of the lifespan.

    Tests that need the database or a model set app.state.sessionmaker / app.state.llm.
    """
    app = create_app()
    app.state.store = store_client
    app.state.registry = registry
    app.state.llm = None
    app.state.auth_cache = TTLCache(ttl=60, clock=clock)
    app.state.rate_limiter = RateLimiter(limit=100, window_s=600, clock=clock)
    app.state.action_rate_limiter = RateLimiter(limit=100, window_s=600, clock=clock)
    app.state.run_locks = RunLocks(stale_after=90, clock=clock)
    return app


@pytest.fixture
async def api(api_app):
    async with AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test") as c:
        yield c


AUTH = {"Authorization": f"Bearer {TOKEN}"}
