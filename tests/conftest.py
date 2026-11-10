from unittest.mock import AsyncMock

import pytest
import respx
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.clients.store import StoreClient
from app.main import create_app
from app.tools.context import UserContext
from app.tools.registry import build_default_registry
from tests.factories import BASE_URL, CUSTOMER_ID

TOKEN = "customer-token"


@pytest.fixture
def store():
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
async def registry(store):
    async with StoreClient(BASE_URL, sleep=AsyncMock()) as client:
        yield build_default_registry(client)


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
