from unittest.mock import AsyncMock

import pytest
import respx
from pydantic import SecretStr

from app.clients.store import StoreClient
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
