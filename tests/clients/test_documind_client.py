import asyncio
from unittest.mock import AsyncMock
from urllib.parse import parse_qs
from uuid import uuid4

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.clients.documind import (
    DocuMindClient,
    DocuMindError,
    DocuMindNotFound,
    DocuMindRateLimited,
    DocuMindUnauthorized,
    DocuMindUnavailable,
)
from tests.factories import DOCUMIND_URL, POLICY_COLLECTION, collection, search_result


@pytest.fixture
def documind():
    with respx.mock(base_url=DOCUMIND_URL, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
async def client():
    async with DocuMindClient(
        DOCUMIND_URL,
        "svc@example.com",
        SecretStr("s3cret"),
        collection_name=POLICY_COLLECTION,
        sleep=AsyncMock(),
    ) as c:
        yield c


def token(value):
    return httpx.Response(200, json={"access_token": value, "token_type": "bearer"})


@pytest.fixture
def coll(documind):
    c = collection()
    documind.get("/collections", name="collections").respond(
        200, json=[collection(name="other"), c]
    )
    return c


async def test_search_logs_in_with_form_and_sends_bearer(documind, client, coll):
    login = documind.post("/auth/login").mock(return_value=token("tok-1"))
    search = documind.post(f"/collections/{coll['id']}/search").respond(
        200, json=[search_result(content="Returns are handled by support.")]
    )
    results = await client.search("returns", limit=4)

    form = parse_qs(login.calls.last.request.content.decode())
    assert form == {"username": ["svc@example.com"], "password": ["s3cret"]}
    assert login.calls.last.request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert search.calls.last.request.headers["Authorization"] == "Bearer tok-1"
    assert search.calls.last.request.read() == b'{"query":"returns","limit":4}'
    assert results[0].content == "Returns are handled by support."


async def test_token_and_collection_id_are_reused(documind, client, coll):
    login = documind.post("/auth/login").mock(return_value=token("tok-1"))
    collections = documind["collections"]
    documind.post(f"/collections/{coll['id']}/search").respond(200, json=[])
    for _ in range(3):
        await client.search("q", limit=3)
    assert login.call_count == 1
    assert collections.call_count == 1


async def test_expired_token_triggers_one_relogin_and_retry(documind, client, coll):
    login = documind.post("/auth/login").mock(side_effect=[token("old"), token("new")])
    search = documind.post(f"/collections/{coll['id']}/search").mock(
        side_effect=[
            httpx.Response(200, json=[]),
            httpx.Response(401),
            httpx.Response(200, json=[]),
        ]
    )
    await client.search("q", limit=3)
    await client.search("q", limit=3)  # old token rejected -> log in again -> retry
    assert login.call_count == 2
    assert [c.request.headers["Authorization"] for c in search.calls] == [
        "Bearer old",
        "Bearer old",
        "Bearer new",
    ]


async def test_rejected_fresh_token_is_unauthorized_without_looping(documind, client, coll):
    login = documind.post("/auth/login").mock(return_value=token("t"))
    documind.get("/collections").respond(401)
    with pytest.raises(DocuMindUnauthorized):
        await client.search("q", limit=3)
    assert login.call_count == 2


async def test_bad_credentials_are_unauthorized(documind, client):
    login = documind.post("/auth/login").respond(
        401, json={"detail": "Incorrect email or password"}
    )
    with pytest.raises(DocuMindUnauthorized):
        await client.search("q", limit=3)
    assert login.call_count == 1


async def test_concurrent_first_calls_share_one_login(documind, client, coll):
    async def slow_login(_request):
        await asyncio.sleep(0.01)
        return token("t")

    login = documind.post("/auth/login").mock(side_effect=slow_login)
    documind.post(f"/collections/{coll['id']}/search").respond(200, json=[])
    await asyncio.gather(*(client.search("q", limit=3) for _ in range(5)))
    assert login.call_count == 1


async def test_retries_503_then_succeeds(documind, client, coll):
    documind.post("/auth/login").mock(return_value=token("t"))
    search = documind.post(f"/collections/{coll['id']}/search").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=[search_result()])]
    )
    assert len(await client.search("q", limit=3)) == 1
    assert search.call_count == 2


async def test_login_is_retried_through_a_cold_start(documind, client, coll):
    login = documind.post("/auth/login").mock(side_effect=[httpx.ReadTimeout("asleep"), token("t")])
    documind.post(f"/collections/{coll['id']}/search").respond(200, json=[])
    await client.search("q", limit=3)
    assert login.call_count == 2


async def test_down_after_retries_is_unavailable(documind, client, coll):
    documind.post("/auth/login").mock(return_value=token("t"))
    search = documind.post(f"/collections/{coll['id']}/search").mock(
        side_effect=httpx.ConnectError("down")
    )
    with pytest.raises(DocuMindUnavailable):
        await client.search("q", limit=3)
    assert search.call_count == 3


@pytest.mark.parametrize(
    ("status", "exc"),
    [(429, DocuMindRateLimited), (500, DocuMindUnavailable), (422, DocuMindError)],
)
async def test_error_mapping_without_retry(documind, client, coll, status, exc):
    documind.post("/auth/login").mock(return_value=token("t"))
    search = documind.post(f"/collections/{coll['id']}/search").respond(status)
    with pytest.raises(exc):
        await client.search("q", limit=3)
    assert search.call_count == 1


async def test_missing_collection_is_not_found(documind, client):
    documind.post("/auth/login").mock(return_value=token("t"))
    documind.get("/collections").respond(200, json=[collection(name="something else")])
    with pytest.raises(DocuMindNotFound):
        await client.search("q", limit=3)


async def test_unexpected_shape_is_documind_error(documind, client, coll):
    documind.post("/auth/login").mock(return_value=token("t"))
    documind.post(f"/collections/{coll['id']}/search").respond(200, json={"results": []})
    with pytest.raises(DocuMindError):
        await client.search("q", limit=3)


async def test_upload_sends_multipart_file(documind, client):
    documind.post("/auth/login").mock(return_value=token("t"))
    cid = uuid4()
    upload = documind.post(f"/collections/{cid}/documents").respond(
        202,
        json={
            "id": str(uuid4()),
            "original_filename": "shipping.md",
            "content_hash": "abc",
            "size_bytes": 5,
            "status": "pending",
            "error_message": None,
        },
    )
    doc = await client.upload_document(cid, "shipping.md", b"# Hi\n", "text/markdown")
    body = upload.calls.last.request.read()
    assert b'filename="shipping.md"' in body and b"# Hi" in body
    assert doc.status == "pending"
