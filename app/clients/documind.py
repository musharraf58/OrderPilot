"""Async client for the DocuMind RAG API (retrieval only).

OrderPilot uses DocuMind's search endpoint to fetch policy passages; it never calls its
answer-generation endpoints (/ask), so no extra LLM call is made. It authenticates as one
service account from settings: the credential never comes from the model or a customer.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Self
from uuid import UUID

import httpx
from pydantic import BaseModel, SecretStr, TypeAdapter

logger = logging.getLogger(__name__)

_RETRY_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
_RETRY_STATUSES = frozenset({502, 503, 504})


class DocuMindError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DocuMindUnauthorized(DocuMindError):
    """Login failed, or a fresh token was still rejected."""


class DocuMindNotFound(DocuMindError): ...


class DocuMindRateLimited(DocuMindError):
    """DocuMind's own rate limit or daily quota (429)."""


class DocuMindUnavailable(DocuMindError):
    """Network failure or 5xx after retries were exhausted."""


class DocuMindModel(BaseModel):
    model_config = {"extra": "ignore"}


class Collection(DocuMindModel):
    id: UUID
    name: str


class Document(DocuMindModel):
    id: UUID
    original_filename: str
    content_hash: str
    size_bytes: int
    status: str  # pending | processing | ready | failed
    error_message: str | None = None


class SearchResult(DocuMindModel):
    chunk_id: UUID
    document_id: UUID
    document_name: str
    content: str
    section_title: str | None = None
    page_number: int | None = None
    distance: float  # cosine distance: smaller = more similar


_COLLECTIONS = TypeAdapter(list[Collection])
_DOCUMENTS = TypeAdapter(list[Document])
_RESULTS = TypeAdapter(list[SearchResult])


class DocuMindClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        password: SecretStr,
        *,
        collection_name: str,
        first_call_timeout: float = 30.0,
        timeout: float = 15.0,
        max_attempts: int = 3,
        backoff: float = 0.5,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"))
        self._email = email
        self._password = password
        self.collection_name = collection_name
        self._first_call_timeout = first_call_timeout
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._backoff = backoff
        self._sleep = sleep
        # The free tier sleeps: allow a long timeout until we've had one healthy response.
        self._warm = False
        # DocuMind has no refresh token: we log in, cache the access token, and log in
        # again when it is rejected. The lock makes concurrent callers share one login.
        self._token: str | None = None
        self._login_lock = asyncio.Lock()
        self._collection_id: UUID | None = None

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # --- transport ----------------------------------------------------------

    async def _send(
        self, method: str, path: str, *, idempotent: bool | None = None, **kwargs: Any
    ) -> httpx.Response:
        """One logical request with retries. Returns any non-retryable response as-is."""
        retryable: tuple[type[Exception], ...] = _RETRY_CONNECT_ERRORS
        if idempotent if idempotent is not None else method == "GET":
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
                    timeout=httpx.Timeout(seconds, connect=min(seconds, 15.0)),
                    **kwargs,
                )
            except retryable as exc:
                last_error = type(exc).__name__
                logger.warning("documind %s %s failed: %s", method, path, last_error)
                continue
            except httpx.HTTPError as exc:
                raise DocuMindUnavailable(f"DocuMind request failed: {type(exc).__name__}") from exc
            if response.status_code in _RETRY_STATUSES:
                last_error = f"HTTP {response.status_code}"
                logger.warning("documind %s %s -> %s", method, path, response.status_code)
                continue
            self._warm = True
            return response
        raise DocuMindUnavailable(f"DocuMind unavailable ({last_error})")

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
        message = detail or f"DocuMind returned HTTP {status}"
        if status == 401:
            raise DocuMindUnauthorized(message, status)
        if status == 404:
            raise DocuMindNotFound(message, status)
        if status == 429:
            raise DocuMindRateLimited(message, status)
        if status >= 500:
            raise DocuMindUnavailable(message, status)
        raise DocuMindError(message, status)

    async def _login(self) -> str:
        # OAuth2 password form; logging in changes nothing, so it may be retried.
        response = await self._send(
            "POST",
            "/auth/login",
            data={"username": self._email, "password": self._password.get_secret_value()},
            idempotent=True,
        )
        if response.status_code in (400, 401, 422):
            raise DocuMindUnauthorized("DocuMind login failed (check DOCUMIND_EMAIL/PASSWORD)")
        self._raise_for_status(response)
        return response.json()["access_token"]

    async def _token_for_request(self, rejected: str | None = None) -> str:
        async with self._login_lock:
            # Only log in if nobody refreshed the token while we waited for the lock.
            if self._token is None or self._token == rejected:
                self._token = await self._login()
            return self._token

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """An authenticated request: on a 401, log in again once and retry once."""
        token = self._token or await self._token_for_request()
        response = await self._send(
            method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs
        )
        if response.status_code == 401:
            token = await self._token_for_request(rejected=token)
            response = await self._send(
                method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs
            )
        self._raise_for_status(response)
        return response

    @staticmethod
    def _parse[T](adapter: TypeAdapter[T], response: httpx.Response) -> T:
        try:
            return adapter.validate_python(response.json())
        except ValueError as exc:
            raise DocuMindError("DocuMind returned an unexpected response shape") from exc

    # --- collections ----------------------------------------------------------

    async def list_collections(self) -> list[Collection]:
        collections: list[Collection] = []
        offset = 0
        while True:
            page = self._parse(
                _COLLECTIONS,
                await self._request("GET", "/collections", params={"limit": 100, "offset": offset}),
            )
            collections += page
            if len(page) < 100:
                return collections
            offset += 100

    async def find_collection(self, name: str) -> Collection | None:
        return next((c for c in await self.list_collections() if c.name == name), None)

    async def create_collection(self, name: str, description: str | None = None) -> Collection:
        response = await self._request(
            "POST", "/collections", json={"name": name, "description": description}
        )
        return self._parse(TypeAdapter(Collection), response)

    async def collection_id(self) -> UUID:
        """The configured collection's id, looked up by name once and cached."""
        if self._collection_id is None:
            collection = await self.find_collection(self.collection_name)
            if collection is None:
                raise DocuMindNotFound(f"DocuMind collection '{self.collection_name}' not found")
            self._collection_id = collection.id
        return self._collection_id

    # --- documents (used by the seed script) ---------------------------------------

    async def list_documents(self, collection_id: UUID) -> list[Document]:
        documents: list[Document] = []
        offset = 0
        while True:
            response = await self._request(
                "GET",
                f"/collections/{collection_id}/documents",
                params={"limit": 100, "offset": offset},
            )
            page = self._parse(_DOCUMENTS, response)
            documents += page
            if len(page) < 100:
                return documents
            offset += 100

    async def upload_document(
        self, collection_id: UUID, filename: str, content: bytes, content_type: str
    ) -> Document:
        response = await self._request(
            "POST",
            f"/collections/{collection_id}/documents",
            files={"file": (filename, content, content_type)},
        )
        return self._parse(TypeAdapter(Document), response)

    async def get_document(self, document_id: UUID) -> Document:
        return self._parse(
            TypeAdapter(Document), await self._request("GET", f"/documents/{document_id}")
        )

    async def delete_document(self, document_id: UUID) -> None:
        await self._request("DELETE", f"/documents/{document_id}")

    # --- retrieval --------------------------------------------------------------

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """Matching chunks from the configured collection (retrieval only, no LLM answer)."""
        collection_id = await self.collection_id()
        response = await self._request(
            "POST",
            f"/collections/{collection_id}/search",
            json={"query": query, "limit": limit},
            # A search changes nothing, so a timed-out one may be retried.
            idempotent=True,
        )
        return self._parse(_RESULTS, response)
