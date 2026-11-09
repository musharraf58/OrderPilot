import logging
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_ctx

HEADER = "X-Request-ID"
logger = logging.getLogger("app.request")


class RequestIDMiddleware:
    """Pure ASGI middleware: assigns a request ID, logs the request, echoes the ID back."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = Headers(scope=scope).get(HEADER) or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info("%s %s -> %s", scope["method"], scope["path"], status)
            request_id_ctx.reset(token)
