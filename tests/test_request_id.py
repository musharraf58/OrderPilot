import json
import logging
import uuid


async def test_generates_request_id(client):
    r = await client.get("/nope")
    assert uuid.UUID(r.headers["X-Request-ID"])


async def test_echoes_incoming_request_id(client):
    r = await client.get("/nope", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


async def test_request_id_in_json_logs(client):
    from app.core.logging import JsonFormatter

    lines: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            lines.append(self.format(record))

    handler = Capture()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("app.request")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        await client.get("/nope", headers={"X-Request-ID": "log-me"})
    finally:
        logger.removeHandler(handler)

    entry = json.loads(lines[0])
    assert entry["request_id"] == "log-me"
    assert "/nope" in entry["message"]
