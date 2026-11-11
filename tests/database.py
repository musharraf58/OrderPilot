"""Test database helpers: a real Postgres (JSONB), migrated with Alembic once per session.

Set TEST_DATABASE_URL to point at it. Locally that's the docker compose Postgres; in CI a
service container. If it is unreachable the DB tests are skipped locally but fail in CI.
"""

import asyncio
import os
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parent.parent
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://orderpilot:orderpilot@localhost:5432/orderpilot_test",
)


async def _ensure_database(url: str) -> None:
    parsed = make_url(url)
    conn = await asyncpg.connect(
        user=parsed.username,
        password=parsed.password,
        host=parsed.host,
        port=parsed.port,
        database="postgres",
        timeout=5,
    )
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", parsed.database
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{parsed.database}"')
    finally:
        await conn.close()


def migrate_test_database() -> str:
    """Create the test DB if needed and run all migrations. Returns the URL."""
    try:
        asyncio.run(_ensure_database(TEST_DATABASE_URL))
    except (OSError, asyncpg.PostgresError) as exc:
        message = (
            f"test database unreachable at {make_url(TEST_DATABASE_URL).render_as_string()}: {exc}"
        )
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)

    from alembic.config import Config

    from alembic import command

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.attributes["database_url"] = TEST_DATABASE_URL
    config.attributes["skip_logging_config"] = True
    command.upgrade(config, "head")
    return TEST_DATABASE_URL
