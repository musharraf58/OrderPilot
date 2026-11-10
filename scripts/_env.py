"""Shared helpers for the manual scripts (never imported by the app or tests)."""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_env() -> dict[str, str]:
    """Values from .env, overridden by real environment variables. Never print these."""
    values = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
    values.update({k: v for k, v in os.environ.items() if k.startswith(("STORE_", "GEMINI_"))})
    return values


async def store_token(client, env: dict[str, str]) -> str | None:
    """STORE_TOKEN if set, otherwise log in with STORE_EMAIL / STORE_PASSWORD."""
    if env.get("STORE_TOKEN"):
        return env["STORE_TOKEN"]
    if env.get("STORE_EMAIL") and env.get("STORE_PASSWORD"):
        return (await client.login(env["STORE_EMAIL"], env["STORE_PASSWORD"])).access_token
    return None
