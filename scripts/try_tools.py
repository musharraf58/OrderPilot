"""Manually exercise the tool registry against the real, deployed store API.

Not part of the test suite and never run in CI (it needs real credentials and network).

Reads STORE_API_URL plus either STORE_TOKEN or STORE_EMAIL + STORE_PASSWORD from the
environment or the local .env file. Credentials are never printed.

    uv run python scripts/try_tools.py
    uv run python scripts/try_tools.py --cancel <order_id>   # actually cancels that order
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.clients.store import StoreClient, StoreError  # noqa: E402
from app.tools.context import UserContext  # noqa: E402
from app.tools.registry import ToolRegistry, build_default_registry  # noqa: E402


def load_env() -> dict[str, str]:
    values = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
    values.update({k: v for k, v in os.environ.items() if k.startswith("STORE_")})
    return values


async def call(registry: ToolRegistry, ctx: UserContext, name: str, args: dict) -> dict | None:
    print(f"\n>>> {name}({json.dumps(args)})")
    result = await registry.execute(name, args, ctx)
    print(json.dumps(result.model_dump(), indent=2, default=str))
    return result.data if result.ok else None


async def main(cancel_id: str | None) -> int:
    env = load_env()
    base_url = env.get("STORE_API_URL")
    if not base_url:
        print("STORE_API_URL is not set (put it in .env).")
        return 2

    print(f"Store: {base_url}  (first call may be slow if the server is asleep)")
    async with StoreClient(base_url) as client:
        try:
            token = env.get("STORE_TOKEN")
            if not token:
                if not (env.get("STORE_EMAIL") and env.get("STORE_PASSWORD")):
                    print("Set STORE_TOKEN, or STORE_EMAIL and STORE_PASSWORD.")
                    return 2
                token = (await client.login(env["STORE_EMAIL"], env["STORE_PASSWORD"])).access_token
            ctx = await UserContext.from_token(client, token)
        except StoreError as exc:
            print(f"Could not authenticate: {exc.message}")
            return 1

        print(f"Authenticated as customer {ctx.customer_id}")
        registry = build_default_registry(client)
        print(f"Registry: {[(t.name, t.kind) for t in registry.list_tools()]}")

        found = await call(registry, ctx, "search_products", {"limit": 3})
        if found and found["products"]:
            await call(registry, ctx, "get_product", {"product_id": found["products"][0]["id"]})

        listing = await call(registry, ctx, "list_my_orders", {"limit": 3})
        if listing and listing["orders"]:
            await call(registry, ctx, "get_order", {"order_id": listing["orders"][0]["id"]})

        print("\n--- safety checks ---")
        await call(registry, ctx, "get_order", {"order_id": str(uuid4())})
        await call(registry, ctx, "cancel_order", {"order_id": str(uuid4())})
        await call(registry, ctx, "get_order", {"order_id": "not-a-uuid"})
        await call(registry, ctx, "no_such_tool", {})

        if cancel_id:
            print("\n--- --cancel given: this really cancels the order ---")
            await call(registry, ctx, "cancel_order", {"order_id": cancel_id})
        else:
            print("\n(cancel_order not run for real; pass --cancel <order_id> to try it)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cancel", metavar="ORDER_ID", help="really cancel this order")
    sys.exit(asyncio.run(main(parser.parse_args().cancel)))
