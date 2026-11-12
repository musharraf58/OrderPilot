"""Terminal chat with the OrderPilot agent, against the real store and real Gemini.

Not part of the test suite and never run in CI (it needs real credentials and uses the
free-tier Gemini quota). Reads STORE_API_URL, STORE_EMAIL/STORE_PASSWORD (or STORE_TOKEN),
GEMINI_API_KEY and optionally GEMINI_MODEL from .env (and DOCUMIND_* for policy answers).

    uv run python scripts/chat.py            # type 'exit' or Ctrl-D to quit
    uv run python scripts/chat.py --trace    # also print every step after each answer

With --api it talks to the running OrderPilot API over HTTP instead (login, a new
conversation, answers streamed live over SSE). It then only needs STORE_EMAIL and
STORE_PASSWORD; the API holds the Gemini key.

    uv run python scripts/chat.py --api http://localhost:8000 [--trace]
"""

import argparse
import asyncio
import json
import sys
import warnings
from collections.abc import AsyncIterator

import httpx
from _env import load_env, store_token

from app.agent.llm import Message
from app.agent.loop import Agent
from app.agent.trace import ModelCallStep, Trace
from app.clients.documind import DocuMindClient
from app.clients.store import StoreClient, StoreError
from app.core.config import get_settings
from app.tools.context import UserContext
from app.tools.registry import build_default_registry


def print_trace(trace: Trace) -> None:
    print("  ┌─ trace")
    for step in trace.steps:
        if isinstance(step, ModelCallStep):
            what = f"tool calls: {', '.join(step.tool_calls)}" if step.tool_calls else "text"
            print(
                f"  │ model  {step.latency_ms:>7.0f} ms  "
                f"in={step.usage.input_tokens} out={step.usage.output_tokens}  -> {what}"
            )
        else:
            flag = " [REFUSED]" if step.refused else ""
            args = json.dumps(step.arguments)
            print(f"  │ tool   {step.latency_ms:>7.0f} ms  {step.name}({args}){flag}")
            print(f"  │        -> {json.dumps(step.result.model_dump(mode='json'))}")
    u = trace.usage
    print(
        f"  └─ {trace.outcome}{f' ({trace.reason})' if trace.reason else ''}  "
        f"{trace.total_latency_ms:.0f} ms  tokens in={u.input_tokens} out={u.output_tokens}"
    )


async def read_question() -> str | None:
    try:
        question = (await asyncio.to_thread(input, "\nyou> ")).strip()
    except EOFError:
        print()
        return None
    return None if question.lower() in {"exit", "quit"} else question


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str, dict]]:
    """Parse a text/event-stream response into (event, data) pairs as they arrive."""
    event, data = "message", ""
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data += line.partition(":")[2].strip()
        elif line == "" and data:
            yield event, json.loads(data)
            event, data = "message", ""


async def main_api(base_url: str, show_trace: bool) -> int:
    env = load_env()
    if not (env.get("STORE_EMAIL") and env.get("STORE_PASSWORD")):
        print("Set STORE_EMAIL and STORE_PASSWORD in .env.")
        return 2
    # Long read timeout: an agent turn can take a while (and the store may be waking up).
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(10, read=120)) as http:
        login = await http.post(
            "/auth/login", json={"email": env["STORE_EMAIL"], "password": env["STORE_PASSWORD"]}
        )
        if login.status_code != 200:
            print(f"Login failed: HTTP {login.status_code} {login.text}")
            return 1
        http.headers["Authorization"] = f"Bearer {login.json()['access_token']}"
        conversation = (await http.post("/conversations")).json()
        print(f"OrderPilot API at {base_url}, conversation {conversation['id']}. 'exit' quits.")

        while (question := await read_question()) is not None:
            if not question:
                continue
            async with http.stream(
                "POST", f"/conversations/{conversation['id']}/messages", json={"text": question}
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    print(f"HTTP {response.status_code}: {response.text}")
                    continue
                print("\norderpilot> ", end="", flush=True)
                proposal = None
                async for event, data in iter_sse(response):
                    if event == "confirmation_required":
                        proposal = data
                    elif event == "text_delta":
                        print(data["text"], end="", flush=True)
                    elif event == "error":
                        print(f"\n[{data['code']}] {data['message']}", end="", flush=True)
                    elif show_trace and event == "tool_started":
                        print(f"\n  · {data['name']}…", end="", flush=True)
                    elif show_trace and event == "tool_finished":
                        print(" ok" if data["ok"] else " failed", end="\n", flush=True)
                    elif show_trace and event == "done":
                        print(f"\n  └─ done (message {data['message_id']})", end="")
                print()
            if proposal is not None:
                await resolve_action(http, conversation["id"], proposal)
    return 0


async def resolve_action(http: httpx.AsyncClient, conversation_id: str, proposal: dict) -> None:
    """Ask the customer, then call the confirm or cancel endpoint. This separate,
    authenticated call is the only thing that can make a write happen."""
    print(f"\n  ┌─ action needs your confirmation (expires {proposal['expires_at']})")
    print(f"  │ {proposal['summary']}")
    try:
        answer = (await asyncio.to_thread(input, "  └─ confirm? [y/N] ")).strip().lower()
    except EOFError:
        answer = ""
    verb = "confirm" if answer in {"y", "yes"} else "cancel"
    response = await http.post(
        f"/conversations/{conversation_id}/actions/{proposal['action_id']}/{verb}"
    )
    if response.status_code == 200:
        print(f"\norderpilot> {response.json()['message']}")
    else:
        print(f"\nHTTP {response.status_code}: {response.json().get('detail')}")


async def main(show_trace: bool) -> int:
    env = load_env()
    missing = [k for k in ("STORE_API_URL", "GEMINI_API_KEY") if not env.get(k)]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}")
        return 2

    # Imported here so the SDK's import-time warnings don't clutter the other scripts.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from app.agent.gemini import GeminiClient

    model = env.get("GEMINI_MODEL") or get_settings().GEMINI_MODEL
    async with StoreClient(env["STORE_API_URL"]) as store:
        try:
            token = await store_token(store, env)
            if not token:
                print("Set STORE_TOKEN, or STORE_EMAIL and STORE_PASSWORD.")
                return 2
            ctx = await UserContext.from_token(store, token)
        except StoreError as exc:
            print(f"Could not sign in to the store: {exc.message}")
            return 1

        settings = get_settings()
        documind = None
        if settings.DOCUMIND_API_URL:
            documind = DocuMindClient(
                settings.DOCUMIND_API_URL,
                settings.DOCUMIND_EMAIL,
                settings.DOCUMIND_PASSWORD,
                collection_name=settings.DOCUMIND_COLLECTION,
            )
        registry = build_default_registry(
            store, documind, policy_min_relevance=settings.POLICY_MIN_RELEVANCE
        )
        agent = Agent(GeminiClient(env["GEMINI_API_KEY"], model), registry)
        print(
            f"OrderPilot ({model}). Signed in as customer {ctx.customer_id}. Type 'exit' to quit."
        )
        history: list[Message] = []
        while (question := await read_question()) is not None:
            if not question:
                continue
            result = await agent.run(question, history, ctx)
            history = result.history
            print(f"\norderpilot> {result.final_text}")
            if show_trace:
                print_trace(result.trace)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", action="store_true", help="print each step after an answer")
    parser.add_argument("--api", metavar="URL", help="talk to the running API over HTTP + SSE")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_api(args.api, args.trace) if args.api else main(args.trace)))
