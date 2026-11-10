"""Terminal chat with the OrderPilot agent, against the real store and real Gemini.

Not part of the test suite and never run in CI (it needs real credentials and uses the
free-tier Gemini quota). Reads STORE_API_URL, STORE_EMAIL/STORE_PASSWORD (or STORE_TOKEN),
GEMINI_API_KEY and optionally GEMINI_MODEL from .env.

    uv run python scripts/chat.py            # type 'exit' or Ctrl-D to quit
    uv run python scripts/chat.py --trace    # also print every step after each answer
"""

import argparse
import asyncio
import json
import sys
import warnings

from _env import load_env, store_token

from app.agent.llm import Message
from app.agent.loop import Agent
from app.agent.trace import ModelCallStep, Trace
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

        agent = Agent(GeminiClient(env["GEMINI_API_KEY"], model), build_default_registry(store))
        print(
            f"OrderPilot ({model}). Signed in as customer {ctx.customer_id}. Type 'exit' to quit."
        )
        history: list[Message] = []
        while True:
            try:
                question = (await asyncio.to_thread(input, "\nyou> ")).strip()
            except EOFError:
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
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
    sys.exit(asyncio.run(main(parser.parse_args().trace)))
