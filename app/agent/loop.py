"""The agent loop: model -> tools -> model ... until the model answers in text."""

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from app.agent.llm import (
    AssistantMessage,
    LLMClient,
    LLMError,
    Message,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from app.agent.prompts import build_system_prompt
from app.agent.trace import ModelCallStep, ToolCallStep, Trace
from app.tools.base import ToolResult
from app.tools.context import UserContext
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_STEPS_MESSAGE = (
    "Sorry, I couldn't finish working that out. Could you try asking in a simpler way?"
)
TIMEOUT_MESSAGE = "Sorry, that took too long. Please try again in a moment."
ERROR_MESSAGE = "Sorry, I'm having trouble answering right now. Please try again shortly."


class AgentResult(BaseModel):
    final_text: str
    trace: Trace
    usage: Usage
    # The conversation to pass as `history` on the customer's next turn.
    history: list[Message]


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
        timeout_s: float = 60.0,
        allow_writes: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.allow_writes = allow_writes
        self.system_prompt = build_system_prompt(allow_writes=allow_writes)

    def _is_offered(self, name: str) -> bool:
        tool = self.registry.get(name)
        return tool is not None and (self.allow_writes or tool.kind == "read")

    def offered_tools(self) -> list[dict[str, Any]]:
        """Schemas shown to the model. Write tools are hidden unless writes are allowed."""
        return [s for s in self.registry.schemas() if self._is_offered(s["name"])]

    async def run(
        self, user_message: str, history: Sequence[Message], ctx: UserContext
    ) -> AgentResult:
        messages: list[Message] = [*history, UserMessage(text=user_message)]
        trace = Trace()
        start = time.perf_counter()
        timeout = asyncio.timeout(self.timeout_s)
        final_text: str | None = None
        try:
            async with timeout:
                final_text = await self._loop(messages, trace, ctx)
        except TimeoutError:
            if not timeout.expired():
                raise
            trace.outcome, trace.reason = "truncated", "timeout"
        except LLMError as exc:
            logger.warning("agent run failed: %s", exc)
            trace.outcome, trace.reason = "error", f"llm_error: {exc}"

        if final_text is None and trace.outcome is None:
            trace.outcome, trace.reason = "truncated", "max_steps"
        if trace.outcome is None:
            trace.outcome = "completed"
            messages.append(AssistantMessage(text=final_text))
        else:
            final_text = {
                "max_steps": MAX_STEPS_MESSAGE,
                "timeout": TIMEOUT_MESSAGE,
            }.get(trace.reason or "", ERROR_MESSAGE)
            # Keep the next turn's history clean and valid: drop this turn's unfinished
            # tool exchanges and keep only the question and our fallback answer.
            messages = [
                *history,
                UserMessage(text=user_message),
                AssistantMessage(text=final_text),
            ]

        trace.total_latency_ms = _ms_since(start)
        return AgentResult(final_text=final_text, trace=trace, usage=trace.usage, history=messages)

    async def _loop(self, messages: list[Message], trace: Trace, ctx: UserContext) -> str | None:
        tools = self.offered_tools()
        for _ in range(self.max_steps):
            started = time.perf_counter()
            response = await self.llm.generate(self.system_prompt, messages, tools)
            trace.usage = trace.usage + response.usage
            trace.steps.append(
                ModelCallStep(
                    latency_ms=_ms_since(started),
                    usage=response.usage,
                    response="tool_calls" if response.tool_calls else "text",
                    tool_calls=[c.name for c in response.tool_calls],
                )
            )
            if not response.tool_calls:
                return response.text or ""

            messages.append(ToolCallMessage(calls=response.tool_calls))
            # Sequential on purpose: deterministic order, and no bursts on the store.
            for call in response.tool_calls:
                result = await self._execute(call, trace, ctx)
                messages.append(
                    ToolResultMessage(
                        call_id=call.id, name=call.name, result=result.model_dump(mode="json")
                    )
                )
        return None

    async def _execute(self, call: ToolCall, trace: Trace, ctx: UserContext) -> ToolResult:
        started = time.perf_counter()
        tool = self.registry.get(call.name)
        refused = tool is not None and not self._is_offered(call.name)
        if refused:
            result = ToolResult.failure(
                "not_allowed", "This action is not available in this conversation."
            )
        else:
            # ctx always comes from our code; the model only supplies `arguments`.
            result = await self.registry.execute(call.name, call.arguments, ctx)
        trace.steps.append(
            ToolCallStep(
                call_id=call.id,
                name=call.name,
                arguments=call.arguments,
                result=result,
                latency_ms=_ms_since(started),
                refused=refused,
            )
        )
        return result
