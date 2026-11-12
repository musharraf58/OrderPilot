"""The agent loop: model -> tools -> model ... until the model answers in text."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from app.agent.llm import (
    AssistantMessage,
    LLMClient,
    LLMError,
    LLMResponse,
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


AWAITING_CONFIRMATION = ToolResult.failure(
    "awaiting_confirmation",
    "The customer has been asked to confirm this action in the app. It has not been done.",
)
ONLY_ONE_ACTION = ToolResult.failure(
    "not_allowed", "Only one action can be proposed at a time; this one was not proposed."
)


def confirmation_text(summary: str) -> str:
    """What the customer reads when an action is proposed. Written by code, not the model."""
    return f"{summary}\n\nNothing has been changed yet. Please confirm or cancel this action."


class ActionProposal(BaseModel):
    """A write the model asked for. Not executed: the API stores it for confirmation."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]  # validated by the tool's input model, then fixed
    summary: str


class AgentResult(BaseModel):
    final_text: str
    trace: Trace
    usage: Usage
    # The conversation to pass as `history` on the customer's next turn.
    history: list[Message]
    # Set when the turn ended by proposing a write action for the customer to confirm.
    proposal: ActionProposal | None = None


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolStarted(BaseModel):
    type: Literal["tool_started"] = "tool_started"
    name: str
    call_id: str


class ToolFinished(BaseModel):
    type: Literal["tool_finished"] = "tool_finished"
    name: str
    call_id: str
    ok: bool


class Finished(BaseModel):
    type: Literal["finished"] = "finished"
    result: AgentResult


AgentEvent = TextDelta | ToolStarted | ToolFinished | Finished


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
        self.system_prompt = build_system_prompt(
            allow_writes=allow_writes, policies=registry.get("search_policies") is not None
        )

    def _is_offered(self, name: str) -> bool:
        tool = self.registry.get(name)
        return tool is not None and (self.allow_writes or tool.kind == "read")

    def offered_tools(self) -> list[dict[str, Any]]:
        """Schemas shown to the model. Write tools are hidden unless writes are allowed."""
        return [s for s in self.registry.schemas() if self._is_offered(s["name"])]

    async def run(
        self, user_message: str, history: Sequence[Message], ctx: UserContext
    ) -> AgentResult:
        """Run one customer turn to completion (the non-streaming view of run_stream)."""
        async for event in self.run_stream(user_message, history, ctx):
            if isinstance(event, Finished):
                return event.result
        raise RuntimeError("run_stream ended without a Finished event")  # pragma: no cover

    async def run_stream(
        self, user_message: str, history: Sequence[Message], ctx: UserContext
    ) -> AsyncIterator[AgentEvent]:
        """Run one customer turn, yielding progress events and finally Finished(result).

        Every await is bounded by one shared deadline (asyncio.timeout_at around each
        individual await, never around a yield: a timeout that spans a yield would fire
        while the consumer is busy and cancel the consumer's code instead of ours).
        """
        messages: list[Message] = [*history, UserMessage(text=user_message)]
        trace = Trace()
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_s
        tools = self.offered_tools()
        final_text: str | None = None
        proposal: ActionProposal | None = None

        try:
            for _ in range(self.max_steps):
                started = time.perf_counter()
                response: LLMResponse | None = None
                stream = self.llm.generate_stream(self.system_prompt, messages, tools)
                try:
                    while True:
                        async with asyncio.timeout_at(deadline):
                            item = await anext(stream, None)
                        if item is None:
                            break
                        if isinstance(item, LLMResponse):
                            response = item
                        else:
                            yield TextDelta(text=item)
                finally:
                    await stream.aclose()
                if response is None:
                    raise LLMError("model stream ended without a response")

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
                    final_text = response.text or ""
                    break

                messages.append(ToolCallMessage(calls=response.tool_calls))
                # Sequential on purpose: deterministic order, and no bursts on the store.
                for call in response.tool_calls:
                    if self._is_write(call.name) and self.allow_writes:
                        # Writes are never executed here: at most one becomes a proposal.
                        async with asyncio.timeout_at(deadline):
                            result, new_proposal = await self._propose(
                                call, trace, ctx, already_proposed=proposal is not None
                            )
                        proposal = proposal or new_proposal
                        if new_proposal is None:
                            yield ToolStarted(name=call.name, call_id=call.id)
                            yield ToolFinished(name=call.name, call_id=call.id, ok=False)
                    else:
                        yield ToolStarted(name=call.name, call_id=call.id)
                        async with asyncio.timeout_at(deadline):
                            result = await self._execute(call, trace, ctx)
                        yield ToolFinished(name=call.name, call_id=call.id, ok=result.ok)
                    messages.append(
                        ToolResultMessage(
                            call_id=call.id, name=call.name, result=result.model_dump(mode="json")
                        )
                    )
                if proposal is not None:
                    # End the turn: the customer decides, through the API, not the model.
                    final_text = confirmation_text(proposal.summary)
                    break
        except TimeoutError:
            if loop.time() < deadline:
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
        yield Finished(
            result=AgentResult(
                final_text=final_text,
                trace=trace,
                usage=trace.usage,
                history=messages,
                proposal=proposal if trace.outcome == "completed" else None,
            )
        )

    def _is_write(self, name: str) -> bool:
        tool = self.registry.get(name)
        return tool is not None and tool.kind == "write"

    async def _propose(
        self, call: ToolCall, trace: Trace, ctx: UserContext, *, already_proposed: bool
    ) -> tuple[ToolResult, ActionProposal | None]:
        """Turn a write call into a proposal (never executing it), or explain why not."""
        started = time.perf_counter()
        proposal = None
        if already_proposed:
            result = ONLY_ONE_ACTION
        else:
            prepared = self.registry.prepare(call.name, call.arguments)
            if isinstance(prepared, ToolResult):
                result = prepared
            else:
                tool, args = prepared
                # Checks ownership and feasibility on fresh store data; refuses early if not.
                described = await tool.propose(args, ctx)
                if described.ok:
                    proposal = ActionProposal(
                        call_id=call.id,
                        tool_name=tool.name,
                        arguments=args.model_dump(mode="json"),
                        summary=described.data["summary"],
                    )
                    result = AWAITING_CONFIRMATION
                else:
                    result = described
        trace.steps.append(
            ToolCallStep(
                call_id=call.id,
                name=call.name,
                arguments=call.arguments,
                result=result,
                latency_ms=_ms_since(started),
                proposed=proposal is not None,
            )
        )
        return result, proposal

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
