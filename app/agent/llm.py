"""Provider-neutral LLM interface used by the agent loop.

The loop only ever sees the message types defined here. Each provider adapter
(GeminiClient in app/agent/gemini.py) translates them at its own boundary.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Opaque provider data that must be replayed with this call (e.g. Gemini's thought
    # signature). The loop never reads it.
    provider_data: dict[str, Any] | None = None
    # True for calls our code recorded itself (e.g. an action the customer confirmed in the
    # app), as opposed to calls the model requested.
    injected: bool = False


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    text: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    text: str


class ToolCallMessage(BaseModel):
    """One model turn that requested one or more tool calls."""

    role: Literal["tool_call"] = "tool_call"
    calls: list[ToolCall]


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    result: dict[str, Any]


Message = UserMessage | AssistantMessage | ToolCallMessage | ToolResultMessage

# Parses any stored message back into the right class, using `role` as the discriminator.
MESSAGE_ADAPTER: TypeAdapter[Message] = TypeAdapter(Annotated[Message, Field(discriminator="role")])


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class LLMResponse(BaseModel):
    """Either a text answer or a list of tool calls (never both empty)."""

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class LLMError(Exception):
    """The model could not produce a response."""


class LLMRateLimited(LLMError):
    """Still rate limited (429) after retries."""


class LLMUnavailable(LLMError):
    """Provider errors (5xx/network) persisted after retries."""


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self, system: str, messages: Sequence[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        """Send the conversation and the offered tool schemas; return the model's turn."""

    async def generate_stream(
        self, system: str, messages: Sequence[Message], tools: list[dict[str, Any]]
    ) -> AsyncIterator[str | LLMResponse]:
        """Like generate, but yields text pieces as they arrive, then the full LLMResponse.

        The final LLMResponse is always the last item and has the complete text (or the
        tool calls) and usage. This default streams nothing until the response is complete.
        """
        response = await self.generate(system, messages, tools)
        if response.text:
            yield response.text
        yield response


ScriptStep = LLMResponse | Exception | Callable[[], Awaitable[LLMResponse]]


class ScriptedLLM(LLMClient):
    """Test double: replays a fixed sequence of responses and records every request.

    A step can be an LLMResponse, an exception to raise, or an async callable (useful
    for simulating a slow model).
    """

    def __init__(self, steps: Sequence[ScriptStep], *, chunk_size: int | None = None) -> None:
        self._steps = list(steps)
        self.requests: list[dict[str, Any]] = []
        # Streaming mode: split text answers into pieces of this many characters.
        self.chunk_size = chunk_size

    async def generate(
        self, system: str, messages: Sequence[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.requests.append(
            {"system": system, "messages": [m.model_copy() for m in messages], "tools": tools}
        )
        if not self._steps:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return await step()
        return step

    async def generate_stream(
        self, system: str, messages: Sequence[Message], tools: list[dict[str, Any]]
    ) -> AsyncIterator[str | LLMResponse]:
        response = await self.generate(system, messages, tools)
        if response.text:
            size = self.chunk_size or len(response.text)
            for start in range(0, len(response.text), size):
                yield response.text[start : start + size]
        yield response
