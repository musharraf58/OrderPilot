"""Gemini adapter for the provider-neutral LLMClient interface.

All google-genai types stay inside this module: the agent loop only sees app.agent.llm.
"""

import asyncio
import base64
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any
from uuid import uuid4

import httpx
from google import genai
from google.genai import errors, types

from app.agent.llm import (
    AssistantMessage,
    LLMClient,
    LLMError,
    LLMRateLimited,
    LLMResponse,
    LLMUnavailable,
    Message,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)

logger = logging.getLogger(__name__)

# Keys inside ToolCall.provider_data.
_NATIVE_ID = "gemini_id"
_SIGNATURE = "gemini_thought_signature"
# Google's documented placeholder signature for function calls the application adds to the
# history itself. Verified against the live API: accepted, while a made-up value is rejected
# with "Corrupted thought signature".
_INJECTED_SIGNATURE = b"skip_thought_signature_validator"


def to_gemini_contents(messages: Sequence[Message]) -> list[types.Content]:
    """Translate our history into Gemini contents.

    Consecutive tool results become one turn, because Gemini expects all the responses to
    a (possibly parallel) function-call turn together. That turn must use role "user":
    verified against the live API, role "tool" is rejected with a 400 even though the SDK
    docs show it.
    """
    calls_by_id: dict[str, ToolCall] = {}
    contents: list[types.Content] = []
    for message in messages:
        match message:
            case UserMessage(text=text):
                contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
            case AssistantMessage(text=text):
                contents.append(types.Content(role="model", parts=[types.Part(text=text)]))
            case ToolCallMessage(calls=calls):
                parts = []
                for call in calls:
                    calls_by_id[call.id] = call
                    data = call.provider_data or {}
                    signature = data.get(_SIGNATURE)
                    if signature:
                        thought_signature = base64.b64decode(signature)
                    else:
                        thought_signature = _INJECTED_SIGNATURE if call.injected else None
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=data.get(_NATIVE_ID), name=call.name, args=call.arguments
                            ),
                            thought_signature=thought_signature,
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
            case ToolResultMessage(call_id=call_id, name=name, result=result):
                original = calls_by_id.get(call_id)
                native_id = (original.provider_data or {}).get(_NATIVE_ID) if original else None
                part = types.Part(
                    function_response=types.FunctionResponse(
                        id=native_id, name=name, response=result
                    )
                )
                previous = contents[-1] if contents else None
                if previous is not None and previous.parts and previous.parts[0].function_response:
                    previous.parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
    return contents


def from_gemini_response(response: types.GenerateContentResponse) -> LLMResponse:
    meta = response.usage_metadata
    usage = Usage(
        input_tokens=(meta.prompt_token_count or 0) if meta else 0,
        # Thinking tokens are billed as output, so count them.
        output_tokens=((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
        if meta
        else 0,
    )
    candidate = response.candidates[0] if response.candidates else None
    parts = (candidate.content.parts if candidate and candidate.content else None) or []

    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    for part in parts:
        if part.function_call:
            fc = part.function_call
            provider_data: dict[str, Any] = {}
            if fc.id:
                provider_data[_NATIVE_ID] = fc.id
            if part.thought_signature:
                provider_data[_SIGNATURE] = base64.b64encode(part.thought_signature).decode()
            tool_calls.append(
                ToolCall(
                    id=fc.id or f"call_{uuid4().hex[:12]}",
                    name=fc.name or "",
                    arguments=dict(fc.args or {}),
                    provider_data=provider_data or None,
                )
            )
        elif part.text and not part.thought:
            texts.append(part.text)

    if tool_calls:
        return LLMResponse(tool_calls=tool_calls, usage=usage)
    text = "".join(texts).strip()
    if not text:
        reason = candidate.finish_reason if candidate else "no candidates"
        raise LLMError(f"model returned no text or tool calls (finish reason: {reason})")
    return LLMResponse(text=text, usage=usage)


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_attempts: int = 4,
        backoff: float = 1.0,
        max_backoff: float = 16.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_attempts = max_attempts
        self._backoff = backoff
        self._max_backoff = max_backoff
        self._sleep = sleep

    async def _call(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        return await self._client.aio.models.generate_content(
            model=self._model, contents=contents, config=config
        )

    async def _call_stream(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> AsyncIterator[types.GenerateContentResponse]:
        return await self._client.aio.models.generate_content_stream(
            model=self._model, contents=contents, config=config
        )

    def _config(self, system: str, tools: list[dict[str, Any]]) -> types.GenerateContentConfig:
        declarations = [
            types.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
            )
            for t in tools
        ]
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            # We run tools ourselves in the agent loop; never let the SDK do it.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def _with_retries[T](self, attempt_fn: Callable[[], Awaitable[T]]) -> T:
        """Run attempt_fn, retrying 429/5xx/network errors with exponential backoff."""
        error: LLMError = LLMUnavailable("no attempts made")
        for attempt in range(self._max_attempts):
            if attempt:
                delay = min(self._backoff * 2 ** (attempt - 1), self._max_backoff)
                await self._sleep(delay + random.uniform(0, delay / 4))
            try:
                return await attempt_fn()
            except errors.APIError as exc:
                if exc.code == 429:
                    error = LLMRateLimited("Gemini rate limit exceeded")
                elif exc.code and exc.code >= 500:
                    error = LLMUnavailable(f"Gemini server error {exc.code}")
                else:
                    raise LLMError(f"Gemini rejected the request ({exc.code})") from exc
                logger.warning("gemini attempt %d failed: %s", attempt + 1, exc.code)
            except (httpx.HTTPError, TimeoutError) as exc:
                error = LLMUnavailable(f"Gemini unreachable: {type(exc).__name__}")
                logger.warning("gemini attempt %d failed: %s", attempt + 1, type(exc).__name__)
        raise error

    async def generate(
        self, system: str, messages: Sequence[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        contents = to_gemini_contents(messages)
        config = self._config(system, tools)
        response = await self._with_retries(lambda: self._call(contents, config))
        return from_gemini_response(response)

    async def generate_stream(
        self, system: str, messages: Sequence[Message], tools: list[dict[str, Any]]
    ) -> AsyncIterator[str | LLMResponse]:
        """Stream text as it arrives; collect function calls and return them at the end.

        Verified against the live API: function calls arrive as whole parts (the first of
        a parallel batch carries the thought signature), usage_metadata is cumulative so
        the last chunk's value is the total, and empty text parts can appear.
        Retries only happen before the first chunk: once text has been yielded, retrying
        would repeat it, so a later failure becomes an LLMError.
        """
        contents = to_gemini_contents(messages)
        config = self._config(system, tools)

        async def open_stream() -> tuple[Any, types.GenerateContentResponse | None]:
            stream = await self._call_stream(contents, config)
            first = await anext(aiter(stream), None)
            return stream, first

        stream, chunk = await self._with_retries(open_stream)
        parts: list[types.Part] = []
        usage = None
        finish_reason = None
        try:
            while chunk is not None:
                candidate = chunk.candidates[0] if chunk.candidates else None
                for part in (
                    candidate.content.parts if candidate and candidate.content else None
                ) or []:
                    parts.append(part)
                    if part.text and not part.thought and not part.function_call:
                        yield part.text
                usage = chunk.usage_metadata or usage
                if candidate and candidate.finish_reason:
                    finish_reason = candidate.finish_reason
                chunk = await anext(aiter(stream), None)
        except (errors.APIError, httpx.HTTPError, TimeoutError) as exc:
            raise LLMUnavailable(f"Gemini stream failed: {type(exc).__name__}") from exc

        yield from_gemini_response(
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(role="model", parts=parts),
                        finish_reason=finish_reason,
                    )
                ],
                usage_metadata=usage,
            )
        )
