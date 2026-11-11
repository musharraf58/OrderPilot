import base64
from unittest.mock import AsyncMock

import httpx
import pytest
from google.genai import errors, types

from app.agent.gemini import GeminiClient, from_gemini_response, to_gemini_contents
from app.agent.llm import (
    AssistantMessage,
    LLMError,
    LLMRateLimited,
    LLMUnavailable,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)

SIG = b"\x01signature\xff"


def fc_response(*calls, text=None, usage=(10, 3, 2)):
    parts = [
        types.Part(
            function_call=types.FunctionCall(id=cid, name=name, args=args),
            thought_signature=sig,
        )
        for cid, name, args, sig in calls
    ]
    if text:
        parts.append(types.Part(text=text))
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=usage[0],
            candidates_token_count=usage[1],
            thoughts_token_count=usage[2],
        ),
    )


def test_parses_parallel_tool_calls_and_keeps_signature():
    resp = from_gemini_response(
        fc_response(("g1", "get_order", {"order_id": "a"}, SIG), (None, "list_my_orders", {}, None))
    )
    assert [c.name for c in resp.tool_calls] == ["get_order", "list_my_orders"]
    first, second = resp.tool_calls
    assert first.id == "g1"
    assert base64.b64decode(first.provider_data["gemini_thought_signature"]) == SIG
    assert second.id.startswith("call_")
    assert second.provider_data is None
    assert resp.text is None
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (10, 5)


def test_parses_text_and_ignores_thought_parts():
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="thinking...", thought=True), types.Part(text="Hi!")],
                )
            )
        ]
    )
    assert from_gemini_response(response).text == "Hi!"


def test_empty_response_is_llm_error():
    with pytest.raises(LLMError):
        from_gemini_response(types.GenerateContentResponse(candidates=[]))


def test_history_round_trip_replays_signature_and_groups_results():
    call_a = ToolCall(
        id="g1",
        name="get_order",
        arguments={"order_id": "a"},
        provider_data={
            "gemini_id": "g1",
            "gemini_thought_signature": base64.b64encode(SIG).decode(),
        },
    )
    call_b = ToolCall(id="call_local", name="list_my_orders", arguments={})
    contents = to_gemini_contents(
        [
            UserMessage(text="hi"),
            ToolCallMessage(calls=[call_a, call_b]),
            ToolResultMessage(call_id="g1", name="get_order", result={"ok": True}),
            ToolResultMessage(call_id="call_local", name="list_my_orders", result={"ok": True}),
            AssistantMessage(text="done"),
        ]
    )
    # Function responses use role "user": Gemini rejects role "tool" (verified live).
    assert [c.role for c in contents] == ["user", "model", "user", "model"]
    model_parts = contents[1].parts
    assert model_parts[0].thought_signature == SIG
    assert model_parts[0].function_call.id == "g1"
    assert model_parts[1].function_call.id is None  # locally generated ids are not sent
    responses = [p.function_response for p in contents[2].parts]
    assert [(r.id, r.name) for r in responses] == [("g1", "get_order"), (None, "list_my_orders")]


@pytest.fixture
def client():
    return GeminiClient(api_key="test-key", model="gemini-test", sleep=AsyncMock())


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}})


async def test_retries_429_then_succeeds(client, monkeypatch):
    call = AsyncMock(side_effect=[api_error(429), api_error(503), fc_response(text="ok")])
    monkeypatch.setattr(client, "_call", call)
    resp = await client.generate("sys", [UserMessage(text="hi")], [])
    assert resp.text == "ok"
    assert call.await_count == 3
    assert client._sleep.await_count == 2


async def test_gives_up_with_typed_error_on_persistent_429(client, monkeypatch):
    monkeypatch.setattr(client, "_call", AsyncMock(side_effect=api_error(429)))
    with pytest.raises(LLMRateLimited):
        await client.generate("sys", [UserMessage(text="hi")], [])
    assert client._call.await_count == 4


async def test_gives_up_with_typed_error_on_network_failure(client, monkeypatch):
    monkeypatch.setattr(client, "_call", AsyncMock(side_effect=httpx.ConnectError("down")))
    with pytest.raises(LLMUnavailable):
        await client.generate("sys", [UserMessage(text="hi")], [])


async def test_4xx_is_not_retried(client, monkeypatch):
    monkeypatch.setattr(client, "_call", AsyncMock(side_effect=api_error(400)))
    with pytest.raises(LLMError) as info:
        await client.generate("sys", [UserMessage(text="hi")], [])
    assert type(info.value) is LLMError
    assert client._call.await_count == 1


def test_config_offers_tools_and_disables_automatic_calling(client):
    config = client._config(
        "sys",
        [{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}],
    )
    assert config.automatic_function_calling.disable is True
    decl = config.tools[0].function_declarations[0]
    assert (decl.name, decl.parameters_json_schema["type"]) == ("t", "object")
    assert client._config("sys", []).tools is None


# --- streaming ----------------------------------------------------------------


def chunk(*parts, prompt=10, cand=0, finish=None):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)), finish_reason=finish
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=cand
        ),
    )


def stream_of(*chunks, fail_after=None):
    async def gen():
        for i, c in enumerate(chunks):
            if fail_after is not None and i == fail_after:
                raise httpx.ReadError("connection dropped")
            yield c

    return gen()


async def collect(client, **kwargs):
    return [item async for item in client.generate_stream("sys", [UserMessage(text="hi")], [])]


async def test_stream_yields_text_pieces_then_full_response(client, monkeypatch):
    chunks = [
        chunk(types.Part(text="Hel"), cand=1),
        chunk(types.Part(text="lo!"), cand=2),
        chunk(types.Part(text="", thought_signature=SIG), cand=2, finish="STOP"),
    ]
    monkeypatch.setattr(client, "_call_stream", AsyncMock(return_value=stream_of(*chunks)))
    items = await collect(client)
    assert items[:-1] == ["Hel", "lo!"]
    final = items[-1]
    assert final.text == "Hello!"
    # Usage is cumulative per chunk: the last value is the total, not a sum.
    assert (final.usage.input_tokens, final.usage.output_tokens) == (10, 2)


async def test_stream_collects_function_calls_with_signature(client, monkeypatch):
    chunks = [
        chunk(
            types.Part(
                function_call=types.FunctionCall(id="g1", name="get_order", args={"order_id": "A"}),
                thought_signature=SIG,
            )
        ),
        chunk(types.Part(function_call=types.FunctionCall(id="g2", name="get_order", args={}))),
        chunk(types.Part(text=""), finish="STOP"),
    ]
    monkeypatch.setattr(client, "_call_stream", AsyncMock(return_value=stream_of(*chunks)))
    items = await collect(client)
    assert len(items) == 1  # no text pieces
    calls = items[0].tool_calls
    assert [c.id for c in calls] == ["g1", "g2"]
    assert base64.b64decode(calls[0].provider_data["gemini_thought_signature"]) == SIG


async def test_stream_retries_before_first_chunk(client, monkeypatch):
    call = AsyncMock(side_effect=[api_error(503), stream_of(chunk(types.Part(text="ok")))])
    monkeypatch.setattr(client, "_call_stream", call)
    items = await collect(client)
    assert items[-1].text == "ok"
    assert call.await_count == 2


async def test_stream_failure_after_text_is_not_retried(client, monkeypatch):
    call = AsyncMock(
        return_value=stream_of(
            chunk(types.Part(text="partial")), chunk(types.Part(text="x")), fail_after=1
        )
    )
    monkeypatch.setattr(client, "_call_stream", call)
    seen = []
    with pytest.raises(LLMUnavailable):
        async for item in client.generate_stream("sys", [UserMessage(text="hi")], []):
            seen.append(item)
    assert seen == ["partial"]
    assert call.await_count == 1


async def test_scripted_llm_streaming_mode_chunks_text():
    from app.agent.llm import LLMResponse, ScriptedLLM

    llm = ScriptedLLM([LLMResponse(text="abcdefg")], chunk_size=3)
    items = [i async for i in llm.generate_stream("sys", [UserMessage(text="hi")], [])]
    assert items[:-1] == ["abc", "def", "g"]
    assert items[-1].text == "abcdefg"


def test_injected_calls_get_the_documented_placeholder_signature():
    contents = to_gemini_contents(
        [
            UserMessage(text="cancel it"),
            ToolCallMessage(
                calls=[ToolCall(id="action_1", name="cancel_order", arguments={}, injected=True)]
            ),
            ToolResultMessage(call_id="action_1", name="cancel_order", result={"ok": True}),
        ]
    )
    assert contents[1].parts[0].thought_signature == b"skip_thought_signature_validator"
    assert contents[1].parts[0].function_call.id is None


def test_model_calls_without_signature_stay_unsigned():
    contents = to_gemini_contents(
        [ToolCallMessage(calls=[ToolCall(id="c", name="t", arguments={})])]
    )
    assert contents[0].parts[0].thought_signature is None


@pytest.mark.parametrize("exc", [httpx.ConnectError("x"), "httpx2"])
async def test_network_errors_from_either_httpx_are_retried(client, monkeypatch, exc):
    # google-genai 3.x raises httpx2 exceptions, which are not httpx exceptions.
    if exc == "httpx2":
        import httpx2

        exc = httpx2.ConnectError("All connection attempts failed")
    call = AsyncMock(side_effect=[exc, fc_response(text="ok")])
    monkeypatch.setattr(client, "_call", call)
    assert (await client.generate("sys", [UserMessage(text="hi")], [])).text == "ok"
    stream_call = AsyncMock(side_effect=[exc, stream_of(chunk(types.Part(text="ok")))])
    monkeypatch.setattr(client, "_call_stream", stream_call)
    assert (await collect(client))[-1].text == "ok"
