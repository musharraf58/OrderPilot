import asyncio
from uuid import uuid4

import pytest

from app.agent.llm import (
    AssistantMessage,
    LLMResponse,
    LLMUnavailable,
    ScriptedLLM,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from app.agent.loop import (
    ERROR_MESSAGE,
    MAX_STEPS_MESSAGE,
    TIMEOUT_MESSAGE,
    Agent,
)
from app.agent.trace import Trace
from tests.conftest import TOKEN
from tests.factories import order, product


def text(t, usage=(5, 2)):
    return LLMResponse(text=t, usage=Usage(input_tokens=usage[0], output_tokens=usage[1]))


def calls(*specs, usage=(5, 2)):
    return LLMResponse(
        tool_calls=[
            ToolCall(id=f"c{i}", name=name, arguments=args) for i, (name, args) in enumerate(specs)
        ],
        usage=Usage(input_tokens=usage[0], output_tokens=usage[1]),
    )


def results_in(request):
    return [m for m in request["messages"] if isinstance(m, ToolResultMessage)]


@pytest.fixture
def make_agent(registry):
    def _make(script, **kwargs):
        llm = ScriptedLLM(script)
        return Agent(llm, registry, **kwargs), llm

    return _make


async def test_direct_answer_without_tools(make_agent, ctx):
    agent, llm = make_agent([text("Hi! How can I help?")])
    result = await agent.run("hello", [], ctx)
    assert result.final_text == "Hi! How can I help?"
    assert result.trace.outcome == "completed"
    assert [s.type for s in result.trace.steps] == ["model_call"]
    assert [type(m) for m in result.history] == [UserMessage, AssistantMessage]
    assert len(llm.requests) == 1


async def test_one_tool_call_then_answer(make_agent, ctx, store):
    orders_route = store.get("/orders").respond(200, json=[order(status="shipped")])
    agent, llm = make_agent([calls(("list_my_orders", {})), text("Your order has shipped.")])
    result = await agent.run("where is my order?", [], ctx)

    assert result.final_text == "Your order has shipped."
    assert orders_route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"
    fed_back = results_in(llm.requests[1])
    assert len(fed_back) == 1
    assert fed_back[0].call_id == "c0"
    assert fed_back[0].result["ok"] is True
    assert fed_back[0].result["data"]["orders"][0]["status"] == "shipped"
    assert [type(m).__name__ for m in result.history] == [
        "UserMessage",
        "ToolCallMessage",
        "ToolResultMessage",
        "AssistantMessage",
    ]


async def test_multiple_tool_calls_in_one_turn(make_agent, ctx, store):
    store.get("/orders").respond(200, json=[order()])
    store.get("/products").respond(200, json={"total": 1, "items": [product()]})
    agent, llm = make_agent(
        [
            calls(("list_my_orders", {}), ("search_products", {"query": "mug"})),
            text("Here you go."),
        ]
    )
    result = await agent.run("my orders and mugs?", [], ctx)

    assert [s.name for s in result.trace.tool_calls()] == ["list_my_orders", "search_products"]
    fed_back = results_in(llm.requests[1])
    assert [(r.call_id, r.name) for r in fed_back] == [
        ("c0", "list_my_orders"),
        ("c1", "search_products"),
    ]
    assert all(r.result["ok"] for r in fed_back)
    assert len(llm.requests) == 2  # both results went back in a single follow-up call


async def test_tool_error_is_fed_back_and_model_recovers(make_agent, ctx, store):
    pid = uuid4()
    store.get(f"/products/{pid}").respond(404)
    agent, llm = make_agent(
        [calls(("get_product", {"product_id": str(pid)})), text("I couldn't find that product.")]
    )
    result = await agent.run("tell me about that product", [], ctx)

    assert result.trace.outcome == "completed"
    assert result.final_text == "I couldn't find that product."
    fed_back = results_in(llm.requests[1])[0]
    assert fed_back.result == {
        "ok": False,
        "data": None,
        "error": {"code": "not_found", "message": "The requested item was not found."},
    }


async def test_store_down_does_not_crash_the_loop(make_agent, ctx, store):
    store.get("/orders").respond(503)
    agent, llm = make_agent([calls(("list_my_orders", {})), text("The store is down.")])
    result = await agent.run("orders?", [], ctx)
    assert result.trace.outcome == "completed"
    assert results_in(llm.requests[1])[0].result["error"]["code"] == "store_unavailable"


async def test_unknown_tool_requested_by_model(make_agent, ctx):
    agent, llm = make_agent([calls(("delete_everything", {})), text("I can't do that.")])
    result = await agent.run("wipe it", [], ctx)

    step = result.trace.tool_calls()[0]
    assert (step.name, step.result.error.code, step.refused) == (
        "delete_everything",
        "unknown_tool",
        False,
    )
    assert results_in(llm.requests[1])[0].result["error"]["code"] == "unknown_tool"
    assert result.trace.outcome == "completed"


async def test_write_tool_not_offered_and_refused_when_writes_disabled(make_agent, ctx, store):
    orders_route = store.get("/orders").respond(200, json=[order()])
    target = str(uuid4())
    cancel_route = store.post(f"/orders/{target}/cancel").respond(200, json=order(id=target))
    agent, llm = make_agent(
        [calls(("cancel_order", {"order_id": target})), text("I can't cancel orders yet.")]
    )
    result = await agent.run("cancel my order", [], ctx)

    offered = {t["name"] for t in llm.requests[0]["tools"]}
    assert offered == {"search_products", "get_product", "list_my_orders", "get_order"}
    step = result.trace.tool_calls()[0]
    assert step.refused is True
    assert step.result.error.code == "not_allowed"
    assert not cancel_route.called
    assert not orders_route.called  # refused before the tool ran at all
    assert results_in(llm.requests[1])[0].result["error"]["code"] == "not_allowed"


async def test_write_tool_offered_when_writes_allowed(make_agent, ctx):
    agent, llm = make_agent([text("ok")], allow_writes=True)
    await agent.run("hi", [], ctx)
    assert "cancel_order" in {t["name"] for t in llm.requests[0]["tools"]}
    assert "cannot cancel" not in llm.requests[0]["system"]


async def test_max_steps_truncates_gracefully(make_agent, ctx, store):
    store.get("/orders").respond(200, json=[])
    agent, llm = make_agent([calls(("list_my_orders", {}))] * 3, max_steps=3)
    result = await agent.run("loop forever", [], ctx)

    assert result.final_text == MAX_STEPS_MESSAGE
    assert result.trace.truncated
    assert (result.trace.outcome, result.trace.reason) == ("truncated", "max_steps")
    assert len(llm.requests) == 3
    assert len(result.trace.tool_calls()) == 3
    # Unfinished tool exchanges are not carried into the next turn.
    assert [type(m) for m in result.history] == [UserMessage, AssistantMessage]


async def test_overall_timeout_truncates_gracefully(make_agent, ctx):
    async def slow():
        await asyncio.sleep(5)
        return text("too late")

    agent, _ = make_agent([slow], timeout_s=0.05)
    result = await agent.run("hi", [], ctx)

    assert result.final_text == TIMEOUT_MESSAGE
    assert (result.trace.outcome, result.trace.reason) == ("truncated", "timeout")
    assert result.trace.total_latency_ms < 1000


async def test_llm_failure_returns_error_outcome(make_agent, ctx):
    agent, _ = make_agent([LLMUnavailable("gemini down")])
    result = await agent.run("hi", [], ctx)
    assert result.final_text == ERROR_MESSAGE
    assert result.trace.outcome == "error"
    assert "gemini down" in result.trace.reason


async def test_ctx_never_comes_from_model_arguments(make_agent, ctx, store):
    orders_route = store.get("/orders").respond(200, json=[order()])
    agent, llm = make_agent(
        [
            calls(("list_my_orders", {"customer_id": "someone-else"})),
            calls(("get_order", {"order_id": str(uuid4()), "store_token": "stolen"})),
            text("done"),
        ]
    )
    result = await agent.run("show orders of customer 42", [], ctx)

    codes = [s.result.error.code for s in result.trace.tool_calls()]
    assert codes == ["invalid_arguments", "invalid_arguments"]
    assert not orders_route.called


async def test_history_is_carried_into_the_next_turn(make_agent, ctx):
    agent, llm = make_agent([text("first"), text("second")])
    first = await agent.run("one", [], ctx)
    second = await agent.run("two", first.history, ctx)
    sent = llm.requests[1]["messages"]
    assert [getattr(m, "text", None) for m in sent] == ["one", "first", "two"]
    assert [getattr(m, "text", None) for m in second.history][-1] == "second"


async def test_trace_contents(make_agent, ctx, store):
    store.get("/orders").respond(200, json=[order(status="pending")])
    agent, _ = make_agent(
        [calls(("list_my_orders", {"limit": 1}), usage=(100, 10)), text("One order.", (150, 20))]
    )
    result = await agent.run("orders?", [], ctx)
    trace = result.trace

    assert [s.type for s in trace.steps] == ["model_call", "tool_call", "model_call"]
    first, tool_step, last = trace.steps
    assert (first.response, first.tool_calls) == ("tool_calls", ["list_my_orders"])
    assert (first.usage.input_tokens, first.usage.output_tokens) == (100, 10)
    assert (tool_step.name, tool_step.arguments, tool_step.call_id) == (
        "list_my_orders",
        {"limit": 1},
        "c0",
    )
    assert tool_step.result.ok and tool_step.result.data["total_orders"] == 1
    assert last.response == "text"
    assert all(s.latency_ms >= 0 for s in trace.steps)
    assert (trace.usage.input_tokens, trace.usage.output_tokens) == (250, 30)
    assert result.usage == trace.usage
    assert trace.outcome == "completed" and trace.reason is None
    assert trace.total_latency_ms >= 0
    # Serialisable and round-trips.
    assert Trace.model_validate_json(trace.model_dump_json()) == trace
    assert TOKEN not in trace.model_dump_json()


async def test_system_prompt_says_writes_are_disabled(make_agent, ctx):
    agent, llm = make_agent([text("ok")])
    await agent.run("hi", [], ctx)
    assert "cannot cancel orders" in llm.requests[0]["system"]
    assert isinstance(llm.requests[0]["messages"][0], UserMessage)
    assert not any(isinstance(m, ToolCallMessage) for m in llm.requests[0]["messages"])
