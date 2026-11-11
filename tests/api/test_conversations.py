import json
from uuid import uuid4

from app.agent.llm import (
    LLMUnavailable,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from app.agent.loop import ERROR_MESSAGE, MAX_STEPS_MESSAGE
from app.db.repositories import ConversationRepository
from tests.api.helpers import parse_sse, text, tool_calls
from tests.conftest import AUTH
from tests.factories import order

OTHER_CUSTOMER = uuid4()


async def test_create_list_get(chat):
    first = await chat.new()
    second = await chat.new()
    listed = (await chat.client.get("/conversations", headers=AUTH)).json()
    assert [c["id"] for c in listed] == [second, first]  # newest first
    detail = (await chat.client.get(f"/conversations/{first}", headers=AUTH)).json()
    assert detail["id"] == first
    assert detail["title"] is None
    assert detail["items"] == []


async def test_requires_auth(chat):
    assert (await chat.client.get("/conversations")).status_code == 401
    assert (await chat.client.post("/conversations")).status_code == 401


async def test_other_customers_conversation_is_404_like_a_missing_one(chat):
    async with chat.sessions() as s, s.begin():
        theirs = (await ConversationRepository(s).create(OTHER_CUSTOMER)).id
    chat.script([text("never used")])
    missing = uuid4()
    for cid in (theirs, missing):
        get = await chat.client.get(f"/conversations/{cid}", headers=AUTH)
        post = await chat.send(cid, "hello")
        assert (get.status_code, post.status_code) == (404, 404)
        assert get.json() == post.json() == {"detail": "Conversation not found."}
    assert (await chat.client.get("/conversations", headers=AUTH)).json() == []
    assert chat.llm.requests == []


async def test_full_streaming_turn(chat, store):
    store.get("/orders").respond(200, json=[order(status="shipped")])
    chat.script([tool_calls(("list_my_orders", {"limit": 1})), text("Your order has shipped.")])
    cid = await chat.new()

    r = await chat.send(cid, "Where is my order?")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[:2] == ["tool_started", "tool_finished"]
    assert set(names[2:-1]) == {"text_delta"} and len(names[2:-1]) > 1  # really streamed
    assert names[-1] == "done"
    assert events[0][1] == {"name": "list_my_orders"}
    assert events[1][1] == {"name": "list_my_orders", "ok": True}
    assert "".join(d["text"] for e, d in events if e == "text_delta") == "Your order has shipped."

    rows = await chat.rows(cid)
    assert [r.role for r in rows] == ["user", "tool_call", "tool_result", "assistant"]
    assert events[-1][1] == {"message_id": str(rows[-1].id)}


async def test_turn_is_persisted_with_provider_data_and_displayed_cleanly(chat, store):
    store.get("/orders").respond(200, json=[order()])
    chat.script([tool_calls(("list_my_orders", {})), text("One order.")])
    cid = await chat.new()
    await chat.send(cid, "What orders   do I have?")

    rows = await chat.rows(cid)
    call = rows[1].content["calls"][0]
    assert call["provider_data"] == {"gemini_id": "g0", "gemini_thought_signature": "U0lH"}
    assert rows[2].content["result"]["ok"] is True

    detail = (await chat.client.get(f"/conversations/{cid}", headers=AUTH)).json()
    assert detail["title"] == "What orders do I have?"
    assert [(i["type"], i.get("role"), i.get("text"), i.get("name")) for i in detail["items"]] == [
        ("message", "user", "What orders   do I have?", None),
        ("tool", None, None, "list_my_orders"),
        ("message", "assistant", "One order.", None),
    ]
    assert "provider_data" not in json.dumps(detail)
    assert "U0lH" not in json.dumps(detail)


async def test_second_turn_replays_full_history_including_signatures(chat, store):
    store.get("/orders").respond(200, json=[order()])
    llm = chat.script([tool_calls(("list_my_orders", {})), text("One order."), text("Sure.")])
    cid = await chat.new()
    await chat.send(cid, "What orders do I have?")
    await chat.send(cid, "Thanks, and the status?")

    sent = llm.requests[2]["messages"]
    assert [m.role for m in sent] == [
        "user",
        "tool_call",
        "tool_result",
        "assistant",
        "user",
    ]
    replayed_call = next(m for m in sent if isinstance(m, ToolCallMessage)).calls[0]
    assert replayed_call.provider_data["gemini_thought_signature"] == "U0lH"
    result = next(m for m in sent if isinstance(m, ToolResultMessage))
    assert result.call_id == "g0"
    assert [r.seq for r in await chat.rows(cid)] == list(range(1, 7))


async def test_history_window_is_applied_without_splitting_tool_pairs(chat, store, monkeypatch):
    from app.api import conversations
    from app.core.config import Settings

    monkeypatch.setattr(conversations, "get_settings", lambda: Settings(HISTORY_WINDOW=3))
    store.get("/orders").respond(200, json=[order()])
    llm = chat.script(
        [tool_calls(("list_my_orders", {})), text("One order."), text("A"), text("B")]
    )
    cid = await chat.new()
    await chat.send(cid, "first")  # 4 messages: user, tool_call, tool_result, assistant
    await chat.send(cid, "second")
    await chat.send(cid, "third")

    # Stored: 8 messages. The last 3 would start at an assistant message, so the window
    # moves forward to the next user message.
    sent = llm.requests[3]["messages"]
    assert [m.role for m in sent] == ["user", "assistant", "user"]
    assert sent[0].text == "second"
    assert len(await chat.rows(cid)) == 8


async def test_failed_turn_persists_only_user_message_and_fallback(chat):
    chat.script([LLMUnavailable("down")])
    cid = await chat.new()
    events = parse_sse((await chat.send(cid, "hello?")).text)
    assert [e for e, _ in events] == ["error", "done"]
    assert events[0][1] == {"code": "llm_error", "message": ERROR_MESSAGE}

    rows = await chat.rows(cid)
    assert [(r.role, r.content["text"]) for r in rows] == [
        ("user", "hello?"),
        ("assistant", ERROR_MESSAGE),
    ]


async def test_truncated_turn_never_stores_a_dangling_tool_call(chat, store, monkeypatch):
    from app.api import conversations
    from app.core.config import Settings

    monkeypatch.setattr(conversations, "get_settings", lambda: Settings(AGENT_MAX_STEPS=2))
    store.get("/orders").respond(200, json=[])
    chat.script([tool_calls(("list_my_orders", {}))] * 2)
    cid = await chat.new()
    events = parse_sse((await chat.send(cid, "loop")).text)

    assert [e for e, _ in events][-2:] == ["error", "done"]
    assert events[-2][1]["code"] == "max_steps"
    rows = await chat.rows(cid)
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[1].content["text"] == MAX_STEPS_MESSAGE

    # The next turn starts from a valid history.
    llm = chat.script([text("Hi again")])
    await chat.send(cid, "again")
    assert [type(m) for m in llm.requests[0]["messages"]][-1] is UserMessage


async def test_assistant_not_configured_is_503(chat):
    cid = await chat.new()
    r = await chat.send(cid, "hi")
    assert r.status_code == 503


async def test_message_validation(chat):
    chat.script([])
    cid = await chat.new()
    assert (await chat.send(cid, "")).status_code == 422
    assert (await chat.send(cid, "x" * 4001)).status_code == 422


async def test_persistence_failure_reports_error_event(chat, monkeypatch):
    chat.script([text("hello")])
    cid = await chat.new()

    async def broken(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(ConversationRepository, "append_turn", broken)
    events = parse_sse((await chat.send(cid, "hi")).text)
    assert [e for e, _ in events][-1] == "error"
    assert events[-1][1]["code"] == "internal_error"
    assert "done" not in [e for e, _ in events]
    assert await chat.rows(cid) == []
