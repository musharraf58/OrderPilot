import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text as sql
from sqlalchemy import update

from app.db.models import PendingAction
from app.db.repositories import ConversationRepository, PendingActionRepository
from tests.api.helpers import parse_sse, text, tool_calls
from tests.conftest import AUTH
from tests.factories import CUSTOMER_ID, order

OTHER = uuid4()


@pytest.fixture
def mine(store):
    pending = order(status="pending", total_amount="144.90", created_at="2026-11-10T12:35:13")
    store.get("/orders").respond(200, json=[pending])
    return pending


@pytest.fixture
def cancel_route(store, mine):
    return store.post(f"/orders/{mine['id']}/cancel").respond(
        200, json={**mine, "status": "cancelled"}
    )


async def propose(chat, mine, extra_script=()):
    """Run a chat turn in which the model asks to cancel `mine`; return (cid, events)."""
    chat.script([tool_calls(("cancel_order", {"order_id": mine["id"]})), *extra_script])
    cid = await chat.new()
    events = parse_sse((await chat.send(cid, "Please cancel my latest order")).text)
    return cid, events


def action_url(cid, action_id, verb):
    return f"/conversations/{cid}/actions/{action_id}/{verb}"


async def get_action(chat, cid, action_id):
    async with chat.sessions() as s:
        return await PendingActionRepository(s).get(CUSTOMER_ID, cid, action_id)


async def test_proposal_creates_pending_action_without_touching_the_store(chat, mine, cancel_route):
    cid, events = await propose(chat, mine)

    assert [e for e, _ in events] == ["confirmation_required", "done"]
    data = events[0][1]
    assert data["summary"] == (
        "Cancel your order from 10 Nov 2026 (2 items, total $144.90)? It is currently pending."
    )
    assert not cancel_route.called

    action = await get_action(chat, cid, data["action_id"])
    assert (action.status, action.tool_name, action.arguments) == (
        "pending",
        "cancel_order",
        {"order_id": mine["id"]},
    )
    rows = await chat.rows(cid)
    assert [r.role for r in rows] == ["user", "tool_call", "tool_result", "assistant"]
    assert rows[2].content["result"]["error"]["code"] == "awaiting_confirmation"
    assert rows[3].content["text"].startswith(data["summary"])
    assert events[1][1]["message_id"] == str(rows[3].id)

    detail = (await chat.client.get(f"/conversations/{cid}", headers=AUTH)).json()
    assert detail["pending_action"]["id"] == data["action_id"]
    assert detail["pending_action"]["summary"] == data["summary"]
    assert [i["type"] for i in detail["items"]] == ["message", "message"]  # no "failed" tool


async def test_confirm_executes_once_and_records_result(chat, mine, cancel_route):
    cid, events = await propose(chat, mine)
    action_id = events[0][1]["action_id"]

    r = await chat.client.post(action_url(cid, action_id, "confirm"), headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert (body["ok"], body["message"]) == (True, "Done: the order is now cancelled.")
    assert body["action"]["status"] == "confirmed"
    assert cancel_route.call_count == 1

    action = await get_action(chat, cid, action_id)
    assert action.result["ok"] is True and action.result["data"]["status"] == "cancelled"
    rows = await chat.rows(cid)
    assert [r.role for r in rows[4:]] == ["tool_call", "tool_result", "assistant"]
    injected = rows[4].content["calls"][0]
    assert injected["injected"] is True
    assert injected["arguments"] == {"order_id": mine["id"]}
    assert rows[5].content["result"]["ok"] is True
    assert rows[6].content["text"] == "Done: the order is now cancelled."
    assert body["message_id"] == str(rows[6].id)

    detail = (await chat.client.get(f"/conversations/{cid}", headers=AUTH)).json()
    assert detail["pending_action"] is None


async def test_next_turn_sees_the_real_result_as_a_tool_result(chat, mine, cancel_route):
    cid, events = await propose(chat, mine)
    await chat.client.post(action_url(cid, events[0][1]["action_id"], "confirm"), headers=AUTH)
    llm = chat.script([text("Yes, it's cancelled.")])
    await chat.send(cid, "Is it cancelled now?")
    sent = llm.requests[0]["messages"]
    results = [m for m in sent if m.role == "tool_result"]
    assert [r.result["ok"] for r in results] == [False, True]  # awaiting, then the real one
    assert results[-1].result["data"]["status"] == "cancelled"


async def test_double_confirm_is_409_and_store_called_once(chat, mine, cancel_route):
    cid, events = await propose(chat, mine)
    url = action_url(cid, events[0][1]["action_id"], "confirm")
    assert (await chat.client.post(url, headers=AUTH)).status_code == 200
    again = await chat.client.post(url, headers=AUTH)
    assert again.status_code == 409
    assert again.json()["detail"] == "This action was already confirmed."
    assert cancel_route.call_count == 1


async def test_concurrent_confirms_execute_once(chat, mine, cancel_route):
    cid, events = await propose(chat, mine)
    url = action_url(cid, events[0][1]["action_id"], "confirm")
    responses = await asyncio.gather(*(chat.client.post(url, headers=AUTH) for _ in range(5)))
    assert sorted(r.status_code for r in responses) == [200, 409, 409, 409, 409]
    assert cancel_route.call_count == 1


async def test_confirm_after_expiry_is_409_and_never_executes(chat, mine, cancel_route):
    cid, events = await propose(chat, mine)
    action_id = events[0][1]["action_id"]
    async with chat.sessions() as s, s.begin():
        await s.execute(
            update(PendingAction)
            .where(PendingAction.id == action_id)
            .values(expires_at=sql("now() - interval '1 second'"))
        )
    r = await chat.client.post(action_url(cid, action_id, "confirm"), headers=AUTH)
    assert r.status_code == 409
    assert "expired" in r.json()["detail"]
    assert not cancel_route.called
    assert (await get_action(chat, cid, action_id)).status == "expired"


async def test_cancelling_the_action_never_calls_the_store(chat, mine, cancel_route):
    cid, events = await propose(chat, mine)
    action_id = events[0][1]["action_id"]
    r = await chat.client.post(action_url(cid, action_id, "cancel"), headers=AUTH)
    assert r.status_code == 200
    assert r.json()["message"] == "Okay, I've left it as it is. Nothing was changed."
    assert r.json()["action"]["status"] == "cancelled"
    assert not cancel_route.called
    # And it can't be confirmed afterwards.
    late = await chat.client.post(action_url(cid, action_id, "confirm"), headers=AUTH)
    assert late.status_code == 409
    assert not cancel_route.called
    assert (await chat.rows(cid))[-1].content["text"] == r.json()["message"]


async def test_other_customers_action_is_404(chat, mine, cancel_route):
    async with chat.sessions() as s, s.begin():
        theirs = await ConversationRepository(s).create(OTHER)
        their_action = await PendingActionRepository(s).create(
            OTHER, theirs.id, "cancel_order", {"order_id": mine["id"]}, "theirs", ttl_s=600
        )
    my_cid = await chat.new()
    for cid in (theirs.id, my_cid):  # their conversation, or my conversation + their action
        for verb in ("confirm", "cancel"):
            r = await chat.client.post(action_url(cid, their_action.id, verb), headers=AUTH)
            assert r.status_code == 404, (cid, verb)
    assert (
        await chat.client.post(action_url(my_cid, uuid4(), "confirm"), headers=AUTH)
    ).status_code == 404
    assert not cancel_route.called
    async with chat.sessions() as s:
        assert (
            await PendingActionRepository(s).get(OTHER, theirs.id, their_action.id)
        ).status == "pending"


async def test_new_message_supersedes_the_pending_action(chat, mine, cancel_route):
    cid, events = await propose(chat, mine, extra_script=[text("Sure, what else?")])
    action_id = events[0][1]["action_id"]
    await chat.send(cid, "Actually, never mind. Do you sell mugs?")
    assert (await get_action(chat, cid, action_id)).status == "superseded"
    r = await chat.client.post(action_url(cid, action_id, "confirm"), headers=AUTH)
    assert r.status_code == 409
    assert r.json()["detail"] == "This action was already superseded."
    assert not cancel_route.called


async def test_arguments_cannot_be_changed_at_confirmation(chat, store, mine, cancel_route):
    other_order = str(uuid4())
    other_cancel = store.post(f"/orders/{other_order}/cancel").respond(200, json=order())
    cid, events = await propose(chat, mine)
    r = await chat.client.post(
        action_url(cid, events[0][1]["action_id"], "confirm"),
        params={"order_id": other_order},
        json={"order_id": other_order, "tool_name": "something_else"},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert cancel_route.call_count == 1  # the order fixed at proposal time
    assert not other_cancel.called


async def test_confirm_reruns_checks_and_store_has_final_say(chat, store, mine):
    store.post(f"/orders/{mine['id']}/cancel").respond(409, json={"detail": "not pending"})
    cid, events = await propose(chat, mine)
    r = await chat.client.post(action_url(cid, events[0][1]["action_id"], "confirm"), headers=AUTH)
    body = r.json()
    assert (r.status_code, body["ok"], body["action"]["status"]) == (200, False, "failed")
    assert body["message"].startswith("The order was not cancelled:")


async def test_order_changed_since_proposal_is_not_cancelled(chat, store, mine, cancel_route):
    cid, events = await propose(chat, mine)
    store.get("/orders").respond(200, json=[{**mine, "status": "shipped"}])  # it shipped meanwhile
    r = await chat.client.post(action_url(cid, events[0][1]["action_id"], "confirm"), headers=AUTH)
    assert r.json()["ok"] is False
    assert "already shipped" in r.json()["message"]
    assert not cancel_route.called


async def test_foreign_order_is_refused_at_proposal_with_no_action(chat, store, mine, cancel_route):
    chat.script([tool_calls(("cancel_order", {"order_id": str(uuid4())})), text("Not found.")])
    cid = await chat.new()
    events = parse_sse((await chat.send(cid, "cancel order 123")).text)
    assert "confirmation_required" not in [e for e, _ in events]
    async with chat.sessions() as s:
        assert await PendingActionRepository(s).current(CUSTOMER_ID, cid) is None


async def test_confirm_while_a_turn_is_running_is_409(chat, api_app, mine, cancel_route):
    cid, events = await propose(chat, mine)
    assert api_app.state.run_locks.try_acquire(UUID(cid))  # a turn is in progress
    r = await chat.client.post(action_url(cid, events[0][1]["action_id"], "confirm"), headers=AUTH)
    assert r.status_code == 409
    assert not cancel_route.called
    api_app.state.run_locks.release(UUID(cid))
    r = await chat.client.post(action_url(cid, events[0][1]["action_id"], "confirm"), headers=AUTH)
    assert r.status_code == 200


async def test_confirm_and_cancel_are_rate_limited(chat, api_app, clock, mine, cancel_route):
    from app.core.limits import RateLimiter

    api_app.state.action_rate_limiter = RateLimiter(limit=1, window_s=600, clock=clock)
    cid, events = await propose(chat, mine)
    url = action_url(cid, events[0][1]["action_id"], "cancel")
    assert (await chat.client.post(url, headers=AUTH)).status_code == 200
    limited = await chat.client.post(url, headers=AUTH)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


async def test_writes_disabled_blocks_proposals_and_confirms(chat, mine, cancel_route, monkeypatch):
    from app.api import actions, conversations
    from app.core.config import Settings

    off = Settings(AGENT_ALLOW_WRITES=False)
    monkeypatch.setattr(conversations, "get_settings", lambda: off)
    monkeypatch.setattr(actions, "get_settings", lambda: off)
    cid, events = await propose(chat, mine, extra_script=[text("I can't cancel orders yet.")])
    assert "confirmation_required" not in [e for e, _ in events]
    assert "cancel_order" not in {t["name"] for t in chat.llm.requests[0]["tools"]}
    assert "cannot cancel orders" in chat.llm.requests[0]["system"]
    assert (
        await chat.client.post(action_url(cid, uuid4(), "confirm"), headers=AUTH)
    ).status_code == 403
    assert not cancel_route.called


async def test_message_length_limit(chat):
    chat.script([])
    cid = await chat.new()
    for message in ("", "   \n ", "x" * 1001):
        assert (await chat.send(cid, message)).status_code == 422, repr(message)
    chat.script([text("ok")])
    assert (await chat.send(cid, "x" * 1000)).status_code == 200
