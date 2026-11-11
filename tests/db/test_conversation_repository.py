from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from app.agent.llm import (
    AssistantMessage,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from app.db.models import Message as MessageRow
from app.db.repositories import ConversationRepository
from app.db.repositories.conversations import make_title, to_message

ALICE = uuid4()
BOB = uuid4()


def a_turn(question="Where is my order?"):
    return [
        UserMessage(text=question),
        ToolCallMessage(
            calls=[
                ToolCall(
                    id="g1",
                    name="list_my_orders",
                    arguments={"limit": 1},
                    provider_data={"gemini_id": "g1", "gemini_thought_signature": "c2ln"},
                )
            ]
        ),
        ToolResultMessage(call_id="g1", name="list_my_orders", result={"ok": True, "data": {}}),
        AssistantMessage(text="It has shipped."),
    ]


async def test_create_list_and_get_are_scoped_by_customer(sessionmaker):
    async with sessionmaker() as s, s.begin():
        repo = ConversationRepository(s)
        first = await repo.create(ALICE)
        second = await repo.create(ALICE)
        bobs = await repo.create(BOB)

    async with sessionmaker() as s:
        repo = ConversationRepository(s)
        assert {c.id for c in await repo.list_for_customer(ALICE)} == {first.id, second.id}
        assert (await repo.get(ALICE, first.id)).id == first.id
        assert await repo.get(ALICE, bobs.id) is None
        assert await repo.get(ALICE, uuid4()) is None
        assert await repo.load_messages(ALICE, bobs.id) == []


async def test_list_is_newest_activity_first(sessionmaker):
    async with sessionmaker() as s, s.begin():
        repo = ConversationRepository(s)
        older = await repo.create(ALICE)
        newer = await repo.create(ALICE)
    async with sessionmaker() as s, s.begin():
        await ConversationRepository(s).append_turn(ALICE, older.id, a_turn())
    async with sessionmaker() as s:
        ids = [c.id for c in await ConversationRepository(s).list_for_customer(ALICE)]
    assert ids == [older.id, newer.id]


async def test_append_turn_round_trips_every_message_including_provider_data(sessionmaker):
    async with sessionmaker() as s, s.begin():
        conv = await ConversationRepository(s).create(ALICE)
    turn = a_turn()
    async with sessionmaker() as s, s.begin():
        await ConversationRepository(s).append_turn(ALICE, conv.id, turn)
        await ConversationRepository(s).append_turn(ALICE, conv.id, a_turn("And the other?"))

    async with sessionmaker() as s:
        rows = await ConversationRepository(s).load_messages(ALICE, conv.id)
        conv = await ConversationRepository(s).get(ALICE, conv.id)
    assert [r.seq for r in rows] == list(range(1, 9))
    assert [r.role for r in rows[:4]] == ["user", "tool_call", "tool_result", "assistant"]
    assert [to_message(r) for r in rows[:4]] == turn
    assert to_message(rows[1]).calls[0].provider_data["gemini_thought_signature"] == "c2ln"
    assert conv.title == "Where is my order?"  # from the first turn, not overwritten


async def test_append_turn_refuses_another_customers_conversation(sessionmaker):
    async with sessionmaker() as s, s.begin():
        conv = await ConversationRepository(s).create(BOB)
    async with sessionmaker() as s:
        with pytest.raises(LookupError):
            await ConversationRepository(s).append_turn(ALICE, conv.id, a_turn())


async def test_seq_is_unique_per_conversation(sessionmaker):
    async with sessionmaker() as s, s.begin():
        conv = await ConversationRepository(s).create(ALICE)
    async with sessionmaker() as s:
        s.add_all(
            [
                MessageRow(conversation_id=conv.id, seq=1, role="user", content={}),
                MessageRow(conversation_id=conv.id, seq=1, role="user", content={}),
            ]
        )
        with pytest.raises(IntegrityError):
            await s.commit()


async def test_deleting_a_conversation_cascades_to_messages(sessionmaker):
    from sqlalchemy import delete, func, select

    from app.db.models import Conversation

    async with sessionmaker() as s, s.begin():
        conv = await ConversationRepository(s).create(ALICE)
        await ConversationRepository(s).append_turn(ALICE, conv.id, a_turn())
    async with sessionmaker() as s, s.begin():
        await s.execute(delete(Conversation).where(Conversation.id == conv.id))
    async with sessionmaker() as s:
        assert await s.scalar(select(func.count()).select_from(MessageRow)) == 0


def test_make_title():
    assert make_title("  hello\n  world ") == "hello world"
    long = make_title("x" * 100)
    assert len(long) == 60 and long.endswith("…")
