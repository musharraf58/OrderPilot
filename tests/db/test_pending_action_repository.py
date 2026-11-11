import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import text, update
from sqlalchemy.exc import IntegrityError

from app.db.models import PendingAction
from app.db.repositories import ConversationRepository, PendingActionRepository

ALICE = uuid4()
BOB = uuid4()
ARGS = {"order_id": str(uuid4())}


@pytest.fixture
async def conversation(sessionmaker):
    async with sessionmaker() as s, s.begin():
        return (await ConversationRepository(s).create(ALICE)).id


async def propose(sessionmaker, conversation_id, customer=ALICE, ttl=600):
    async with sessionmaker() as s, s.begin():
        return await PendingActionRepository(s).create(
            customer, conversation_id, "cancel_order", ARGS, "Cancel order from 1 Nov?", ttl
        )


async def make_expired(sessionmaker, action_id):
    async with sessionmaker() as s, s.begin():
        await s.execute(
            update(PendingAction)
            .where(PendingAction.id == action_id)
            .values(expires_at=text("now() - interval '1 second'"))
        )


async def test_create_and_read_back(sessionmaker, conversation):
    action = await propose(sessionmaker, conversation)
    assert (action.status, action.arguments, action.result) == ("pending", ARGS, None)
    assert action.expires_at > action.created_at
    async with sessionmaker() as s:
        repo = PendingActionRepository(s)
        assert (await repo.current(ALICE, conversation)).id == action.id
        assert await repo.get(BOB, conversation, action.id) is None
        assert await repo.current(BOB, conversation) is None


async def test_resolve_happens_exactly_once(sessionmaker, conversation):
    action = await propose(sessionmaker, conversation)
    async with sessionmaker() as s, s.begin():
        first = await PendingActionRepository(s).resolve(
            ALICE, conversation, action.id, "confirmed"
        )
    async with sessionmaker() as s, s.begin():
        second = await PendingActionRepository(s).resolve(
            ALICE, conversation, action.id, "confirmed"
        )
    assert first.status == "confirmed" and first.resolved_at is not None
    assert second is None


async def test_concurrent_resolves_only_one_wins(sessionmaker, conversation):
    action = await propose(sessionmaker, conversation)

    async def attempt(status):
        async with sessionmaker() as s, s.begin():
            return await PendingActionRepository(s).resolve(ALICE, conversation, action.id, status)

    results = await asyncio.gather(*(attempt(st) for st in ["confirmed", "cancelled"] * 5))
    assert sum(r is not None for r in results) == 1


async def test_cannot_resolve_expired_or_someone_elses(sessionmaker, conversation):
    action = await propose(sessionmaker, conversation)
    async with sessionmaker() as s, s.begin():
        assert (
            await PendingActionRepository(s).resolve(BOB, conversation, action.id, "confirmed")
            is None
        )
    await make_expired(sessionmaker, action.id)
    async with sessionmaker() as s, s.begin():
        repo = PendingActionRepository(s)
        assert await repo.resolve(ALICE, conversation, action.id, "confirmed") is None
        assert await repo.current(ALICE, conversation) is None
        await repo.expire_if_due(action.id)
    async with sessionmaker() as s:
        assert (
            await PendingActionRepository(s).get(ALICE, conversation, action.id)
        ).status == "expired"


async def test_record_result_marks_failures(sessionmaker, conversation):
    action = await propose(sessionmaker, conversation)
    async with sessionmaker() as s, s.begin():
        repo = PendingActionRepository(s)
        await repo.resolve(ALICE, conversation, action.id, "confirmed")
        stored = await repo.record_result(action.id, ok=False, result={"ok": False})
    assert (stored.status, stored.result) == ("failed", {"ok": False})


async def test_new_proposal_supersedes_the_old_one(sessionmaker, conversation):
    old = await propose(sessionmaker, conversation)
    new = await propose(sessionmaker, conversation)
    async with sessionmaker() as s:
        repo = PendingActionRepository(s)
        assert (await repo.get(ALICE, conversation, old.id)).status == "superseded"
        assert (await repo.current(ALICE, conversation)).id == new.id


async def test_close_pending_marks_stale_ones_expired(sessionmaker, conversation):
    action = await propose(sessionmaker, conversation)
    await make_expired(sessionmaker, action.id)
    async with sessionmaker() as s, s.begin():
        assert await PendingActionRepository(s).close_pending(ALICE, conversation) == 1
    async with sessionmaker() as s:
        assert (
            await PendingActionRepository(s).get(ALICE, conversation, action.id)
        ).status == "expired"


async def test_database_allows_only_one_pending_per_conversation(sessionmaker, conversation):
    await propose(sessionmaker, conversation)
    async with sessionmaker() as s:
        s.add(
            PendingAction(
                customer_id=ALICE,
                conversation_id=conversation,
                tool_name="cancel_order",
                arguments=ARGS,
                summary="x",
                status="pending",
                expires_at=text("now() + interval '1 hour'"),
            )
        )
        with pytest.raises(IntegrityError):
            await s.commit()
