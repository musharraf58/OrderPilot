"""All conversation/message queries. Every method is scoped by customer_id.

Repositories never commit: the caller owns the transaction, so a whole turn can be
written atomically.
"""

import re
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm import MESSAGE_ADAPTER, Message, UserMessage
from app.db.models import Conversation
from app.db.models import Message as MessageRow

TITLE_LENGTH = 60


def make_title(text: str) -> str:
    title = re.sub(r"\s+", " ", text).strip()
    return title if len(title) <= TITLE_LENGTH else title[: TITLE_LENGTH - 1].rstrip() + "…"


def to_message(row: MessageRow) -> Message:
    return MESSAGE_ADAPTER.validate_python(row.content)


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, customer_id: UUID, title: str | None = None) -> Conversation:
        conversation = Conversation(customer_id=customer_id, title=title)
        self.session.add(conversation)
        await self.session.flush()
        await self.session.refresh(conversation)
        return conversation

    async def list_for_customer(self, customer_id: UUID, limit: int = 50) -> list[Conversation]:
        result = await self.session.scalars(
            select(Conversation)
            .where(Conversation.customer_id == customer_id)
            .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
            .limit(limit)
        )
        return list(result)

    async def get(self, customer_id: UUID, conversation_id: UUID) -> Conversation | None:
        """None for both "does not exist" and "belongs to someone else"."""
        return await self.session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.customer_id == customer_id
            )
        )

    async def load_messages(self, customer_id: UUID, conversation_id: UUID) -> list[MessageRow]:
        result = await self.session.scalars(
            select(MessageRow)
            .join(Conversation)
            .where(Conversation.id == conversation_id, Conversation.customer_id == customer_id)
            .order_by(MessageRow.seq)
        )
        return list(result)

    async def append_turn(
        self, customer_id: UUID, conversation_id: UUID, messages: Sequence[Message]
    ) -> list[MessageRow]:
        """Append one turn's messages with consecutive seq numbers.

        Locks the conversation row so concurrent writers (even in other processes) cannot
        hand out the same seq. Sets the title from the first user message if missing.
        """
        conversation = await self.session.scalar(
            select(Conversation)
            .where(Conversation.id == conversation_id, Conversation.customer_id == customer_id)
            .with_for_update()
        )
        if conversation is None:
            raise LookupError("conversation not found")

        last_seq = await self.session.scalar(
            select(func.coalesce(func.max(MessageRow.seq), 0)).where(
                MessageRow.conversation_id == conversation_id
            )
        )
        rows = [
            MessageRow(
                conversation_id=conversation_id,
                seq=last_seq + i,
                role=message.role,
                content=message.model_dump(mode="json"),
            )
            for i, message in enumerate(messages, start=1)
        ]
        self.session.add_all(rows)

        if conversation.title is None:
            first_user = next((m for m in messages if isinstance(m, UserMessage)), None)
            if first_user is not None:
                conversation.title = make_title(first_user.text)
        conversation.updated_at = func.now()
        await self.session.flush()
        return rows
