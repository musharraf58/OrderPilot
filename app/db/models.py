import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Indexed through ix_conversations_customer_updated (customer_id is its leading column).
    customer_id: Mapped[uuid.UUID]
    title: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        order_by="Message.seq",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (Index("ix_conversations_customer_updated", "customer_id", "updated_at"),)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    seq: Mapped[int]
    # user | assistant | tool_call | tool_result (mirrors content["role"]).
    role: Mapped[str] = mapped_column(String(20))
    # The full app.agent.llm message, including tool calls, tool results and opaque
    # provider_data (e.g. Gemini thought signatures), so the turn can be replayed exactly.
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uq_messages_conversation_seq"),
    )


class PendingAction(Base):
    """A write action the model proposed and the customer has not resolved yet.

    The model can only create these (through the agent loop); only the customer, through
    the confirm/cancel endpoints, can resolve them. Arguments are fixed at proposal time.
    """

    __tablename__ = "pending_actions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(index=True)
    tool_name: Mapped[str] = mapped_column(String(64))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Built by our code from fresh store data, never by the model.
    summary: Mapped[str] = mapped_column(Text)
    # pending | confirmed | cancelled | expired | superseded | failed
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'confirmed', 'cancelled', 'expired', 'superseded', 'failed')",
            name="ck_pending_actions_status",
        ),
        # The database itself guarantees at most one pending action per conversation.
        Index(
            "uq_pending_actions_one_pending_per_conversation",
            "conversation_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )
