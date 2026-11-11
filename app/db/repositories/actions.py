"""Pending write actions. Every query is scoped by customer_id (and conversation).

Status transitions are single conditional UPDATEs (`... WHERE status = 'pending' AND
expires_at > now() RETURNING ...`): whoever gets the row back performed the transition,
so an action can be resolved exactly once even under concurrent requests.
"""

from datetime import timedelta
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PendingAction

Resolution = Literal["confirmed", "cancelled"]


class PendingActionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        customer_id: UUID,
        conversation_id: UUID,
        tool_name: str,
        arguments: dict[str, Any],
        summary: str,
        ttl_s: float,
    ) -> PendingAction:
        # Close any earlier pending action first (the DB allows one per conversation).
        await self.close_pending(customer_id, conversation_id)
        action = PendingAction(
            customer_id=customer_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            arguments=arguments,
            summary=summary,
            status="pending",
            expires_at=func.now() + timedelta(seconds=ttl_s),
        )
        self.session.add(action)
        await self.session.flush()
        await self.session.refresh(action)
        return action

    async def get(
        self, customer_id: UUID, conversation_id: UUID, action_id: UUID
    ) -> PendingAction | None:
        return await self.session.scalar(
            select(PendingAction).where(
                PendingAction.id == action_id,
                PendingAction.conversation_id == conversation_id,
                PendingAction.customer_id == customer_id,
            )
        )

    async def current(self, customer_id: UUID, conversation_id: UUID) -> PendingAction | None:
        """The conversation's live pending action, if any (expired ones don't count)."""
        return await self.session.scalar(
            select(PendingAction).where(
                PendingAction.conversation_id == conversation_id,
                PendingAction.customer_id == customer_id,
                PendingAction.status == "pending",
                PendingAction.expires_at > func.now(),
            )
        )

    async def resolve(
        self, customer_id: UUID, conversation_id: UUID, action_id: UUID, status: Resolution
    ) -> PendingAction | None:
        """Atomically move a live pending action to `status`. None if that isn't possible
        (unknown, someone else's, already resolved, or expired)."""
        return await self.session.scalar(
            update(PendingAction)
            .where(
                PendingAction.id == action_id,
                PendingAction.conversation_id == conversation_id,
                PendingAction.customer_id == customer_id,
                PendingAction.status == "pending",
                PendingAction.expires_at > func.now(),
            )
            .values(status=status, resolved_at=func.now())
            .returning(PendingAction)
        )

    async def record_result(
        self, action_id: UUID, *, ok: bool, result: dict[str, Any]
    ) -> PendingAction:
        """Store the execution result of a confirmed action; a failed execution is `failed`."""
        return await self.session.scalar(
            update(PendingAction)
            .where(PendingAction.id == action_id, PendingAction.status == "confirmed")
            .values(result=result, status="confirmed" if ok else "failed")
            .returning(PendingAction)
        )

    async def close_pending(self, customer_id: UUID, conversation_id: UUID) -> int:
        """Close the conversation's pending action because something newer happened:
        `superseded` if it was still live, `expired` if its time had already run out."""
        result = await self.session.execute(
            update(PendingAction)
            .where(
                PendingAction.conversation_id == conversation_id,
                PendingAction.customer_id == customer_id,
                PendingAction.status == "pending",
            )
            .values(
                status=case((PendingAction.expires_at > func.now(), "superseded"), else_="expired"),
                resolved_at=func.now(),
            )
        )
        return result.rowcount

    async def expire_if_due(self, action_id: UUID) -> None:
        await self.session.execute(
            update(PendingAction)
            .where(
                PendingAction.id == action_id,
                PendingAction.status == "pending",
                PendingAction.expires_at <= func.now(),
            )
            .values(status="expired", resolved_at=func.now())
        )
