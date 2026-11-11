"""Confirming or cancelling a proposed write action.

This is the only way a write ever runs: the model can only propose; the customer decides
here, with an authenticated request that is not a chat message. The action's arguments
were fixed when it was proposed, and these endpoints take no body, so nothing about what
gets executed can change at confirmation time.
"""

import logging
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.agent.llm import AssistantMessage, ToolCall, ToolCallMessage, ToolResultMessage
from app.api.conversations import NOT_FOUND, SessionMaker
from app.api.deps import CurrentUser, get_registry
from app.core.config import get_settings
from app.core.limits import RateLimiter, RunLocks
from app.db.models import PendingAction
from app.db.repositories import ConversationRepository, PendingActionRepository
from app.tools.context import UserContext
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations/{conversation_id}/actions", tags=["actions"])

DECLINED_MESSAGE = "Okay, I've left it as it is. Nothing was changed."


class ActionOut(BaseModel):
    id: UUID
    tool_name: str
    summary: str
    status: str
    expires_at: datetime
    resolved_at: datetime | None


class ActionResponse(BaseModel):
    action: ActionOut
    ok: bool
    # Code-written text for the customer; also stored as the assistant's message.
    message: str
    message_id: UUID


async def _claim(
    request: Request,
    sessions: SessionMaker,
    user: UserContext,
    conversation_id: UUID,
    action_id: UUID,
    resolution: Literal["confirmed", "cancelled"],
) -> PendingAction:
    """Checks, then atomically move the action out of `pending`. On success the caller
    holds the conversation's run lock and must release it."""
    async with sessions() as s:
        if await ConversationRepository(s).get(user.customer_id, conversation_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)
        if (
            await PendingActionRepository(s).get(user.customer_id, conversation_id, action_id)
            is None
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Action not found.")

    limiter: RateLimiter = request.app.state.action_rate_limiter
    retry_after = limiter.hit(user.customer_id)
    if retry_after is not None:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many requests. Please wait a moment.",
            headers={"Retry-After": str(retry_after)},
        )
    locks: RunLocks = request.app.state.run_locks
    if not locks.try_acquire(conversation_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "A reply is being generated in this conversation."
        )

    try:
        async with sessions() as s, s.begin():
            repo = PendingActionRepository(s)
            claimed = await repo.resolve(user.customer_id, conversation_id, action_id, resolution)
            if claimed is None:
                await repo.expire_if_due(action_id)
                current = await repo.get(user.customer_id, conversation_id, action_id)
        if claimed is None:
            detail = (
                "This action has expired. Ask again if you still want to do it."
                if current.status == "expired"
                else f"This action was already {current.status}."
            )
            raise HTTPException(status.HTTP_409_CONFLICT, detail)
    except BaseException:
        locks.release(conversation_id)
        raise
    return claimed


@router.post("/{action_id}/confirm")
async def confirm_action(
    conversation_id: UUID,
    action_id: UUID,
    request: Request,
    user: CurrentUser,
    sessions: SessionMaker,
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> ActionResponse:
    if not get_settings().AGENT_ALLOW_WRITES:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Actions are disabled.")
    action = await _claim(request, sessions, user, conversation_id, action_id, "confirmed")
    try:
        # The full tool, with the stored arguments and our ctx: ownership and feasibility
        # are checked again on fresh data, and the store still has the final say.
        result = await registry.execute(action.tool_name, action.arguments, user)
        tool = registry.get(action.tool_name)
        message = tool.outcome_message(result) if tool else "The action could not be run."

        # Record the execution as a tool call our code made plus its real result, so the
        # model sees the outcome as a tool result on later turns (it does not trust plain
        # assistant text for that), then the code-written outcome message.
        call = ToolCall(
            id=f"action_{action.id.hex}",
            name=action.tool_name,
            arguments=action.arguments,
            injected=True,
        )
        async with sessions() as s, s.begin():
            stored = await PendingActionRepository(s).record_result(
                action.id, ok=result.ok, result=result.model_dump(mode="json")
            )
            rows = await ConversationRepository(s).append_turn(
                user.customer_id,
                conversation_id,
                [
                    ToolCallMessage(calls=[call]),
                    ToolResultMessage(
                        call_id=call.id, name=call.name, result=result.model_dump(mode="json")
                    ),
                    AssistantMessage(text=message),
                ],
            )
        logger.info(
            "action %s confirmed: tool=%s ok=%s code=%s",
            action.id,
            action.tool_name,
            result.ok,
            result.error.code if result.error else None,
        )
        return ActionResponse(
            action=ActionOut.model_validate(stored, from_attributes=True),
            ok=result.ok,
            message=message,
            message_id=rows[-1].id,
        )
    finally:
        request.app.state.run_locks.release(conversation_id)


@router.post("/{action_id}/cancel")
async def cancel_action(
    conversation_id: UUID,
    action_id: UUID,
    request: Request,
    user: CurrentUser,
    sessions: SessionMaker,
) -> ActionResponse:
    action = await _claim(request, sessions, user, conversation_id, action_id, "cancelled")
    try:
        async with sessions() as s, s.begin():
            rows = await ConversationRepository(s).append_turn(
                user.customer_id, conversation_id, [AssistantMessage(text=DECLINED_MESSAGE)]
            )
        logger.info("action %s cancelled by the customer", action.id)
        return ActionResponse(
            action=ActionOut.model_validate(action, from_attributes=True),
            ok=True,
            message=DECLINED_MESSAGE,
            message_id=rows[-1].id,
        )
    finally:
        request.app.state.run_locks.release(conversation_id)
