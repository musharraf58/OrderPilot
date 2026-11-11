import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.history import window
from app.agent.llm import LLMClient, Message
from app.agent.loop import Agent, Finished, TextDelta, ToolFinished, ToolStarted
from app.api.deps import CurrentUser, get_llm, get_registry
from app.core.config import get_settings
from app.core.limits import RateLimiter, RunLocks
from app.db.models import Conversation
from app.db.models import Message as MessageRow
from app.db.repositories import ConversationRepository, PendingActionRepository
from app.db.repositories.conversations import to_message
from app.tools.context import UserContext
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])

NOT_FOUND = "Conversation not found."


class ConversationOut(BaseModel):
    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageItem(BaseModel):
    type: Literal["message"] = "message"
    id: UUID
    role: Literal["user", "assistant"]
    text: str
    created_at: datetime


class ToolItem(BaseModel):
    type: Literal["tool"] = "tool"
    name: str
    ok: bool
    created_at: datetime


class PendingActionOut(BaseModel):
    id: UUID
    tool_name: str
    summary: str
    expires_at: datetime


class ConversationDetail(ConversationOut):
    items: list[MessageItem | ToolItem]
    # A proposed action waiting for the customer's confirm/cancel, if any.
    pending_action: PendingActionOut | None


class SendMessage(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _length(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Message must not be empty.")
        limit = get_settings().MAX_MESSAGE_CHARS
        if len(text) > limit:
            raise ValueError(f"Message is too long (at most {limit} characters).")
        return text


def _sessionmaker(request: Request) -> async_sessionmaker[AsyncSession]:
    return request.app.state.sessionmaker


SessionMaker = Annotated[async_sessionmaker[AsyncSession], Depends(_sessionmaker)]


def _out(c: Conversation) -> ConversationOut:
    return ConversationOut(id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at)


def _display_items(rows: Sequence[MessageRow]) -> list[MessageItem | ToolItem]:
    """What a UI shows: chat text plus one-line tool summaries. No arguments, raw results
    or provider data."""
    items: list[MessageItem | ToolItem] = []
    for row in rows:
        if row.role in ("user", "assistant"):
            items.append(
                MessageItem(
                    id=row.id, role=row.role, text=row.content["text"], created_at=row.created_at
                )
            )
        elif row.role == "tool_result":
            error = row.content["result"].get("error") or {}
            if error.get("code") == "awaiting_confirmation":
                continue  # shown as the confirmation prompt, not as a failed tool
            items.append(
                ToolItem(
                    name=row.content["name"],
                    ok=bool(row.content["result"].get("ok")),
                    created_at=row.created_at,
                )
            )
    return items


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(user: CurrentUser, sessions: SessionMaker) -> ConversationOut:
    async with sessions() as s, s.begin():
        conversation = await ConversationRepository(s).create(user.customer_id)
    return _out(conversation)


@router.get("")
async def list_conversations(user: CurrentUser, sessions: SessionMaker) -> list[ConversationOut]:
    """The customer's conversations, most recently active first."""
    async with sessions() as s:
        return [
            _out(c) for c in await ConversationRepository(s).list_for_customer(user.customer_id)
        ]


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: UUID, user: CurrentUser, sessions: SessionMaker
) -> ConversationDetail:
    async with sessions() as s:
        repo = ConversationRepository(s)
        conversation = await repo.get(user.customer_id, conversation_id)
        if conversation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)
        rows = await repo.load_messages(user.customer_id, conversation_id)
        action = await PendingActionRepository(s).current(user.customer_id, conversation_id)
    return ConversationDetail(
        **_out(conversation).model_dump(),
        items=_display_items(rows),
        pending_action=PendingActionOut.model_validate(action, from_attributes=True)
        if action
        else None,
    )


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _error_code(reason: str | None) -> str:
    if reason in ("timeout", "max_steps"):
        return reason
    return "llm_error"


async def stream_turn(
    agent: Agent,
    sessions: async_sessionmaker[AsyncSession],
    user: UserContext,
    conversation_id: UUID,
    text: str,
    history: list[Message],
    release: Callable[[], None],
    action_ttl_s: float,
) -> AsyncIterator[str]:
    """Run the agent and translate its events to SSE; persist the turn when it ends."""
    try:
        async for event in agent.run_stream(text, history, user):
            match event:
                case ToolStarted(name=name):
                    yield sse("tool_started", {"name": name})
                case ToolFinished(name=name, ok=ok):
                    yield sse("tool_finished", {"name": name, "ok": ok})
                case TextDelta(text=delta):
                    yield sse("text_delta", {"text": delta})
                case Finished(result=result):
                    # result.history = the history we sent + this turn's messages. For a
                    # failed or truncated turn it is only the user message and the fallback.
                    new_messages = result.history[len(history) :]
                    action = None
                    # One transaction: the turn's messages and (if any) the proposed action.
                    async with sessions() as s, s.begin():
                        rows = await ConversationRepository(s).append_turn(
                            user.customer_id, conversation_id, new_messages
                        )
                        if result.proposal is not None:
                            action = await PendingActionRepository(s).create(
                                user.customer_id,
                                conversation_id,
                                result.proposal.tool_name,
                                result.proposal.arguments,
                                result.proposal.summary,
                                ttl_s=action_ttl_s,
                            )
                    trace = result.trace
                    logger.info(
                        "turn %s: steps=%d tools=%s latency_ms=%.0f tokens_in=%d tokens_out=%d",
                        trace.outcome,
                        len(trace.model_calls()),
                        [t.name for t in trace.tool_calls()],
                        trace.total_latency_ms,
                        trace.usage.input_tokens,
                        trace.usage.output_tokens,
                    )
                    if trace.outcome != "completed":
                        yield sse(
                            "error",
                            {"code": _error_code(trace.reason), "message": result.final_text},
                        )
                    if action is not None:
                        yield sse(
                            "confirmation_required",
                            {
                                "action_id": str(action.id),
                                "summary": action.summary,
                                "expires_at": action.expires_at.isoformat(),
                            },
                        )
                    yield sse("done", {"message_id": str(rows[-1].id)})
    except Exception:
        logger.exception("conversation turn failed")
        yield sse(
            "error",
            {"code": "internal_error", "message": "Something went wrong. Please try again."},
        )
    finally:
        release()


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: UUID,
    body: SendMessage,
    request: Request,
    user: CurrentUser,
    sessions: SessionMaker,
    llm: Annotated[LLMClient, Depends(get_llm)],
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> StreamingResponse:
    """Run one agent turn and stream it as Server-Sent Events.

    Every check that can fail with a plain HTTP error (auth, 404, 409, 429) happens here,
    before the stream starts. After that the outcome is reported as SSE events.
    """
    settings = get_settings()
    async with sessions() as s:
        if await ConversationRepository(s).get(user.customer_id, conversation_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)

    # No await between the lock and the rate-limit check, so they cannot interleave with
    # another request. History is loaded only once we hold the lock, so it can't change
    # underneath this turn (e.g. by a confirm finishing).
    locks: RunLocks = request.app.state.run_locks
    limiter: RateLimiter = request.app.state.rate_limiter
    if not locks.try_acquire(conversation_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "A reply is already being generated in this conversation."
        )
    retry_after = limiter.hit(user.customer_id)
    if retry_after is not None:
        locks.release(conversation_id)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many messages. Please wait a moment.",
            headers={"Retry-After": str(retry_after)},
        )
    try:
        async with sessions() as s, s.begin():
            # The customer moved on: an action still waiting for confirmation no longer applies.
            await PendingActionRepository(s).close_pending(user.customer_id, conversation_id)
            rows = await ConversationRepository(s).load_messages(user.customer_id, conversation_id)
    except BaseException:
        locks.release(conversation_id)
        raise
    history = window([to_message(r) for r in rows], settings.HISTORY_WINDOW)

    agent = Agent(
        llm,
        registry,
        max_steps=settings.AGENT_MAX_STEPS,
        timeout_s=settings.AGENT_TIMEOUT_S,
        allow_writes=settings.AGENT_ALLOW_WRITES,
    )
    return StreamingResponse(
        stream_turn(
            agent,
            sessions,
            user,
            conversation_id,
            body.text,
            history,
            release=lambda: locks.release(conversation_id),
            action_ttl_s=settings.ACTION_TTL_S,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
