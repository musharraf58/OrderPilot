import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from app.clients.store import (
    StoreClient,
    StoreConflict,
    StoreError,
    StoreForbidden,
    StoreNotFound,
    StoreUnauthorized,
    StoreUnavailable,
)
from app.tools.context import UserContext

logger = logging.getLogger(__name__)

ErrorCode = Literal[
    "not_found",
    "forbidden",
    "unauthorized",
    "conflict",
    "not_cancellable",
    "not_allowed",
    "awaiting_confirmation",
    "store_unavailable",
    "store_error",
    "knowledge_unavailable",
    "knowledge_rate_limited",
    "knowledge_error",
    "invalid_arguments",
    "unknown_tool",
    "internal_error",
]


class ToolError(BaseModel):
    code: ErrorCode
    message: str


class ToolResult(BaseModel):
    """What every tool returns: either data or a stable error code, never an exception."""

    ok: bool
    data: Any | None = None
    error: ToolError | None = None

    @classmethod
    def success(cls, data: Any) -> "ToolResult":
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, code: ErrorCode, message: str) -> "ToolResult":
        return cls(ok=False, error=ToolError(code=code, message=message))


class ToolFailure(Exception):
    """Raised inside a tool for an expected failure; converted to ok=false by Tool.run."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# Store exception -> (stable code, message safe to show the model).
_STORE_ERRORS: list[tuple[type[StoreError], ErrorCode, str]] = [
    (StoreNotFound, "not_found", "The requested item was not found."),
    (StoreForbidden, "forbidden", "You are not allowed to do that."),
    (StoreUnauthorized, "unauthorized", "The customer's session is invalid or expired."),
    (StoreConflict, "conflict", "The request conflicts with the current state."),
    (
        StoreUnavailable,
        "store_unavailable",
        "The store is temporarily unavailable. Try again later.",
    ),
]


class Tool(ABC):
    """Base class for a model-callable tool.

    Subclasses set the class attributes and implement `_execute`. Callers use `run`,
    which converts every expected failure into a ToolResult.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    kind: ClassVar[Literal["read", "write"]]

    def __init__(self, client: StoreClient) -> None:
        self.client = client

    @abstractmethod
    async def _execute(self, args: Any, ctx: UserContext) -> Any:
        """Do the work and return JSON-serialisable data, or raise ToolFailure/StoreError."""

    async def describe(self, args: Any, ctx: UserContext) -> str | None:
        """For write tools: a human-readable summary of what the action will do, built from
        fresh store data, for the customer to confirm. Raise ToolFailure/StoreError to
        refuse early (e.g. the target doesn't exist or the action isn't possible). Tools
        without a description cannot be proposed."""
        return None

    def outcome_message(self, result: ToolResult) -> str:
        """Code-written text telling the customer how a confirmed action went."""
        if result.ok:
            return "Done."
        return "That didn't work, and the action may not have been completed."

    async def run(self, args: Any, ctx: UserContext) -> ToolResult:
        return await self._guard(self._execute(args, ctx))

    async def propose(self, args: Any, ctx: UserContext) -> ToolResult:
        """Run `describe` with the same error handling as `run`: ok with
        {"summary": ...}, or a failure explaining why the action can't be proposed."""
        return await self._guard(self._summary(args, ctx))

    async def _summary(self, args: Any, ctx: UserContext) -> dict[str, str]:
        summary = await self.describe(args, ctx)
        if summary is None:
            raise ToolFailure("not_allowed", "This action can't be confirmed here.")
        return {"summary": summary}

    async def _guard(self, work: Awaitable[Any]) -> ToolResult:
        try:
            return ToolResult.success(await work)
        except ToolFailure as exc:
            return ToolResult.failure(exc.code, exc.message)
        except StoreError as exc:
            for exc_type, code, message in _STORE_ERRORS:
                if isinstance(exc, exc_type):
                    return ToolResult.failure(code, message)
            logger.warning("tool %s: unexpected store error: %s", self.name, exc.message)
            return ToolResult.failure("store_error", "The store returned an unexpected error.")
