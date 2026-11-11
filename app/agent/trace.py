"""A structured, serialisable record of everything one agent run did."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from app.agent.llm import Usage
from app.tools.base import ToolResult

Outcome = Literal["completed", "truncated", "error"]


class ModelCallStep(BaseModel):
    type: Literal["model_call"] = "model_call"
    latency_ms: float
    usage: Usage
    response: Literal["text", "tool_calls"]
    tool_calls: list[str] = Field(default_factory=list, description="Names of requested tools.")


class ToolCallStep(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: ToolResult
    latency_ms: float
    refused: bool = Field(default=False, description="Blocked by the loop, never executed.")
    proposed: bool = Field(
        default=False, description="A write turned into a proposal; not executed."
    )


Step = Annotated[ModelCallStep | ToolCallStep, Field(discriminator="type")]


class Trace(BaseModel):
    steps: list[Step] = Field(default_factory=list)
    outcome: Outcome | None = None
    reason: str | None = Field(default=None, description="Why the run was truncated or failed.")
    total_latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)

    @property
    def truncated(self) -> bool:
        return self.outcome == "truncated"

    def model_calls(self) -> list[ModelCallStep]:
        return [s for s in self.steps if isinstance(s, ModelCallStep)]

    def tool_calls(self) -> list[ToolCallStep]:
        return [s for s in self.steps if isinstance(s, ToolCallStep)]
