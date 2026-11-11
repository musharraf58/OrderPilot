import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from app.clients.store import StoreClient
from app.tools.base import Tool, ToolResult
from app.tools.context import UserContext

logger = logging.getLogger(__name__)


def _strip_titles(node: Any) -> Any:
    """Drop pydantic's auto-generated "title" keys: they cost tokens and add nothing."""
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def _describe_validation_error(exc: ValidationError) -> str:
    # Field name and reason only: never echo the raw input back.
    problems = [f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()]
    return "Invalid arguments. " + "; ".join(problems)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """Model-facing tool definitions. UserContext is never part of these."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": _strip_titles(t.input_model.model_json_schema()),
            }
            for t in self._tools.values()
        ]

    def prepare(
        self, name: str, raw_args: dict[str, Any] | None
    ) -> tuple[Tool, BaseModel] | ToolResult:
        """Look up the tool and validate the arguments: (tool, args), or a failure result."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure("unknown_tool", f"There is no tool named '{name}'.")
        try:
            return tool, tool.input_model.model_validate(raw_args or {})
        except ValidationError as exc:
            return ToolResult.failure("invalid_arguments", _describe_validation_error(exc))

    async def execute(
        self, name: str, raw_args: dict[str, Any] | None, ctx: UserContext
    ) -> ToolResult:
        """Validate the model's arguments and run the tool. Never raises."""
        prepared = self.prepare(name, raw_args)
        if isinstance(prepared, ToolResult):
            return prepared
        tool, args = prepared
        try:
            return await tool.run(args, ctx)
        except Exception:
            logger.exception("tool %s crashed", name)
            return ToolResult.failure("internal_error", "The tool failed unexpectedly.")


def build_default_registry(client: StoreClient) -> ToolRegistry:
    from app.tools.orders import CancelOrder, GetOrder, ListMyOrders
    from app.tools.products import GetProduct, SearchProducts

    registry = ToolRegistry()
    for tool_cls in (SearchProducts, GetProduct, ListMyOrders, GetOrder, CancelOrder):
        registry.register(tool_cls(client))
    return registry
