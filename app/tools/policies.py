import logging
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.clients.documind import (
    DocuMindClient,
    DocuMindError,
    DocuMindRateLimited,
    DocuMindUnavailable,
    SearchResult,
)
from app.tools.base import Tool, ToolFailure
from app.tools.context import UserContext

logger = logging.getLogger(__name__)

_PASSAGE_LIMIT = 600
NO_RELEVANT_POLICY = "no_relevant_policy"


class SearchPoliciesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=300,
        description=(
            "The customer's policy question as a full sentence, in their own words, e.g. "
            "'Can I cancel an order after it has shipped?'. Full questions find better "
            "passages than single keywords."
        ),
    )
    limit: int = Field(default=3, ge=1, le=5, description="Maximum number of passages.")


def policy_title(document_name: str) -> str:
    """'returns-and-refunds.md' -> 'Returns and refunds policy'."""
    words = PurePosixPath(document_name).stem.replace("-", " ").replace("_", " ").strip()
    title = words[:1].upper() + words[1:]
    return title if title.lower().endswith("policy") else f"{title} policy"


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _PASSAGE_LIMIT:
        return text
    return text[: _PASSAGE_LIMIT - 1].rstrip() + "…"


class SearchPolicies(Tool):
    name = "search_policies"
    description = (
        "Search the store's official policies (shipping, returns and refunds, order "
        "cancellation, payment) and return the most relevant passages. Use it for any "
        "question about store rules: delivery times and costs, returns, refunds, which "
        "orders can be cancelled, accepted payment methods. Pass the customer's question as "
        "a full sentence. Do NOT use it for questions about the customer's own orders (use "
        "the order tools) or about specific products (use the product tools). If it returns "
        "no passages (note: no_relevant_policy), the policies do not cover the question."
    )
    input_model = SearchPoliciesInput
    kind = "read"

    def __init__(self, client: DocuMindClient, min_relevance: float) -> None:
        super().__init__(client)
        self.min_relevance = min_relevance

    async def _execute(self, args: SearchPoliciesInput, ctx: UserContext) -> dict[str, Any]:
        try:
            # Ask for a few extra so the threshold still leaves up to `limit` passages.
            hits = await self.client.search(args.query, limit=args.limit + 2)
        except DocuMindRateLimited as exc:
            raise ToolFailure(
                "knowledge_rate_limited", "The policy library is busy. Try again in a minute."
            ) from exc
        except DocuMindUnavailable as exc:
            raise ToolFailure(
                "knowledge_unavailable", "The policy library can't be reached right now."
            ) from exc
        except DocuMindError as exc:
            logger.warning("search_policies failed: %s", exc.message)
            raise ToolFailure("knowledge_error", "The policy library returned an error.") from exc

        relevant = sorted(
            (h for h in hits if 1 - h.distance >= self.min_relevance), key=lambda h: h.distance
        )[: args.limit]
        if not relevant:
            return {"results": [], "note": NO_RELEVANT_POLICY}
        return {"results": [self._compact(h) for h in relevant]}

    @staticmethod
    def _compact(hit: SearchResult) -> dict[str, Any]:
        return {
            "policy": policy_title(hit.document_name),
            "section": hit.section_title,
            "passage": _truncate(hit.content),
            "relevance": round(1 - hit.distance, 2),
        }
