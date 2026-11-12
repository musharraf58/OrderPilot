import json

import pytest

from app.tools.policies import policy_title
from tests.factories import search_result


async def test_returns_compact_relevant_passages_best_first(registry, ctx, documind_search):
    route = documind_search(
        search_result(
            document_name="returns-and-refunds.md",
            section_title="Return window",
            content="You can return most items within 30 days of delivery. " * 30,
            distance=0.40,
        ),
        search_result(
            document_name="shipping.md",
            section_title="Late or lost parcels",
            content="Contact support after 10 business days.",
            distance=0.30,
        ),
    )
    result = await registry.execute(
        "search_policies", {"query": "How long do I have to return something?"}, ctx
    )
    assert result.ok
    results = result.data["results"]
    assert [r["policy"] for r in results] == ["Shipping policy", "Returns and refunds policy"]
    assert results[0] == {
        "policy": "Shipping policy",
        "section": "Late or lost parcels",
        "passage": "Contact support after 10 business days.",
        "relevance": 0.7,
    }
    assert len(results[1]["passage"]) <= 600 and results[1]["passage"].endswith("…")
    assert "note" not in result.data
    # Only what the model needs: no ids, ranks or raw distances.
    assert "chunk_id" not in json.dumps(result.data) and "distance" not in json.dumps(result.data)
    sent = json.loads(route.calls.last.request.read())
    assert sent == {"query": "How long do I have to return something?", "limit": 5}


async def test_low_relevance_results_are_dropped(registry, ctx, documind_search):
    documind_search(
        search_result(section_title="Kept", distance=0.55),  # relevance 0.45
        search_result(section_title="Dropped", distance=0.61),  # relevance 0.39
    )
    result = await registry.execute("search_policies", {"query": "Which cards do you take?"}, ctx)
    assert [r["section"] for r in result.data["results"]] == ["Kept"]


async def test_limit_is_respected(registry, ctx, documind_search):
    documind_search(*(search_result(distance=0.2 + i / 100) for i in range(5)))
    result = await registry.execute(
        "search_policies", {"query": "Tell me about shipping", "limit": 2}, ctx
    )
    assert len(result.data["results"]) == 2


@pytest.mark.parametrize("hits", [[], [search_result(distance=0.8)]], ids=["empty", "irrelevant"])
async def test_nothing_relevant_is_ok_with_a_note(registry, ctx, documind_search, hits):
    documind_search(*hits)
    result = await registry.execute(
        "search_policies", {"query": "Do you price match with other stores?"}, ctx
    )
    assert result.ok
    assert result.data == {"results": [], "note": "no_relevant_policy"}


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (503, "knowledge_unavailable"),
        (429, "knowledge_rate_limited"),
        (500, "knowledge_unavailable"),
        (422, "knowledge_error"),
    ],
)
async def test_documind_failures_are_error_results(registry, ctx, documind_search, status, code):
    documind_search(status=status)
    result = await registry.execute("search_policies", {"query": "What is the return policy?"}, ctx)
    assert (result.ok, result.error.code) == (False, code)


async def test_bad_service_credentials_are_an_error_result(registry, ctx, documind):
    documind.post("/auth/login").respond(401)
    result = await registry.execute("search_policies", {"query": "What is the return policy?"}, ctx)
    assert (result.ok, result.error.code) == (False, "knowledge_error")
    assert "password" not in result.error.message.lower()


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"query": "hi"},
        {"query": "x" * 301},
        {"query": "returns?", "limit": 0},
        {"query": "returns?", "limit": 6},
        {"query": "returns?", "customer_id": "someone"},
    ],
)
async def test_invalid_input(registry, ctx, args):
    result = await registry.execute("search_policies", args, ctx)
    assert (result.ok, result.error.code) == (False, "invalid_arguments")


def test_schema_is_read_only_and_has_no_identity(registry):
    schema = next(s for s in registry.schemas() if s["name"] == "search_policies")
    assert registry.get("search_policies").kind == "read"
    params = schema["parameters"]
    assert set(params["properties"]) == {"query", "limit"}
    assert params["required"] == ["query"]
    assert params["additionalProperties"] is False
    assert params["properties"]["query"]["maxLength"] == 300


def test_not_registered_without_documind(store_client):
    from app.tools.registry import build_default_registry

    assert build_default_registry(store_client).get("search_policies") is None


def test_policy_title():
    assert policy_title("returns-and-refunds.md") == "Returns and refunds policy"
    assert policy_title("order-cancellation.md") == "Order cancellation policy"
    assert policy_title("privacy_policy.md") == "Privacy policy"
