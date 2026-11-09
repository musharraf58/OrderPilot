import pytest
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.tools.base import Tool, ToolFailure
from app.tools.context import UserContext
from app.tools.registry import ToolRegistry
from tests.factories import CUSTOMER_ID


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(description="Text to echo back.")
    times: int = Field(default=1, ge=1, le=3)


class Echo(Tool):
    name = "echo"
    description = "Echo text."
    input_model = EchoInput
    kind = "read"

    async def _execute(self, args, ctx):
        return {"echo": args.text * args.times, "who": str(ctx.customer_id)}


class Boom(Tool):
    name = "boom"
    description = "Always crashes."
    input_model = EchoInput
    kind = "write"

    async def _execute(self, args, ctx):
        raise RuntimeError("secret internal detail")


class Refuse(Tool):
    name = "refuse"
    description = "Always refuses."
    input_model = EchoInput
    kind = "read"

    async def _execute(self, args, ctx):
        raise ToolFailure("forbidden", "No.")


@pytest.fixture
def ctx():
    return UserContext(customer_id=CUSTOMER_ID, store_token=SecretStr("s3cr3t-value"))


@pytest.fixture
def registry():
    r = ToolRegistry()
    for t in (Echo, Boom, Refuse):
        r.register(t(client=None))
    return r


async def test_execute_success(registry, ctx):
    result = await registry.execute("echo", {"text": "ab", "times": 2}, ctx)
    assert result.ok
    assert result.data == {"echo": "abab", "who": str(CUSTOMER_ID)}
    assert result.error is None


async def test_unknown_tool_is_a_result(registry, ctx):
    result = await registry.execute("nope", {}, ctx)
    assert not result.ok
    assert result.error.code == "unknown_tool"


@pytest.mark.parametrize(
    "args",
    [{}, {"text": 5}, {"text": "x", "times": 99}, {"text": "x", "customer_id": "evil"}],
)
async def test_invalid_args_is_a_result(registry, ctx, args):
    result = await registry.execute("echo", args, ctx)
    assert not result.ok
    assert result.error.code == "invalid_arguments"


async def test_invalid_args_message_does_not_echo_input(registry, ctx):
    result = await registry.execute("echo", {"text": "x", "times": "SENTINEL-VALUE"}, ctx)
    assert "SENTINEL-VALUE" not in result.error.message


async def test_tool_failure_becomes_error_result(registry, ctx):
    result = await registry.execute("refuse", {"text": "x"}, ctx)
    assert (result.ok, result.error.code, result.error.message) == (False, "forbidden", "No.")


async def test_unexpected_exception_never_escapes_or_leaks(registry, ctx):
    result = await registry.execute("boom", {"text": "x"}, ctx)
    assert not result.ok
    assert result.error.code == "internal_error"
    assert "secret internal detail" not in result.error.message


async def test_none_args_treated_as_empty(registry, ctx):
    result = await registry.execute("echo", None, ctx)
    assert result.error.code == "invalid_arguments"


def test_schemas_are_model_facing_and_exclude_ctx(registry):
    schemas = {s["name"]: s for s in registry.schemas()}
    assert set(schemas) == {"echo", "boom", "refuse"}
    echo = schemas["echo"]
    assert set(echo) == {"name", "description", "parameters"}
    props = echo["parameters"]["properties"]
    assert set(props) == {"text", "times"}
    assert not {"ctx", "customer_id", "store_token", "token"} & set(props)
    assert "title" not in echo["parameters"]
    assert echo["parameters"]["additionalProperties"] is False


def test_kind_is_marked_on_tools(registry):
    assert {t.name: t.kind for t in registry.list_tools()} == {
        "echo": "read",
        "boom": "write",
        "refuse": "read",
    }


def test_duplicate_registration_rejected(registry):
    with pytest.raises(ValueError):
        registry.register(Echo(client=None))


def test_user_context_hides_token_in_repr(ctx):
    assert "s3cr3t-value" not in repr(ctx)
