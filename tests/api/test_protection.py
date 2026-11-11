import asyncio

from app.agent.llm import LLMResponse
from app.core.limits import RateLimiter, RunLocks
from tests.api.helpers import parse_sse, text


async def test_rate_limit_returns_429_with_retry_after(chat, api_app, clock):
    api_app.state.rate_limiter = RateLimiter(limit=2, window_s=600, clock=clock)
    chat.script([text("a"), text("b"), text("c")])
    cid = await chat.new()
    assert (await chat.send(cid, "one")).status_code == 200
    clock.now += 100
    assert (await chat.send(cid, "two")).status_code == 200

    limited = await chat.send(cid, "three")
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "500"  # first hit leaves the window in 500s
    assert len(await chat.rows(cid)) == 4  # the rejected message was not stored

    clock.now += 500
    assert (await chat.send(cid, "three")).status_code == 200


async def test_rate_limit_is_per_customer(chat, api_app, clock):
    limiter = RateLimiter(limit=1, window_s=600, clock=clock)
    api_app.state.rate_limiter = limiter
    assert limiter.hit("someone-else") is None
    chat.script([text("a")])
    cid = await chat.new()
    assert (await chat.send(cid, "mine")).status_code == 200


async def test_second_message_while_running_is_409(chat):
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_answer():
        started.set()
        await release.wait()
        return LLMResponse(text="first answer")

    chat.script([slow_answer, text("third answer")])
    cid = await chat.new()

    first = asyncio.create_task(chat.send(cid, "first"))
    await asyncio.wait_for(started.wait(), 5)
    second = await chat.send(cid, "second")
    assert second.status_code == 409
    other = await chat.new()  # other conversations are not blocked
    chat.llm._steps.insert(0, LLMResponse(text="other answer"))
    assert (await chat.send(other, "elsewhere")).status_code == 200

    release.set()
    assert (await first).status_code == 200
    # Lock released after the run: the conversation accepts messages again.
    third = await chat.send(cid, "third")
    assert parse_sse(third.text)[-1][0] == "done"
    texts = [r.content.get("text") for r in await chat.rows(cid)]
    assert texts == ["first", "first answer", "third", "third answer"]


async def test_lock_is_released_after_a_failed_turn(chat):
    from app.agent.llm import LLMUnavailable

    chat.script([LLMUnavailable("down"), text("ok")])
    cid = await chat.new()
    await chat.send(cid, "one")
    assert (await chat.send(cid, "two")).status_code == 200


def test_run_lock_goes_stale(clock):
    locks = RunLocks(stale_after=90, clock=clock)
    assert locks.try_acquire("c")
    assert not locks.try_acquire("c")
    clock.now += 90
    assert locks.try_acquire("c")
    locks.release("c")
    assert locks.try_acquire("c")


def test_rate_limiter_sliding_window(clock):
    limiter = RateLimiter(limit=2, window_s=10, clock=clock)
    assert limiter.hit("k") is None
    clock.now += 4
    assert limiter.hit("k") is None
    assert limiter.hit("k") == 6
    clock.now += 6
    assert limiter.hit("k") is None
