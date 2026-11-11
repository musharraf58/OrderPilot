from app.agent.history import window
from app.agent.llm import (
    AssistantMessage,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)


def turn(q, tools=0):
    msgs = [UserMessage(text=q)]
    if tools:
        msgs.append(ToolCallMessage(calls=[ToolCall(id=f"{q}{i}", name="t") for i in range(tools)]))
        msgs += [ToolResultMessage(call_id=f"{q}{i}", name="t", result={}) for i in range(tools)]
    msgs.append(AssistantMessage(text=f"a-{q}"))
    return msgs


def assert_no_split_pairs(msgs):
    call_ids = {c.id for m in msgs if isinstance(m, ToolCallMessage) for c in m.calls}
    result_ids = {m.call_id for m in msgs if isinstance(m, ToolResultMessage)}
    assert call_ids == result_ids


def test_short_history_is_unchanged():
    msgs = turn("a") + turn("b")
    assert window(msgs, 20) == msgs


def test_window_always_starts_at_a_user_message_and_keeps_tool_pairs():
    msgs = turn("a", tools=2) + turn("b", tools=1) + turn("c") + turn("d", tools=3)
    for n in range(1, len(msgs) + 1):
        w = window(msgs, n)
        assert isinstance(w[0], UserMessage), n
        assert w == msgs[len(msgs) - len(w) :]  # a suffix of the history
        assert_no_split_pairs(w)


def test_window_never_exceeds_n_unless_a_single_turn_is_longer():
    msgs = turn("a") + turn("b") + turn("c", tools=4)  # last turn has 7 messages
    assert len(window(msgs, 7)) == 7
    w = window(msgs, 3)
    assert w[0].text == "c" and len(w) == 7  # starts at the long turn, not inside it


def test_zero_window_is_empty():
    assert window(turn("a"), 0) == []
