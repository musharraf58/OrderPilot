from collections.abc import Sequence

from app.agent.llm import Message, UserMessage


def window(messages: Sequence[Message], max_messages: int) -> list[Message]:
    """The most recent part of a conversation to send to the model.

    Always starts at a user message. A turn's tool calls and tool results sit between its
    user message and its answer, so starting at a user message can never separate a tool
    call from its result. If one turn alone is longer than `max_messages`, the window
    starts at that turn's user message rather than cutting into it.
    """
    if max_messages <= 0:
        return []
    if len(messages) <= max_messages:
        return list(messages)
    for i in range(len(messages) - max_messages, len(messages)):
        if isinstance(messages[i], UserMessage):
            return list(messages[i:])
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], UserMessage):
            return list(messages[i:])
    return list(messages)
