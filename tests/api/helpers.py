import json

from app.agent.llm import LLMResponse, ToolCall, Usage


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def text(t):
    return LLMResponse(text=t, usage=Usage(input_tokens=3, output_tokens=1))


def tool_calls(*specs):
    return LLMResponse(
        tool_calls=[
            ToolCall(
                id=f"g{i}",
                name=name,
                arguments=args,
                provider_data={"gemini_id": f"g{i}", "gemini_thought_signature": "U0lH"},
            )
            for i, (name, args) in enumerate(specs)
        ]
    )
