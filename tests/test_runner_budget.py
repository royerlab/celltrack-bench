from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from celltrack_bench import conditions as C
from celltrack_bench.runner import _turn_policy, call_once, call_once_anthropic


class _FakeResponses:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return SimpleNamespace(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
                output=[],
                output_text="",
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=6144,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=6144),
                ),
            )
        return SimpleNamespace(
            status="completed",
            incomplete_details=None,
            output=[],
            output_text='{"links": [], "unlinked_sources": []}',
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=128,
                output_tokens_details=SimpleNamespace(reasoning_tokens=64),
            ),
        )


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return SimpleNamespace(
                stop_reason="max_tokens",
                content=[],
                usage=SimpleNamespace(input_tokens=100, output_tokens=6144),
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(
                    type="text",
                    text='{"links": [], "unlinked_sources": []}',
                )
            ],
            usage=SimpleNamespace(input_tokens=120, output_tokens=128),
        )


def _args() -> Namespace:
    return Namespace(
        model="gpt-5.5",
        output_budget=8192,
        answer_reserve=2048,
        reasoning_effort="low",
        final_reasoning_effort="none",
    )


def test_turn_policy_keeps_reserve_outside_tool_turn() -> None:
    assert _turn_policy(8192, 2048, True, False) == (6144, True)
    assert _turn_policy(2048, 2048, True, False) == (2048, False)
    assert _turn_policy(8192, 2048, True, True) == (8192, False)


def test_responses_recovers_from_bounded_incomplete_tool_turn(tmp_path: Path) -> None:
    responses = _FakeResponses()
    client = SimpleNamespace(responses=responses)
    built = C.Built(
        text="Return JSON.",
        images=[],
        tools=[{"type": "function", "name": "zoom", "parameters": {}}],
        panels={},
    )

    result = call_once(client, built, _args(), tmp_path)

    assert [request["max_output_tokens"] for request in responses.requests] == [6144, 2048]
    assert "tools" in responses.requests[0]
    assert "tools" not in responses.requests[1]
    assert responses.requests[0]["reasoning"]["effort"] == "low"
    assert responses.requests[1]["reasoning"]["effort"] == "none"
    assert result["text"] == '{"links": [], "unlinked_sources": []}'
    assert result["truncated"] is False
    assert result["budget_exhausted"] is False
    assert result["recovered_incomplete_turns"] == 1


def test_anthropic_recovers_from_bounded_incomplete_tool_turn(tmp_path: Path) -> None:
    messages = _FakeAnthropicMessages()
    client = SimpleNamespace(messages=messages)
    built = C.Built(
        text="Return JSON.",
        images=[],
        tools=[{"type": "function", "name": "zoom", "parameters": {}}],
        panels={},
    )

    result = call_once_anthropic(client, built, _args(), tmp_path)

    assert [request["max_tokens"] for request in messages.requests] == [6144, 2048]
    assert "tools" in messages.requests[0]
    assert "tools" not in messages.requests[1]
    assert messages.requests[0]["thinking"] == {"type": "adaptive"}
    assert messages.requests[1]["thinking"] == {"type": "disabled"}
    assert messages.requests[0]["output_config"] == {"effort": "low"}
    assert messages.requests[1]["output_config"] == {"effort": "low"}
    assert result["text"] == '{"links": [], "unlinked_sources": []}'
    assert result["truncated"] is False
    assert result["budget_exhausted"] is False
    assert result["recovered_incomplete_turns"] == 1
