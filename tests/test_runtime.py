from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from sales_agents.llm import ScriptedModel, Usage
from sales_agents.llm.anthropic import ClaudeModel
from sales_agents.llm.scripted import text_turn, tool_turn
from sales_agents.runtime import Agent, AgentError, Tool, ToolError, strict_schema
from sales_agents.tracing import Tracer


class Lookup(BaseModel):
    city: str


class Inner(BaseModel):
    label: str


class Weather(BaseModel):
    city: str
    celsius: float = Field(ge=-60, le=60)
    tags: list[Inner]


def lookup(args: Lookup) -> dict[str, Any]:
    if args.city == "Atlantis":
        raise ToolError("City not found.")
    if args.city == "Crash":
        raise RuntimeError("boom")
    return {"city": args.city, "celsius": 21.5}


def make_agent(max_turns: int = 6) -> Agent[Weather]:
    return Agent(
        name="weather",
        system="Look up the weather, then submit it.",
        tools=[Tool("lookup", "Get weather for a city", Lookup, lookup)],
        output_model=Weather,
        submit_description="Submit the weather report",
        max_turns=max_turns,
    )


GOOD = {"city": "Lisbon", "celsius": 21.5, "tags": [{"label": "sunny"}]}


def test_strict_schema_closes_every_object() -> None:
    schema = strict_schema(Weather)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["city", "celsius", "tags"]
    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False and inner["required"] == ["label"]


def test_happy_path_returns_typed_output_and_usage() -> None:
    model = ScriptedModel(
        [tool_turn(("lookup", {"city": "Lisbon"})), tool_turn(("submit_weather", GOOD))]
    )
    tracer = Tracer()
    result = make_agent().run("Weather in Lisbon?", model, tracer, lead_id="L1")
    assert result.output.celsius == 21.5
    assert result.turns == 2
    assert result.usage == Usage(1000, 240)
    assert result.tool_calls == ["lookup", "submit_weather"]
    assert [e["event"] for e in tracer.events] == [
        "model_call",
        "tool_call",
        "model_call",
        "agent_done",
    ]
    # tools sent to the model include the submit tool, all strict
    tools = model.requests[0]["tools"]
    assert [t["name"] for t in tools] == ["lookup", "submit_weather"]
    assert all(t["strict"] for t in tools)
    # the tool result goes back in one user message with the right id
    second = model.requests[1]["messages"]
    assert second[1]["role"] == "assistant"
    assert second[2]["content"][0]["tool_use_id"] == second[1]["content"][0]["id"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"city": "Atlantis"}, "City not found."),
        ({"town": "Lisbon"}, "Invalid input: city: Field required"),
        ({"city": "Crash"}, "failed unexpectedly"),
    ],
)
def test_tool_errors_are_returned_to_the_model(args: dict[str, Any], message: str) -> None:
    model = ScriptedModel([tool_turn(("lookup", args)), tool_turn(("submit_weather", GOOD))])
    tracer = Tracer()
    make_agent().run("q", model, tracer)
    result = model.requests[1]["messages"][2]["content"][0]
    assert result["is_error"] is True and message in result["content"]
    assert tracer.of("tool_call")[0]["is_error"] is True


def test_unknown_tool_and_parallel_calls_share_one_message() -> None:
    model = ScriptedModel(
        [
            tool_turn(("lookup", {"city": "Lisbon"}), ("teleport", {})),
            tool_turn(("submit_weather", GOOD)),
        ]
    )
    make_agent().run("q", model)
    results = model.requests[1]["messages"][2]["content"]
    assert len(results) == 2
    assert "is_error" not in results[0] and "Unknown tool" in results[1]["content"]


def test_invalid_submit_is_corrected() -> None:
    bad = {"city": "Lisbon", "celsius": 999, "tags": []}
    model = ScriptedModel([tool_turn(("submit_weather", bad)), tool_turn(("submit_weather", GOOD))])
    tracer = Tracer()
    result = make_agent().run("q", model, tracer)
    assert result.turns == 2
    feedback = model.requests[1]["messages"][2]["content"][0]
    assert feedback["is_error"] and "celsius" in feedback["content"]
    assert len(tracer.of("submit_invalid")) == 1


def test_nudges_once_then_fails() -> None:
    model = ScriptedModel([text_turn("It is sunny."), tool_turn(("submit_weather", GOOD))])
    assert make_agent().run("q", model).turns == 2
    assert "Call submit_weather now" in model.requests[1]["messages"][-1]["content"]

    with pytest.raises(AgentError, match="without calling"):
        make_agent().run("q", ScriptedModel([text_turn("a"), text_turn("b")]))
    with pytest.raises(AgentError, match="without calling"):
        make_agent().run("q", ScriptedModel([text_turn("cut", stop_reason="max_tokens")]))


def test_refusal_and_turn_budget() -> None:
    with pytest.raises(AgentError, match="declined"):
        make_agent().run("q", ScriptedModel([text_turn("", stop_reason="refusal")]))
    loop = [tool_turn(("lookup", {"city": "Lisbon"})) for _ in range(3)]
    with pytest.raises(AgentError, match="after 3 turns"):
        make_agent(max_turns=3).run("q", ScriptedModel(loop))


def test_scripted_model_supports_callables_and_runs_out() -> None:
    model = ScriptedModel([lambda messages: tool_turn(("submit_weather", GOOD))])
    assert make_agent().run("q", model).output.city == "Lisbon"
    with pytest.raises(AssertionError):
        model.create("s", [], [])


def test_tracer_writes_jsonl_and_truncates(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path / "runs" / "t.jsonl")
    tracer.emit("tool_call", input="x" * 5000, other=1)
    line = json.loads((tmp_path / "runs" / "t.jsonl").read_text())
    assert line["event"] == "tool_call" and len(line["input"]) < 2100 and line["other"] == 1


def test_usage_cost_includes_cache_reads_and_writes() -> None:
    assert Usage(1_000_000, 100_000).cost_usd("claude-opus-5-5") == pytest.approx(6.0)
    cached = Usage(0, 0, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert cached.cost_usd("claude-opus-5-5") == pytest.approx(0.20 + 5.0)
    assert Usage(10, 10).cost_usd("unknown") == 0.0
    assert Usage(25, 0, 75, 0).cache_hit_rate == 0.75
    assert Usage().cache_hit_rate == 0.0
    assert (Usage(1, 1, 1, 1) + Usage(1, 1, 1, 1)) == Usage(2, 2, 2, 2)


def test_claude_model_maps_response() -> None:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Looking up. "),
            SimpleNamespace(type="tool_use", id="toolu_1", name="lookup", input={"city": "Porto"}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            cache_read_input_tokens=900,
            cache_creation_input_tokens=None,
        ),
    )
    turn = ClaudeModel(client=client).create("sys", [{"role": "user", "content": "q"}], [])
    assert turn.tool_calls[0].input == {"city": "Porto"}
    assert turn.text == "Looking up. " and turn.usage == Usage(11, 7, 900, 0)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5" and kwargs["output_config"] == {"effort": "medium"}
    assert "tool_choice" not in kwargs
    assert kwargs["cache_control"] == {"type": "ephemeral"}
