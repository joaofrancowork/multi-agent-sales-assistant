"""A small, explicit agent runtime on top of the Messages API tool-use loop.

Each agent has a system prompt, a set of typed tools, and one *submit* tool whose
input schema is the agent's output. The agent is done when it calls the submit
tool with input that validates. Everything else is guardrails:

* tool inputs are validated with Pydantic; validation and tool errors go back to
  the model as ``is_error`` results so it can correct itself;
* all tool results of a turn are returned in a single message (keeps parallel
  tool use working);
* a turn budget, one nudge if the model stops without submitting, refusal handling;
* every step is traced with latency and token usage.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from sales_agents.llm.base import ChatModel, ToolCall, Usage
from sales_agents.tracing import Tracer

log = logging.getLogger(__name__)

In = TypeVar("In", bound=BaseModel)
Out = TypeVar("Out", bound=BaseModel)


class ToolError(Exception):
    """An expected tool failure whose message is safe to show the model."""


class AgentError(Exception):
    """The agent could not produce a valid output."""


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema accepted by strict tool use: closed objects, every field required."""
    schema = model.model_json_schema()

    def close(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(schema)
    schema.pop("title", None)
    return schema


@dataclass
class Tool(Generic[In]):
    name: str
    description: str
    input_model: type[In]
    fn: Callable[[In], Any]

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": strict_schema(self.input_model),
            "strict": True,
        }


@dataclass
class AgentResult(Generic[Out]):
    output: Out
    usage: Usage
    turns: int
    tool_calls: list[str] = field(default_factory=list)


class Agent(Generic[Out]):
    def __init__(
        self,
        name: str,
        system: str,
        tools: list[Tool[Any]],
        output_model: type[Out],
        submit_description: str,
        max_turns: int = 8,
    ) -> None:
        self.name = name
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.output_model = output_model
        self.submit_name = f"submit_{name}"
        self.max_turns = max_turns
        self._specs = [t.spec() for t in tools] + [
            {
                "name": self.submit_name,
                "description": submit_description,
                "input_schema": strict_schema(output_model),
                "strict": True,
            }
        ]

    def run(
        self, task: str, model: ChatModel, tracer: Tracer | None = None, lead_id: str = ""
    ) -> AgentResult[Out]:
        tracer = tracer or Tracer()
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        usage = Usage()
        called: list[str] = []
        nudged = False

        for turn in range(1, self.max_turns + 1):
            started = time.perf_counter()
            reply = model.create(self.system, messages, self._specs)
            usage = usage + reply.usage
            tracer.emit(
                "model_call",
                agent=self.name,
                lead_id=lead_id,
                turn=turn,
                stop_reason=reply.stop_reason,
                tool_calls=[c.name for c in reply.tool_calls],
                input_tokens=reply.usage.input_tokens,
                output_tokens=reply.usage.output_tokens,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            if reply.stop_reason == "refusal":
                raise AgentError(f"{self.name}: the model declined the task")
            messages.append({"role": "assistant", "content": reply.raw_content})

            if not reply.tool_calls:
                if nudged or reply.stop_reason == "max_tokens":
                    raise AgentError(f"{self.name}: finished without calling {self.submit_name}")
                nudged = True
                messages.append(
                    {"role": "user", "content": f"Call {self.submit_name} now with your result."}
                )
                continue

            results: list[dict[str, Any]] = []
            for call in reply.tool_calls:
                called.append(call.name)
                if call.name == self.submit_name:
                    output = self._validate_submit(call, tracer, lead_id)
                    if isinstance(output, dict):
                        results.append(output)
                        continue
                    tracer.emit("agent_done", agent=self.name, lead_id=lead_id, turns=turn)
                    return AgentResult(output, usage, turn, called)
                else:
                    results.append(self._execute(call, tracer, lead_id))
            messages.append({"role": "user", "content": results})

        raise AgentError(f"{self.name}: no valid result after {self.max_turns} turns")

    def _validate_submit(
        self, call: ToolCall, tracer: Tracer, lead_id: str
    ) -> Out | dict[str, Any]:
        try:
            return self.output_model.model_validate(call.input)
        except ValidationError as exc:
            tracer.emit("submit_invalid", agent=self.name, lead_id=lead_id, output=str(exc))
            return _result(call.id, f"Invalid result, fix these fields: {_errors(exc)}", True)

    def _execute(self, call: ToolCall, tracer: Tracer, lead_id: str) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        started = time.perf_counter()
        is_error = True
        if tool is None:
            content = f"Unknown tool {call.name!r}."
        else:
            try:
                value = tool.fn(tool.input_model.model_validate(call.input))
                content = value if isinstance(value, str) else json.dumps(value, default=str)
                is_error = False
            except ValidationError as exc:
                content = f"Invalid input: {_errors(exc)}"
            except ToolError as exc:
                content = str(exc)
            except Exception:
                log.exception("tool %s crashed", call.name)
                content = "The tool failed unexpectedly. Try a different approach."
        tracer.emit(
            "tool_call",
            agent=self.name,
            lead_id=lead_id,
            tool=call.name,
            input=call.input,
            output=content,
            is_error=is_error,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return _result(call.id, content, is_error)


def _result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def _errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors())
