"""A deterministic model that replays scripted turns.

Used by the tests and by ``--offline`` demo runs: every agent behaviour
(tool errors, retries, missing submits, refusals) can be reproduced exactly.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from sales_agents.llm.base import ModelTurn, ToolCall, Usage

Step = ModelTurn | Callable[[list[dict[str, Any]]], ModelTurn]

_ids = itertools.count(1)


def tool_turn(*calls: tuple[str, dict[str, Any]], text: str = "") -> ModelTurn:
    tool_calls = [ToolCall(id=f"toolu_{next(_ids)}", name=n, input=i) for n, i in calls]
    raw: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    raw += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in tool_calls]
    return ModelTurn(raw, text, tool_calls, "tool_use", Usage(500, 120))


def text_turn(text: str, stop_reason: str = "end_turn") -> ModelTurn:
    return ModelTurn([{"type": "text", "text": text}], text, [], stop_reason, Usage(300, 40))


class ScriptedModel:
    def __init__(self, steps: list[Step], model: str = "scripted") -> None:
        self.model = model
        self._steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    def create(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        if not self._steps:
            raise AssertionError("ScriptedModel ran out of steps")
        step = self._steps.pop(0)
        return step(messages) if callable(step) else step
