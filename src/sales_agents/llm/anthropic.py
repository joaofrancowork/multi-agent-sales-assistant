from __future__ import annotations

from typing import Any, Literal

import anthropic

from sales_agents.llm.base import ModelTurn, ToolCall, Usage

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ClaudeModel:
    """Claude via the Messages API with client-side tools.

    Forced ``tool_choice`` is not available on current models, so agents are told
    in the system prompt to finish with their submit tool, and the runtime nudges
    once if they do not.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: Effort = "medium",
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 16000,
    ) -> None:
        self.model = model
        self._effort = effort
        self._client = client or anthropic.Anthropic()
        self._max_tokens = max_tokens

    def create(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        response = self._client.messages.create(  # type: ignore[call-overload]
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            output_config={"effort": self._effort},
            # Tools and the system prompt are identical for every lead an agent handles,
            # so the prefix is cached and later requests read it at a fraction of the price.
            cache_control={"type": "ephemeral"},
        )
        calls = [
            ToolCall(id=block.id, name=block.name, input=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelTurn(
            raw_content=response.content,
            text=text,
            tool_calls=calls,
            stop_reason=response.stop_reason or "end_turn",
            usage=Usage(
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.usage.cache_read_input_tokens or 0,
                response.usage.cache_creation_input_tokens or 0,
            ),
        )
