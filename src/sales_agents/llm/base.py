"""Provider-neutral view of one model turn.

The agent runtime only depends on this module, so it can be tested with a
scripted model and run against Claude in production without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# USD per million tokens: (input, output, cache read). Cache writes cost 1.25x input.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    """Token counts as the API reports them; ``input_tokens`` excludes cached tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def cache_hit_rate(self) -> float:
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0

    def cost_usd(self, model: str) -> float:
        price_in, price_out, price_cached = PRICES.get(model, (0.0, 0.0, 0.0))
        return (
            self.input_tokens * price_in
            + self.cache_write_tokens * price_in * CACHE_WRITE_MULTIPLIER
            + self.cache_read_tokens * price_cached
            + self.output_tokens * price_out
        ) / 1_000_000


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ModelTurn:
    """What the runtime needs from one model response.

    ``raw_content`` is sent back verbatim on the next request: it may contain
    thinking blocks that the API requires to be returned unchanged.
    """

    raw_content: Any
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


class ChatModel(Protocol):
    model: str

    def create(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn: ...
