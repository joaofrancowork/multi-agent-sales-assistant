"""Structured traces: one JSON line per model call, tool call and agent outcome.

Traces answer the questions you have when an agent misbehaves in production:
which tool it called, with which input, what came back, how long it took and
what it cost.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

MAX_FIELD_CHARS = 2000


def _truncate(value: Any) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "…"


class Tracer:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "event": event}
        record.update(
            {k: _truncate(v) if k in {"input", "output"} else v for k, v in fields.items()}
        )
        with self._lock:
            self.events.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")

    def of(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == event]
