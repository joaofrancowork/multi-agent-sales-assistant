"""sales-agents run | eval"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from sales_agents.llm.base import ChatModel
from sales_agents.orchestrator import Pipeline
from sales_agents.report import write_outputs
from sales_agents.tools.directory import DATA_DIR, Directory
from sales_agents.tracing import Tracer


def _model_factory(offline: bool, model: str) -> type[ChatModel] | object:
    if offline:
        from sales_agents.llm.demo import DemoModel

        return DemoModel
    from sales_agents.llm.anthropic import ClaudeModel  # pragma: no cover

    return lambda: ClaudeModel(model=model)  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sales-agents")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--companies", default="all", help="Comma-separated ids, or 'all'")
        p.add_argument("--data", type=Path, default=DATA_DIR)
        p.add_argument("--out", type=Path, default=Path("out"))
        p.add_argument("--offline", action="store_true", help="Rule-based model, no API key")
        p.add_argument("--model", default=os.getenv("SALES_AGENTS_MODEL", "claude-opus-5-5"))
        p.add_argument(
            "--concurrency", type=int, default=int(os.getenv("SALES_AGENTS_CONCURRENCY", "4"))
        )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    directory = Directory.load(args.data)
    ids = list(directory.companies) if args.companies == "all" else args.companies.split(",")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    tracer = Tracer(args.out / "traces" / f"{run_id}.jsonl")
    factory = _model_factory(args.offline, args.model)
    pipeline = Pipeline(directory, factory, tracer, args.concurrency)  # type: ignore[arg-type]
    results = pipeline.run(ids)

    if args.command == "eval":
        from sales_agents.evaluation import evaluate, to_markdown

        expected = json.loads((args.data / "expected_tiers.json").read_text())
        report = evaluate(results, expected, tracer)
        print(to_markdown(report, model=args.model if not args.offline else "demo-rules"))
        (args.out / "eval.json").parent.mkdir(parents=True, exist_ok=True)
        (args.out / "eval.json").write_text(json.dumps(report, indent=2))
        return 0

    summary = write_outputs(results, args.out, run_id)
    print(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
