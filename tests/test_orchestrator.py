from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sales_agents import cli
from sales_agents.evaluation import evaluate, to_markdown
from sales_agents.llm.demo import DemoModel
from sales_agents.llm.scripted import ScriptedModel, text_turn, tool_turn
from sales_agents.orchestrator import Pipeline
from sales_agents.report import review_queue, summarise
from sales_agents.tools.directory import DATA_DIR, Directory
from sales_agents.tracing import Tracer

DIRECTORY = Directory.load()
EXPECTED = json.loads((DATA_DIR / "expected_tiers.json").read_text())


def _pipeline(factory: Any = DemoModel, concurrency: int = 4) -> tuple[Pipeline, Tracer]:
    tracer = Tracer()
    return Pipeline(DIRECTORY, factory, tracer, concurrency), tracer


def test_offline_run_matches_labels_and_drafts_pass_compliance() -> None:
    pipeline, tracer = _pipeline()
    results = pipeline.run(list(DIRECTORY.companies))
    by_id = {r.company_id: r for r in results}
    assert by_id["c01"].status == "drafted" and by_id["c01"].email
    assert by_id["c04"].status == "disqualified" and by_id["c04"].email is None
    assert all(not r.compliance_issues for r in results if r.email)

    report = evaluate(results, EXPECTED, tracer)
    assert report["tier_accuracy"] == 1.0
    assert report["contact_precision"] == 1.0
    assert report["compliance_pass_rate"] == 1.0
    assert report["grounded_signal_rate"] == 1.0
    assert "Tier accuracy | 100%" in to_markdown(report, "demo-rules")

    assert "7 drafted" in summarise(results) and "5 disqualified" in summarise(results)
    assert "Nothing has been sent" in review_queue(results)


BRIEF = {
    "company_id": "c01",
    "company_name": "Clínica Sorriso Azul",
    "sector": "dental clinic",
    "employees": 14,
    "summary": "Dental clinic in Braga.",
    "pain_points": ["impossible to reach by phone"],
    "signals": [{"name": "no online booking", "evidence": "Book your appointment by phone"}],
}


def _qualify(score: int, tier: str) -> dict[str, Any]:
    return {"company_id": "c01", "score": score, "tier": tier, "reasons": ["x"], "risks": []}


def test_disagreement_with_rubric_goes_to_review() -> None:
    model = ScriptedModel(
        [tool_turn(("submit_research", BRIEF)), tool_turn(("submit_qualify", _qualify(30, "C")))]
    )
    pipeline, _ = _pipeline(lambda: model)
    result = pipeline.process("c01")
    assert result.status == "needs_review"
    assert "agent score 30 vs rubric 90" in (result.error or "")


def test_failing_draft_is_rewritten_once_then_flagged() -> None:
    bad = {
        "subject": "Hello!",
        "body": "We guarantee more bookings. " * 6,
        "personalisation": ["x"],
    }
    model = ScriptedModel(
        [
            tool_turn(("submit_research", BRIEF)),
            tool_turn(("submit_qualify", _qualify(88, "A"))),
            tool_turn(("submit_writer", bad)),
            tool_turn(("submit_writer", bad)),
        ]
    )
    pipeline, tracer = _pipeline(lambda: model)
    result = pipeline.process("c01")
    assert result.status == "needs_review"
    assert any("guarantee" in i for i in result.compliance_issues)
    retry_task = model.requests[3]["messages"][0]["content"]
    assert "failed these checks" in retry_task
    assert result.cost_usd == 0.0 and result.input_tokens == 4 * 500
    assert tracer.of("lead_done")[0]["status"] == "needs_review"


def test_agent_failure_is_isolated_to_one_lead() -> None:
    def factory() -> ScriptedModel:
        return ScriptedModel([text_turn("no"), text_turn("still no")])

    pipeline, _ = _pipeline(factory, concurrency=2)
    results = pipeline.run(["c01", "c02"])
    assert [r.status for r in results] == ["failed", "failed"]
    assert "without calling submit_research" in (results[0].error or "")


def test_cli_run_and_eval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--offline", "--companies", "c01,c04", "--out", str(tmp_path)]) == 0
    assert "2 leads: 1 drafted" in capsys.readouterr().out
    assert list(tmp_path.glob("*-review.md")) and list((tmp_path / "traces").glob("*.jsonl"))

    assert cli.main(["eval", "--offline", "--out", str(tmp_path)]) == 0
    assert "Tier accuracy | 100%" in capsys.readouterr().out
    assert json.loads((tmp_path / "eval.json").read_text())["leads"] == 12
