"""Evaluate a run against hand-labelled tiers.

Metrics:
* tier accuracy and a confusion matrix (A/B/C);
* contact precision: of the leads we would email, how many should be emailed;
* compliance pass rate of the drafts;
* grounding: share of brief signals whose evidence appears in a tool output
  (catches invented facts);
* cost and tool calls per lead.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sales_agents.schemas import LeadResult
from sales_agents.tracing import Tracer


def evaluate(results: list[LeadResult], expected: dict[str, str], tracer: Tracer) -> dict[str, Any]:
    scored = [r for r in results if r.qualification and r.company_id in expected]
    confusion: Counter[str] = Counter()
    for r in scored:
        assert r.qualification
        confusion[f"{expected[r.company_id]}->{r.qualification.tier}"] += 1
    correct = sum(n for k, n in confusion.items() if k[0] == k[-1])
    contacted = [r for r in scored if r.qualification and r.qualification.tier != "C"]
    should = [r for r in contacted if expected[r.company_id] != "C"]
    drafts = [r for r in results if r.email]

    tool_outputs = " ".join(str(e.get("output", "")) for e in tracer.of("tool_call")).lower()
    evidence = [s.evidence for r in results if r.brief for s in r.brief.signals]
    grounded = [e for e in evidence if e.lower()[:60] in tool_outputs]

    n = len(results) or 1
    return {
        "leads": len(results),
        "failed": sum(r.status == "failed" for r in results),
        "tier_accuracy": correct / len(scored) if scored else 0.0,
        "confusion": dict(sorted(confusion.items())),
        "contact_precision": len(should) / len(contacted) if contacted else 0.0,
        "compliance_pass_rate": (
            sum(not r.compliance_issues for r in drafts) / len(drafts) if drafts else 0.0
        ),
        "grounded_signal_rate": len(grounded) / len(evidence) if evidence else 0.0,
        "tool_calls_per_lead": len(tracer.of("tool_call")) / n,
        "cost_per_lead_usd": sum(r.cost_usd for r in results) / n,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def to_markdown(report: dict[str, Any], model: str) -> str:
    return "\n".join(
        [
            f"# Evaluation · `{model}` · {report['leads']} leads",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Tier accuracy | {_pct(report['tier_accuracy'])} |",
            f"| Contact precision | {_pct(report['contact_precision'])} |",
            f"| Compliance pass rate | {_pct(report['compliance_pass_rate'])} |",
            f"| Grounded signals | {_pct(report['grounded_signal_rate'])} |",
            f"| Failed leads | {report['failed']} |",
            f"| Tool calls per lead | {report['tool_calls_per_lead']:.1f} |",
            f"| Cost per lead | ${report['cost_per_lead_usd']:.4f} |",
            "",
            "Confusion (expected->predicted): "
            + ", ".join(f"{k}: {v}" for k, v in report["confusion"].items()),
        ]
    )
