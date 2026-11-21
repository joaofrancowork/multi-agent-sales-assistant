"""Write run outputs: full JSON, a review queue in Markdown, and a summary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sales_agents.schemas import LeadResult


def summarise(results: list[LeadResult]) -> str:
    counts = Counter(r.status for r in results)
    cost = sum(r.cost_usd for r in results)
    tokens = sum(r.input_tokens + r.output_tokens for r in results)
    parts = [f"{counts[s]} {s}" for s in ("drafted", "needs_review", "disqualified", "failed")]
    return f"{len(results)} leads: " + ", ".join(parts) + f" · {tokens:,} tokens · ${cost:.3f}"


def review_queue(results: list[LeadResult]) -> str:
    lines = ["# Drafts for review", "", "Nothing has been sent. Approve and send manually.", ""]
    for r in sorted(results, key=lambda r: (r.status != "drafted", r.company_id)):
        name = r.brief.company_name if r.brief else r.company_id
        q = r.qualification
        lines.append(f"## {name} · {r.status}")
        if q:
            lines.append(
                f"Tier {q.tier} · score {q.score} (rubric {r.rubric_score}) · "
                + "; ".join(q.reasons)
            )
        if r.error:
            lines.append(f"> {r.error}")
        for issue in r.compliance_issues:
            lines.append(f"- ⚠️ {issue}")
        if r.email:
            lines += ["", f"**Subject:** {r.email.subject}", "", "```text", r.email.body, "```"]
        lines.append("")
    return "\n".join(lines)


def write_outputs(results: list[LeadResult], out: Path, run_id: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}-results.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / f"{run_id}-review.md").write_text(review_queue(results), encoding="utf-8")
    return summarise(results)
