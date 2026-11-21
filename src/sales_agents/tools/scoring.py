"""Deterministic ICP rubric.

The qualification agent sees this score as one input among others. The
orchestrator also uses it as a guardrail: when the agent's score and the rubric
disagree by more than ``MAX_DISAGREEMENT`` points, the lead goes to a human.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from sales_agents.runtime import Tool
from sales_agents.tools.directory import CompanyId, Directory, NoInput

MAX_DISAGREEMENT = 25
COMPLAINT_WORDS = ("phone", "call", "reach", "answer", "confirm", "record", "back")


def rubric(company: dict[str, Any], icp: dict[str, Any]) -> tuple[int, list[str]]:
    points: list[tuple[int, str]] = []
    employees = company["employees"]
    if company["sector"] in icp["target_sectors"]:
        points.append((30, "target sector"))
    if employees < icp["employees_min"]:
        return 15, ["solo or micro business (disqualifier)"]
    if employees <= icp["employees_max"]:
        points.append((15 if employees >= 10 else 5, f"{employees} employees"))
    tech = company["tech"]
    if not tech["online_booking"]:
        points.append((20, "no online booking"))
    if tech["reminders"] in ("none", "manual"):
        points.append((10, f"reminders: {tech['reminders']}"))
    snippets = " ".join(company["reviews"]["snippets"]).lower()
    if any(word in snippets for word in COMPLAINT_WORDS):
        points.append((15, "reviews mention booking or contact problems"))
    return min(100, sum(p for p, _ in points)), [why for _, why in points]


def tier_for(score: int) -> str:
    return "A" if score >= 70 else "B" if score >= 50 else "C"


class ScoreResult(BaseModel):
    score: int
    suggested_tier: str
    breakdown: list[str]


def qualification_tools(directory: Directory) -> list[Tool[Any]]:
    def get_icp(_: NoInput) -> dict[str, Any]:
        return directory.icp

    def rubric_score(args: CompanyId) -> ScoreResult:
        score, breakdown = rubric(directory.get(args.company_id), directory.icp)
        return ScoreResult(score=score, suggested_tier=tier_for(score), breakdown=breakdown)

    return [
        Tool("get_icp", "The ideal customer profile and tier rules.", NoInput, get_icp),
        Tool(
            "rubric_score",
            "Deterministic ICP score (0-100) with a breakdown. Use it as a baseline.",
            CompanyId,
            lambda a: rubric_score(a).model_dump(),
        ),
    ]
