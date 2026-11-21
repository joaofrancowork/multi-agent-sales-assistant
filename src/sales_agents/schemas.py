"""Typed outputs that agents hand to each other.

Agents never pass free text downstream: each one submits a validated object,
which keeps the hand-offs inspectable and the failure modes explicit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["A", "B", "C"]


class Signal(BaseModel):
    name: str = Field(description="Short label, e.g. 'no online booking'")
    evidence: str = Field(description="Quote or fact from a tool result that supports it")


class ResearchBrief(BaseModel):
    company_id: str
    company_name: str
    sector: str
    employees: int | None
    summary: str = Field(description="Two sentences on what the business does and how it books")
    pain_points: list[str] = Field(description="Problems customers or the business mention")
    signals: list[Signal]


class Qualification(BaseModel):
    company_id: str
    score: int = Field(ge=0, le=100)
    tier: Tier
    reasons: list[str] = Field(min_length=1)
    risks: list[str]


class EmailDraft(BaseModel):
    subject: str = Field(min_length=5, max_length=80)
    body: str = Field(min_length=80)
    personalisation: list[str] = Field(
        description="The specific facts about this company the email relies on"
    )


LeadStatus = Literal["drafted", "disqualified", "needs_review", "failed"]


class LeadResult(BaseModel):
    company_id: str
    status: LeadStatus
    brief: ResearchBrief | None = None
    qualification: Qualification | None = None
    rubric_score: int | None = None
    email: EmailDraft | None = None
    compliance_issues: list[str] = Field(default_factory=list)
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
