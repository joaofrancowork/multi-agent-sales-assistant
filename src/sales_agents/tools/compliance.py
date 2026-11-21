"""Checks every outbound email must pass, whatever the model wrote.

Cold B2B email in the EU relies on legitimate interest, so every message has
to identify the sender and offer a one-click way to opt out. The checks also
catch the classic LLM failure modes: unfilled placeholders, invented claims and
emails that are too long to read.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from sales_agents.runtime import Tool
from sales_agents.schemas import EmailDraft

MAX_WORDS = 150
MAX_SUBJECT = 60
BANNED = ("guarantee", "100%", "risk-free", "act now", "limited time", "click here")
OPT_OUT = re.compile(r"\b(unsubscribe|opt out|not interested|reply ['\"]?stop)", re.IGNORECASE)
PLACEHOLDER = re.compile(r"\{\{?[^}]*\}?\}|\[(?:first ?name|company|name)\]", re.IGNORECASE)


def check_email(draft: EmailDraft, company_name: str, sender: str) -> list[str]:
    issues: list[str] = []
    words = len(draft.body.split())
    if words > MAX_WORDS:
        issues.append(f"body has {words} words; keep it under {MAX_WORDS}")
    if len(draft.subject) > MAX_SUBJECT:
        issues.append(f"subject has {len(draft.subject)} characters; max {MAX_SUBJECT}")
    text = f"{draft.subject}\n{draft.body}"
    if PLACEHOLDER.search(text):
        issues.append("contains an unfilled placeholder")
    for phrase in BANNED:
        if phrase in text.lower():
            issues.append(f"uses the phrase {phrase!r}")
    if company_name.lower() not in text.lower():
        issues.append("does not mention the company by name")
    if sender.lower() not in draft.body.lower():
        issues.append(f"does not identify the sender ({sender})")
    if not OPT_OUT.search(draft.body):
        issues.append("missing an opt-out line")
    if not draft.personalisation:
        issues.append("no company-specific personalisation listed")
    return issues


class DraftToCheck(BaseModel):
    company_name: str
    subject: str
    body: str


def compliance_tools(sender: str) -> list[Tool[DraftToCheck]]:
    def check(args: DraftToCheck) -> dict[str, object]:
        draft = EmailDraft.model_construct(
            subject=args.subject, body=args.body, personalisation=["-"]
        )
        issues = check_email(draft, args.company_name, sender)
        return {"passes": not issues, "issues": issues}

    return [
        Tool(
            "check_draft",
            "Run the compliance checks on a draft before submitting it.",
            DraftToCheck,
            check,
        )
    ]
