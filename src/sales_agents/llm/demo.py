"""A rule-based stand-in for the LLM, for ``--offline`` runs and end-to-end tests.

It is *not* intelligent: it calls each agent's lookup tools, then fills the submit
schema from the tool results with simple rules. Its job is to exercise the real
runtime, tools, guardrails and orchestration with no API key, and to provide a
baseline the real model has to beat in the evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sales_agents.llm.base import ModelTurn, Usage
from sales_agents.llm.scripted import tool_turn
from sales_agents.tools.scoring import COMPLAINT_WORDS


def _tool_results(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Map tool name -> parsed result for the latest round of tool calls."""
    calls = {b["id"]: b["name"] for b in messages[-2]["content"] if b.get("type") == "tool_use"}
    out: dict[str, Any] = {}
    for block in messages[-1]["content"]:
        content = block["content"]
        try:
            out[calls[block["tool_use_id"]]] = json.loads(content)
        except json.JSONDecodeError:
            out[calls[block["tool_use_id"]]] = content
    return out


def _task(messages: list[dict[str, Any]]) -> str:
    return str(messages[0]["content"])


class DemoModel:
    model = "demo-rules"

    def create(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        submit = next(t["name"] for t in tools if t["name"].startswith("submit_"))
        first_turn = len(messages) == 1
        if submit == "submit_research":
            turn = self._research(messages, first_turn)
        elif submit == "submit_qualify":
            turn = self._qualify(messages, first_turn)
        else:
            turn = self._write(messages, first_turn)
        turn.usage = Usage()  # no model, no tokens
        return turn

    def _research(self, messages: list[dict[str, Any]], first: bool) -> ModelTurn:
        cid = re.search(r"company (\w+)", _task(messages))
        company_id = cid.group(1) if cid else ""
        if first:
            arg = {"company_id": company_id}
            return tool_turn(
                ("get_company_profile", arg), ("read_website", arg), ("get_reviews", arg)
            )
        r = _tool_results(messages)
        profile, reviews, site = r["get_company_profile"], r["get_reviews"], r["read_website"]
        pains = [s for s in reviews["snippets"] if any(w in s.lower() for w in COMPLAINT_WORDS)]
        signals = []
        if not profile["tech"]["online_booking"]:
            signals.append({"name": "no online booking", "evidence": site})
        if profile["tech"]["reminders"] != "automatic":
            signals.append(
                {
                    "name": "no automatic reminders",
                    "evidence": f"reminders: {profile['tech']['reminders']}",
                }
            )
        signals += [{"name": "contact friction in reviews", "evidence": p} for p in pains[:2]]
        brief = {
            "company_id": profile["id"],
            "company_name": profile["name"],
            "sector": profile["sector"],
            "employees": profile["employees"],
            "summary": f"{profile['name']} is a {profile['sector']} in {profile['city']}. {site}",
            "pain_points": pains,
            "signals": signals,
        }
        return tool_turn(("submit_research", brief))

    def _qualify(self, messages: list[dict[str, Any]], first: bool) -> ModelTurn:
        brief = json.loads(_task(messages).split("Research brief:\n", 1)[1])
        if first:
            return tool_turn(("get_icp", {}), ("rubric_score", {"company_id": brief["company_id"]}))
        score = _tool_results(messages)["rubric_score"]
        return tool_turn(
            (
                "submit_qualify",
                {
                    "company_id": brief["company_id"],
                    "score": score["score"],
                    "tier": score["suggested_tier"],
                    "reasons": score["breakdown"] or ["no ICP signals"],
                    "risks": []
                    if brief["employees"] and brief["employees"] >= 10
                    else ["small team"],
                },
            )
        )

    def _write(self, messages: list[dict[str, Any]], first: bool) -> ModelTurn:
        task = _task(messages)
        brief = json.loads(task.split("Research brief:\n", 1)[1].split("\n\nQualification:")[0])
        if first:
            return tool_turn(("get_case_study", {"sector": brief["sector"]}))
        story = _tool_results(messages).get("get_case_study", "")
        name = brief["company_name"]
        hook = (
            brief["pain_points"][0] if brief["pain_points"] else "you still take bookings by hand"
        )
        body = (
            f'Hi {name} team,\n\nOne of your reviews says: "{hook}" That usually means '
            f"bookings are being lost when the phone or inbox is busy.\n\n"
            f"{story if isinstance(story, str) else ''}\n\n"
            "Would a 15-minute call next week be useful?\n\n"
            "Ana, Northwind Labs\n"
            "Not relevant? Reply 'stop' and I won't write again."
        )
        draft = {
            "subject": f"Bookings at {name}"[:60],
            "body": body,
            "personalisation": [hook],
        }
        return tool_turn(("submit_writer", draft))
