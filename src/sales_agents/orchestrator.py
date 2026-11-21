"""Code-controlled orchestration of the three agents.

The control flow is fixed and lives in code, not in a "manager" LLM: research,
then qualify, then (only for tiers A and B) write. That keeps cost predictable,
makes each step testable, and puts the guardrails where a model cannot skip them:

* rubric disagreement -> ``needs_review``;
* compliance failures after one rewrite -> ``needs_review``;
* any agent error -> ``failed`` for that lead only; the batch keeps going.

Nothing is ever sent: approved drafts are exported for a human to send.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sales_agents.agents import qualify_agent, research_agent, writer_agent
from sales_agents.llm.base import ChatModel, Usage
from sales_agents.runtime import AgentError
from sales_agents.schemas import EmailDraft, LeadResult
from sales_agents.tools.compliance import check_email
from sales_agents.tools.directory import Directory
from sales_agents.tools.scoring import MAX_DISAGREEMENT, rubric
from sales_agents.tracing import Tracer

log = logging.getLogger(__name__)


@dataclass
class Pipeline:
    directory: Directory
    model_factory: Callable[[], ChatModel]
    tracer: Tracer
    concurrency: int = 4

    def run(self, company_ids: list[str]) -> list[LeadResult]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self.process, company_ids))

    def process(self, company_id: str) -> LeadResult:
        model = self.model_factory()
        usage = Usage()
        result = LeadResult(company_id=company_id, status="failed")
        try:
            research = research_agent(self.directory).run(
                f"Research company {company_id}.", model, self.tracer, company_id
            )
            usage += research.usage
            result.brief = research.output

            qualify = qualify_agent(self.directory).run(
                "Qualify this lead.\n\nResearch brief:\n"
                + research.output.model_dump_json(indent=2),
                model,
                self.tracer,
                company_id,
            )
            usage += qualify.usage
            result.qualification = qualify.output
            company = self.directory.get(company_id)
            result.rubric_score, _ = rubric(company, self.directory.icp)

            if abs(qualify.output.score - result.rubric_score) > MAX_DISAGREEMENT:
                result.status = "needs_review"
                result.error = f"agent score {qualify.output.score} vs rubric {result.rubric_score}"
            elif qualify.output.tier == "C":
                result.status = "disqualified"
            else:
                email, usage = self._write(company_id, company["name"], result, model, usage)
                result.email = email
                result.compliance_issues = check_email(email, company["name"], self._sender)
                result.status = "needs_review" if result.compliance_issues else "drafted"
        except AgentError as exc:
            log.warning("lead %s failed: %s", company_id, exc)
            result.error = str(exc)
        result.input_tokens, result.output_tokens = usage.input_tokens, usage.output_tokens
        result.cost_usd = round(usage.cost_usd(model.model), 4)
        self.tracer.emit(
            "lead_done", lead_id=company_id, status=result.status, cost=result.cost_usd
        )
        return result

    @property
    def _sender(self) -> str:
        return str(self.directory.icp["seller"])

    def _write(
        self, company_id: str, name: str, result: LeadResult, model: ChatModel, usage: Usage
    ) -> tuple[EmailDraft, Usage]:
        assert result.brief and result.qualification
        task = (
            f"Write the email for {name} (tier {result.qualification.tier}).\n\n"
            f"Research brief:\n{result.brief.model_dump_json(indent=2)}\n\n"
            f"Qualification:\n{result.qualification.model_dump_json(indent=2)}"
        )
        draft = writer_agent(self.directory).run(task, model, self.tracer, company_id)
        usage += draft.usage
        issues = check_email(draft.output, name, self._sender)
        if not issues:
            return draft.output, usage
        # One rewrite with the exact issues, then a human decides.
        retry = writer_agent(self.directory).run(
            f"{task}\n\nYour previous draft failed these checks: {json.dumps(issues)}.\n"
            f"Previous draft:\n{draft.output.model_dump_json(indent=2)}",
            model,
            self.tracer,
            company_id,
        )
        return retry.output, usage + retry.usage
