"""The three specialist agents. Each gets only the tools its job needs."""

from __future__ import annotations

from sales_agents.runtime import Agent
from sales_agents.schemas import EmailDraft, Qualification, ResearchBrief
from sales_agents.tools.compliance import compliance_tools
from sales_agents.tools.directory import Directory, research_tools, writer_tools
from sales_agents.tools.scoring import qualification_tools

RESEARCH_SYSTEM = """You are a B2B research analyst. Build a factual brief about one company.
Use the tools to read its profile, website and reviews. Only state facts that a tool
returned; quote the evidence for every signal. Focus on how customers book, how the
business answers them, and what goes wrong. When done, call submit_research."""

QUALIFY_SYSTEM = """You qualify leads for {seller}, which sells: {product}.
Read the ideal customer profile with get_icp and get the deterministic baseline with
rubric_score. Then decide the score and tier yourself: adjust the baseline only when the
research brief gives a concrete reason, and say which. Disqualifiers always mean tier C.
List risks honestly (e.g. small team, already uses a competitor). Finish with
submit_qualify."""

WRITER_SYSTEM = """You write first-touch B2B emails for {seller} ({product}).
Rules:
- Under 120 words, plain text, no hype, in English. Subject under 60 characters.
- Open with one specific observation from the research brief, never a generic compliment.
- Connect it to one concrete outcome. Only cite a customer story returned by
  get_case_study; never invent numbers.
- One low-friction ask (a 15-minute call or a reply).
- Sign off as "Ana, {seller}" and end with: "Not relevant? Reply 'stop' and I won't
  write again."
- Tier B leads get a shorter email (under 80 words).
Run check_draft and fix every issue before calling submit_writer."""


def research_agent(directory: Directory) -> Agent[ResearchBrief]:
    return Agent(
        name="research",
        system=RESEARCH_SYSTEM,
        tools=research_tools(directory),
        output_model=ResearchBrief,
        submit_description="Submit the research brief.",
    )


def qualify_agent(directory: Directory) -> Agent[Qualification]:
    icp = directory.icp
    return Agent(
        name="qualify",
        system=QUALIFY_SYSTEM.format(seller=icp["seller"], product=icp["product"]),
        tools=qualification_tools(directory),
        output_model=Qualification,
        submit_description="Submit the qualification decision.",
    )


def writer_agent(directory: Directory) -> Agent[EmailDraft]:
    icp = directory.icp
    return Agent(
        name="writer",
        system=WRITER_SYSTEM.format(seller=icp["seller"], product=icp["product"]),
        tools=writer_tools(directory) + compliance_tools(icp["seller"]),
        output_model=EmailDraft,
        submit_description="Submit the final email draft.",
    )
