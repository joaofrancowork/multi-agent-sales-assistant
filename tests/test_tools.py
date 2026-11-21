from __future__ import annotations

import json

import pytest

from sales_agents.runtime import ToolError
from sales_agents.schemas import EmailDraft
from sales_agents.tools.compliance import check_email, compliance_tools
from sales_agents.tools.directory import (
    CompanyId,
    Directory,
    NoInput,
    Search,
    Sector,
    research_tools,
    writer_tools,
)
from sales_agents.tools.scoring import qualification_tools, rubric, tier_for


@pytest.fixture(scope="module")
def directory() -> Directory:
    return Directory.load()


def test_expected_tiers_match_the_rubric(directory: Directory) -> None:
    """The labels and the rubric encode the same ICP; this guards both."""
    expected = json.loads(
        (Directory.load.__globals__["DATA_DIR"] / "expected_tiers.json").read_text()
    )
    for cid, tier in expected.items():
        score, _ = rubric(directory.get(cid), directory.icp)
        assert tier_for(score) == tier, cid


def test_rubric_breakdown_and_disqualifier(directory: Directory) -> None:
    score, why = rubric(directory.get("c01"), directory.icp)
    assert score == 90 and "no online booking" in why
    score, why = rubric(directory.get("c09"), directory.icp)
    assert score == 15 and "disqualifier" in why[0]
    assert [tier_for(s) for s in (70, 69, 50, 49)] == ["A", "B", "B", "C"]


def test_research_tools(directory: Directory) -> None:
    tools = {t.name: t for t in research_tools(directory)}
    hits = tools["search_directory"].fn(Search(query="hvac porto"))
    assert [h["id"] for h in hits] == ["c03"]
    profile = tools["get_company_profile"].fn(CompanyId(company_id="c02"))
    assert profile["tech"]["whatsapp"] is True and "reviews" not in profile
    assert "Instagram" in tools["read_website"].fn(CompanyId(company_id="c02"))
    assert tools["get_reviews"].fn(CompanyId(company_id="c02"))["count"] == 389
    with pytest.raises(ToolError, match="search_directory"):
        tools["read_website"].fn(CompanyId(company_id="zzz"))


def test_writer_and_qualification_tools(directory: Directory) -> None:
    case = writer_tools(directory)[0]
    assert "Guimarães" in case.fn(Sector(sector="dental clinic"))
    with pytest.raises(ToolError):
        case.fn(Sector(sector="restaurant"))
    q = {t.name: t for t in qualification_tools(directory)}
    assert q["get_icp"].fn(NoInput())["seller"] == "Northwind Labs"
    assert q["rubric_score"].fn(CompanyId(company_id="c12"))["suggested_tier"] == "B"


GOOD_BODY = (
    "Hi Lumi team, a review mentions waiting two days for an Instagram reply. "
    "Our assistant answers in under a minute. Worth a 15-minute call?\n\n"
    "Ana, Northwind Labs\nNot relevant? Reply 'stop' and I won't write again."
)


def _draft(
    subject: str = "Instagram bookings at Estética Lumi", body: str = GOOD_BODY
) -> EmailDraft:
    return EmailDraft(subject=subject, body=body, personalisation=["Instagram delay"])


def test_compliance_passes_a_good_draft() -> None:
    assert check_email(_draft(), "Estética Lumi", "Northwind Labs") == []


@pytest.mark.parametrize(
    ("draft", "issue"),
    [
        (_draft(body=GOOD_BODY + " word" * 150), "words"),
        (_draft(subject="A" * 50 + " Estética Lumi"), "subject has"),
        (_draft(body=GOOD_BODY.replace("Lumi team", "{first_name}")), "placeholder"),
        (_draft(body=GOOD_BODY + " We guarantee results."), "guarantee"),
        (_draft(subject="Hello there", body=GOOD_BODY.replace("Lumi", "you")), "by name"),
        (_draft(body=GOOD_BODY.replace("Ana, Northwind Labs", "Ana")), "sender"),
        (_draft(body=GOOD_BODY.split("\nNot relevant")[0] + " Thanks for reading."), "opt-out"),
    ],
)
def test_compliance_catches_issues(draft: EmailDraft, issue: str) -> None:
    issues = check_email(draft, "Estética Lumi", "Northwind Labs")
    assert any(issue in i for i in issues), issues


def test_compliance_requires_personalisation_and_tool_wrapper() -> None:
    draft = EmailDraft(
        subject="Instagram bookings at Estética Lumi", body=GOOD_BODY, personalisation=[]
    )
    assert "no company-specific personalisation listed" in check_email(
        draft, "Estética Lumi", "Northwind Labs"
    )
    tool = compliance_tools("Northwind Labs")[0]
    ok = tool.fn(
        tool.input_model(company_name="Estética Lumi", subject=draft.subject, body=GOOD_BODY)
    )
    assert ok == {"passes": True, "issues": []}
