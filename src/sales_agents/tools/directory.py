"""Read-only access to the company dataset.

In production these tools would wrap a business registry, a web fetcher and a
reviews API. The sample dataset keeps the project runnable and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sales_agents.runtime import Tool, ToolError

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


@dataclass
class Directory:
    companies: dict[str, dict[str, Any]]
    icp: dict[str, Any]
    case_studies: dict[str, str]

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> Directory:
        companies = json.loads((data_dir / "companies.json").read_text(encoding="utf-8"))
        return cls(
            companies={c["id"]: c for c in companies},
            icp=json.loads((data_dir / "icp.json").read_text(encoding="utf-8")),
            case_studies=json.loads((data_dir / "case_studies.json").read_text(encoding="utf-8")),
        )

    def get(self, company_id: str) -> dict[str, Any]:
        try:
            return self.companies[company_id]
        except KeyError:
            raise ToolError(
                f"No company with id {company_id!r}. Use search_directory to find ids."
            ) from None


class CompanyId(BaseModel):
    company_id: str = Field(description="Company id, e.g. 'c01'")


class Search(BaseModel):
    query: str = Field(description="Words to match against name, sector or city")


class Sector(BaseModel):
    sector: str


class NoInput(BaseModel):
    pass


def research_tools(directory: Directory) -> list[Tool[Any]]:
    def search(args: Search) -> list[dict[str, str]]:
        words = args.query.lower().split()
        hits = [
            {"id": c["id"], "name": c["name"], "sector": c["sector"], "city": c["city"]}
            for c in directory.companies.values()
            if all(w in f"{c['name']} {c['sector']} {c['city']}".lower() for w in words)
        ]
        return hits[:10]

    def profile(args: CompanyId) -> dict[str, Any]:
        c = directory.get(args.company_id)
        return {k: c[k] for k in ("id", "name", "sector", "city", "employees", "tech")}

    def website(args: CompanyId) -> str:
        return str(directory.get(args.company_id)["website"])

    def reviews(args: CompanyId) -> dict[str, Any]:
        return dict(directory.get(args.company_id)["reviews"])

    return [
        Tool("search_directory", "Find companies by name, sector or city.", Search, search),
        Tool(
            "get_company_profile",
            "Basic facts: sector, city, employees and the booking tools the company uses.",
            CompanyId,
            profile,
        ),
        Tool("read_website", "Text of the company's website.", CompanyId, website),
        Tool(
            "get_reviews",
            "Public review rating, count and recent review snippets.",
            CompanyId,
            reviews,
        ),
    ]


def writer_tools(directory: Directory) -> list[Tool[Any]]:
    def case_study(args: Sector) -> str:
        story = directory.case_studies.get(args.sector)
        if story is None:
            raise ToolError(f"No case study for {args.sector!r}. Write without one.")
        return story

    return [
        Tool(
            "get_case_study",
            "A short, approved customer story for a sector. Only cite stories returned here.",
            Sector,
            case_study,
        )
    ]
