from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CompanyProfile(BaseModel):
    company_name: str = ""
    industry: str = ""
    core_business: str = ""
    technology_domains: List[str] = Field(default_factory=list)
    innovation_focus: List[str] = Field(default_factory=list)
    strategic_objectives: List[str] = Field(default_factory=list)
    research_queries: List[str] = Field(default_factory=list)
    research_snippets: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep2Request(BaseModel):
    workshop_analysis: Optional[Dict[str, Any]] = None
    workshop_analysis_path: Optional[str] = None
    company_name: str = ""
    provider: str = "ionos"
    max_research_queries: int = Field(default=16, ge=1, le=16)
    max_context_chars: int = Field(default=18000, ge=3000, le=70000)
    brave_enable_research: bool = True
    brave_stream: bool = True
    brave_language: Optional[str] = "de"
    brave_country: Optional[str] = "DE"

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep2Request":
        has_workshop = isinstance(self.workshop_analysis, dict) and bool(self.workshop_analysis)
        has_workshop_path = bool((self.workshop_analysis_path or "").strip())
        has_company = bool((self.company_name or "").strip())
        if not has_workshop and not has_workshop_path and not has_company:
            raise ValueError("Provide company_name or workshop_analysis(_path).")
        return self


class StartupMatchupStep2Response(BaseModel):
    company_profile: CompanyProfile


__all__ = [
    "CompanyProfile",
    "StartupMatchupStep2Request",
    "StartupMatchupStep2Response",
]
