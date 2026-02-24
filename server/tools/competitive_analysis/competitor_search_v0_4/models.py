from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CompanyCompetitorCandidate(BaseModel):
    name: str
    cluster: str = "manufacturer"
    year_founded: int = 0
    headquarters_country: str = ""
    company_description: str = ""
    primary_business_segments: List[str] = Field(default_factory=list)
    relevance_in_reference_segment: str = ""
    competitor_type: str = "Direct competitor"
    company_website_url: str = ""
    brand_domain_whitelist: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0


class CompetitorSearchResults(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    generated_queries: List[str] = Field(default_factory=list)
    min_competitors_target: int = 6
    competitors: List[CompanyCompetitorCandidate] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorSearchRequest(BaseModel):
    analysis_plan: Optional[Dict[str, Any]] = None
    analysis_plan_path: Optional[str] = None
    provider: str = "openai"
    max_queries: int = Field(default=20, ge=1, le=40)
    per_query_results: int = Field(default=10, ge=3, le=20)
    shortlist_size: int = Field(default=12, ge=3, le=50)
    max_candidates_to_check: int = Field(default=40, ge=5, le=120)
    min_relevance_score: float = Field(default=0.15, ge=0.0, le=1.0)
    search_timeout_ms: int = Field(default=20000, ge=5000, le=60000)
    verbose_terminal: bool = True

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorSearchRequest":
        if not self.analysis_plan and not (self.analysis_plan_path or "").strip():
            raise ValueError("Either analysis_plan or analysis_plan_path must be provided.")
        return self


class CompetitorSearchResponse(BaseModel):
    competitor_search_results: CompetitorSearchResults


__all__ = [
    "CompanyCompetitorCandidate",
    "CompetitorSearchRequest",
    "CompetitorSearchResponse",
    "CompetitorSearchResults",
]
