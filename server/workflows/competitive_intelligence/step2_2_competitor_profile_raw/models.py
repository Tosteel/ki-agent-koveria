from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Step22CompanyInput(BaseModel):
    company: str
    website: str = ""
    region: str = ""

    @field_validator("company")
    @classmethod
    def _validate_company(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("company must not be empty.")
        return text


class Step22RawSearchItem(BaseModel):
    query_id: str
    topic: str
    query: str
    raw_text: str = ""
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    warning: str = ""


class Step22CompanyRawProfile(BaseModel):
    company: str
    website: str = ""
    region: str = ""
    raw_searches: List[Step22RawSearchItem] = Field(default_factory=list)


class Step22CompetitorProfileRawResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "brave"
    output_file: str = "step2.2_competitor_profile_raw.json"
    companies: List[Step22CompanyRawProfile] = Field(default_factory=list)
    trend_context: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step22CompetitorProfileRawRequest(BaseModel):
    companies: Optional[List[Step22CompanyInput]] = None
    companies_path: Optional[str] = None
    market_trends_summary: Optional[Dict[str, Any]] = None
    market_trends_summary_path: Optional[str] = None
    provider: str = "brave"
    max_companies: int = Field(default=40, ge=1, le=500)
    brave_stream: bool = True
    brave_language: Optional[str] = "de"
    brave_country: Optional[str] = "DE"
    brave_enable_research: bool = False
    brave_enable_citations: bool = True
    brave_enable_entities: bool = True
    timeout_s: int = Field(default=90, ge=10, le=300)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step22CompetitorProfileRawRequest":
        has_companies = isinstance(self.companies, list) and len(self.companies) > 0
        has_companies_path = bool((self.companies_path or "").strip())
        if not has_companies and not has_companies_path:
            raise ValueError("Either companies or companies_path must be provided.")
        return self


class Step22CompetitorProfileRawResponse(BaseModel):
    competitor_profile_raw: Step22CompetitorProfileRawResult


__all__ = [
    "Step22CompanyInput",
    "Step22CompanyRawProfile",
    "Step22CompetitorProfileRawRequest",
    "Step22CompetitorProfileRawResponse",
    "Step22CompetitorProfileRawResult",
    "Step22RawSearchItem",
]
