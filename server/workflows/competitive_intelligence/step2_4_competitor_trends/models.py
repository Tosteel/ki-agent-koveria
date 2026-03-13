from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class StructuredSection(BaseModel):
    summary: str = ""
    source_urls: List[str] = Field(default_factory=list)


class RatingsReachSection(StructuredSection):
    google_rating: Optional[float] = None
    google_review_count: Optional[int] = None
    social_reach: str = ""


class TrendMatchItem(BaseModel):
    trend_summary: str
    trend_source_urls: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    matched: bool = False
    match_score: float = 0.0
    matched_keywords: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class Step24CompetitorTrendProfile(BaseModel):
    company: str
    website: str = ""
    region: str = ""
    company_profile_target_audience: StructuredSection = Field(default_factory=StructuredSection)
    offers_actions: StructuredSection = Field(default_factory=StructuredSection)
    ratings_reach: RatingsReachSection = Field(default_factory=RatingsReachSection)
    press_coverage: StructuredSection = Field(default_factory=StructuredSection)
    trend_matches: List[TrendMatchItem] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step24CompetitorTrendsResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step2.4_competitor_trends.json"
    profiles: List[Step24CompetitorTrendProfile] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step24CompetitorTrendsRequest(BaseModel):
    competitor_profile_structured: Optional[Dict[str, Any]] = None
    competitor_profile_structured_path: Optional[str] = None
    market_trends_summary: Optional[Dict[str, Any]] = None
    market_trends_summary_path: Optional[str] = None
    provider: str = "ionos"
    max_companies: int = Field(default=120, ge=1, le=1000)
    max_trends: int = Field(default=40, ge=1, le=200)
    keywords_per_trend: int = Field(default=8, ge=2, le=20)
    min_keyword_len: int = Field(default=4, ge=3, le=20)
    min_keyword_hits: int = Field(default=1, ge=1, le=10)
    website_timeout_ms: int = Field(default=20000, ge=2000, le=120000)
    website_max_chars: int = Field(default=120000, ge=5000, le=400000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step24CompetitorTrendsRequest":
        has_profiles_inline = isinstance(self.competitor_profile_structured, dict) and bool(self.competitor_profile_structured)
        has_profiles_path = bool((self.competitor_profile_structured_path or "").strip())
        has_trends_inline = isinstance(self.market_trends_summary, dict) and bool(self.market_trends_summary)
        has_trends_path = bool((self.market_trends_summary_path or "").strip())
        if not has_profiles_inline and not has_profiles_path:
            raise ValueError("Either competitor_profile_structured or competitor_profile_structured_path must be provided.")
        if not has_trends_inline and not has_trends_path:
            raise ValueError("Either market_trends_summary or market_trends_summary_path must be provided.")
        return self


class Step24CompetitorTrendsResponse(BaseModel):
    competitor_trends: Step24CompetitorTrendsResult


__all__ = [
    "RatingsReachSection",
    "Step24CompetitorTrendProfile",
    "Step24CompetitorTrendsRequest",
    "Step24CompetitorTrendsResponse",
    "Step24CompetitorTrendsResult",
    "StructuredSection",
    "TrendMatchItem",
]
