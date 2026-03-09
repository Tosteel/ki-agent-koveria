from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class FeatureValue(BaseModel):
    name: str
    value: Optional[Any] = None
    unit: str = ""


class SoftFeatureValue(BaseModel):
    name: str
    available: bool = False


class ClaimValue(BaseModel):
    text: str
    claim_type: str = "value"
    evidence: str = ""


class PriceIndicatorValue(BaseModel):
    raw: str = ""
    value: Optional[float] = None
    currency: str = ""
    period: str = ""
    context: str = ""


class ProductCompetitorCandidate(BaseModel):
    product_name: str
    manufacturer: str
    url: str
    url_type: str = "unknown"
    performance_parameters: List[FeatureValue] = Field(default_factory=list)
    price_indicators: List[PriceIndicatorValue] = Field(default_factory=list)
    soft_features: List[SoftFeatureValue] = Field(default_factory=list)
    claims: List[ClaimValue] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)
    enrichment_delta: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0
    similarity_score: float = 0.0


class CompetitorSearchResultsV04(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    generated_queries: List[str] = Field(default_factory=list)
    competitors: List[ProductCompetitorCandidate] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorSearchV04Request(BaseModel):
    analysis_plan: Optional[Dict[str, Any]] = None
    analysis_plan_path: Optional[str] = None
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    provider: str = "openai"
    max_queries: int = Field(default=20, ge=1, le=50)
    per_query_results: int = Field(default=10, ge=3, le=20)
    max_candidates_to_check: int = Field(default=200, ge=10, le=1000)
    use_llm_feature_enrichment: bool = False
    llm_min_relevance_for_enrichment: float = Field(default=0.2, ge=0.0, le=1.0)
    include_page_fetch: bool = False
    page_fetch_timeout_s: int = Field(default=8, ge=2, le=30)
    page_fetch_max_chars: int = Field(default=6000, ge=1000, le=30000)
    verbose_terminal: bool = False
    verbose_search_hits: bool = False

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorSearchV04Request":
        if not self.analysis_plan and not (self.analysis_plan_path or "").strip():
            raise ValueError("Either analysis_plan or analysis_plan_path must be provided.")
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        return self


class CompetitorSearchV04Response(BaseModel):
    competitor_search_results: CompetitorSearchResultsV04


__all__ = [
    "FeatureValue",
    "SoftFeatureValue",
    "ClaimValue",
    "PriceIndicatorValue",
    "ProductCompetitorCandidate",
    "CompetitorSearchResultsV04",
    "CompetitorSearchV04Request",
    "CompetitorSearchV04Response",
]
