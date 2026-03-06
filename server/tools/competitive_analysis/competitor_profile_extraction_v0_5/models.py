from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class FeatureValue(BaseModel):
    name: str
    value: Optional[float] = None
    unit: str = ""


class PriceIndicatorValue(BaseModel):
    raw: str = ""
    value: Optional[float] = None
    currency: str = ""
    period: str = ""
    context: str = ""


class SoftFeatureValue(BaseModel):
    name: str
    available: bool = False


class ClaimValue(BaseModel):
    text: str
    claim_type: str = "value"
    evidence: str = ""


class CompetitorEnrichedV05(BaseModel):
    product_name: str
    manufacturer: str
    url: str
    url_type: str = "unknown"
    performance_parameters: List[FeatureValue] = Field(default_factory=list)
    price_indicators: List[PriceIndicatorValue] = Field(default_factory=list)
    soft_features: List[SoftFeatureValue] = Field(default_factory=list)
    claims: List[ClaimValue] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    similarity_score: float = 0.0


class CompetitorProfileExtractionResultsV05(BaseModel):
    schema_version: str = "1.0"
    provider: str = "brave"
    competitors: List[CompetitorEnrichedV05] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorProfileExtractionV05Request(BaseModel):
    competitor_search_results: Optional[Dict[str, Any]] = None
    competitor_search_results_path: Optional[str] = None
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    provider: str = "brave"
    max_competitors: int = Field(default=200, ge=1, le=1000)
    include_page_fetch: bool = True
    page_fetch_timeout_s: int = Field(default=8, ge=2, le=30)
    page_fetch_max_chars: int = Field(default=8000, ge=1000, le=40000)
    verbose_terminal: bool = False

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorProfileExtractionV05Request":
        if not self.competitor_search_results and not (self.competitor_search_results_path or "").strip():
            raise ValueError("Either competitor_search_results or competitor_search_results_path must be provided.")
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        return self


class CompetitorProfileExtractionV05Response(BaseModel):
    competitor_profile_results: CompetitorProfileExtractionResultsV05


__all__ = [
    "FeatureValue",
    "PriceIndicatorValue",
    "SoftFeatureValue",
    "ClaimValue",
    "CompetitorEnrichedV05",
    "CompetitorProfileExtractionResultsV05",
    "CompetitorProfileExtractionV05Request",
    "CompetitorProfileExtractionV05Response",
]
