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


class CompetitorEnrichedV06(BaseModel):
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


class CompetitorProfileExtractionResultsV06(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    competitors: List[CompetitorEnrichedV06] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorProfileExtractionV06Request(BaseModel):
    competitor_profile_text: Optional[Dict[str, Any]] = None
    competitor_profile_text_path: Optional[str] = None
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    provider: str = "openai"
    max_competitors: int = Field(default=200, ge=1, le=1000)
    verbose_terminal: bool = False

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorProfileExtractionV06Request":
        if not self.competitor_profile_text and not (self.competitor_profile_text_path or "").strip():
            raise ValueError("Either competitor_profile_text or competitor_profile_text_path must be provided.")
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        return self


class CompetitorProfileExtractionV06Response(BaseModel):
    competitor_profile_results: CompetitorProfileExtractionResultsV06


__all__ = [
    "FeatureValue",
    "PriceIndicatorValue",
    "SoftFeatureValue",
    "ClaimValue",
    "CompetitorEnrichedV06",
    "CompetitorProfileExtractionResultsV06",
    "CompetitorProfileExtractionV06Request",
    "CompetitorProfileExtractionV06Response",
]

