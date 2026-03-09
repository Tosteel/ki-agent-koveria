from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class NormalizedFeature(BaseModel):
    name: str
    value: float | int | str
    unit: str = ""
    normalized_value: float | int | str | None = None
    normalized_unit: str = ""
    source: str = ""


class ClaimItem(BaseModel):
    text: str
    claim_type: str = "benefit"
    evidence: str = ""


class PriceIndicator(BaseModel):
    raw: str
    value: float | int | None = None
    currency: str = ""
    period: str = ""
    context: str = ""


class SoftFeature(BaseModel):
    name: str
    available: bool = True
    source: str = ""


class ExtractionQualityReport(BaseModel):
    normalized_features_count: int = 0
    performance_parameters_count: int = 0
    metric_features_count: int = 0
    price_indicators_count: int = 0
    claims_count: int = 0
    soft_features_count: int = 0
    staged_llm_steps: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ProductProfileV2(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    product_category: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    normalized_features: List[NormalizedFeature] = Field(default_factory=list)
    performance_parameters: List[NormalizedFeature] = Field(default_factory=list)
    metric_features: List[NormalizedFeature] = Field(default_factory=list)
    price_indicators: List[PriceIndicator] = Field(default_factory=list)
    soft_features: List[SoftFeature] = Field(default_factory=list)
    claims: List[ClaimItem] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)
    target_segments: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)
    quality_report: ExtractionQualityReport = Field(default_factory=ExtractionQualityReport)


class CompetitiveFeatureClaimExtractionV2Request(BaseModel):
    parsed_doc: Optional[Dict[str, Any]] = None
    parsed_doc_path: Optional[str] = None
    provider: str = "ionos"
    max_context_chars: int = Field(default=18000, ge=2000, le=80000)

    @model_validator(mode="after")
    def _validate_input(self) -> "CompetitiveFeatureClaimExtractionV2Request":
        if not self.parsed_doc and not (self.parsed_doc_path or "").strip():
            raise ValueError("Either parsed_doc or parsed_doc_path must be provided.")
        return self


class CompetitiveFeatureClaimExtractionV2Response(BaseModel):
    product_profile: ProductProfileV2


__all__ = [
    "ClaimItem",
    "CompetitiveFeatureClaimExtractionV2Request",
    "CompetitiveFeatureClaimExtractionV2Response",
    "ExtractionQualityReport",
    "NormalizedFeature",
    "PriceIndicator",
    "ProductProfileV2",
    "SoftFeature",
]
