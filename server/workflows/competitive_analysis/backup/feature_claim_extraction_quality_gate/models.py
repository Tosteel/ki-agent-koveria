from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from server.workflows.competitive_analysis.backup.feature_claim_extraction.models import ProductProfile


class FeatureClaimExtractionQualityGateRequest(BaseModel):
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    provider: str = "openai"
    max_context_chars: int = Field(default=18000, ge=2000, le=80000)
    remove_nonsensical_features: bool = True
    repair_feature_names: bool = True
    min_alpha_chars: int = Field(default=2, ge=1, le=10)
    max_feature_name_length: int = Field(default=96, ge=24, le=240)
    allow_llm_fallback: bool = True

    @model_validator(mode="after")
    def _validate_input(self) -> "FeatureClaimExtractionQualityGateRequest":
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        return self


class FeatureClaimQualityReport(BaseModel):
    total_input_features: int
    total_output_features: int
    dropped_features: int
    repaired_features: int
    drop_reasons: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class FeatureClaimExtractionQualityGateResponse(BaseModel):
    product_profile: ProductProfile
    quality_report: FeatureClaimQualityReport


__all__ = [
    "FeatureClaimExtractionQualityGateRequest",
    "FeatureClaimExtractionQualityGateResponse",
    "FeatureClaimQualityReport",
]
