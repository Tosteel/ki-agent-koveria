from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class FeatureMatrixGapAnalysisQualityGateRequest(BaseModel):
    feature_matrix_gap: Optional[Dict[str, Any]] = None
    feature_matrix_gap_path: Optional[str] = None
    provider: str = "perplexity"

    max_missing_features_per_competitor: int = Field(default=6, ge=1, le=30)
    max_urls_per_feature: int = Field(default=3, ge=1, le=8)
    max_llm_calls: int = Field(default=80, ge=1, le=1000)
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    verbose_progress: bool = True

    @model_validator(mode="after")
    def _validate_input(self) -> "FeatureMatrixGapAnalysisQualityGateRequest":
        if not self.feature_matrix_gap and not (self.feature_matrix_gap_path or "").strip():
            raise ValueError("Either feature_matrix_gap or feature_matrix_gap_path must be provided.")
        return self


class FeatureMatrixGapAnalysisQualityReport(BaseModel):
    total_missing_features_scanned: int
    llm_calls: int
    filled_features: int
    skipped_features: int
    notes: List[str] = Field(default_factory=list)


class FeatureMatrixGapAnalysisQualityGateResponse(BaseModel):
    comparison_matrix: Dict[str, Any]
    gaps_and_usps: Dict[str, Any]
    cluster_assignment: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)
    quality_report: FeatureMatrixGapAnalysisQualityReport


__all__ = [
    "FeatureMatrixGapAnalysisQualityGateRequest",
    "FeatureMatrixGapAnalysisQualityGateResponse",
    "FeatureMatrixGapAnalysisQualityReport",
]
