from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class FeatureCell(BaseModel):
    feature: str
    value: str = ""
    normalized_value: float | int | str | None = None
    unit: str = ""
    present: bool = False


class CompetitorRow(BaseModel):
    competitor: str
    cluster: str = "unknown"
    features: List[FeatureCell] = Field(default_factory=list)
    avg_price: float | None = None
    value_score: float | None = None


class ComparisonMatrix(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    feature_dimensions: List[str] = Field(default_factory=list)
    baseline_product: str = ""
    baseline_row: CompetitorRow
    competitor_rows: List[CompetitorRow] = Field(default_factory=list)


class GapItem(BaseModel):
    feature: str
    status: str
    market_presence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation: str = ""


class UspItem(BaseModel):
    feature: str
    rationale: str = ""


class GapsAndUsps(BaseModel):
    gaps: List[GapItem] = Field(default_factory=list)
    usps: List[UspItem] = Field(default_factory=list)
    market_standards: List[str] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)


class ClusterAssignment(BaseModel):
    competitor: str
    cluster: str
    avg_price: float | None = None
    value_score: float | None = None


class FeatureMatrixGapAnalysisResult(BaseModel):
    comparison_matrix: ComparisonMatrix
    gaps_and_usps: GapsAndUsps
    cluster_assignment: List[ClusterAssignment] = Field(default_factory=list)


class FeatureMatrixGapAnalysisRequest(BaseModel):
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    competitor_profiles: Optional[Dict[str, Any]] = None
    competitor_profiles_path: Optional[str] = None
    provider: str = "ionos"

    @model_validator(mode="after")
    def _validate_input(self) -> "FeatureMatrixGapAnalysisRequest":
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        if not self.competitor_profiles and not (self.competitor_profiles_path or "").strip():
            raise ValueError("Either competitor_profiles or competitor_profiles_path must be provided.")
        return self


class FeatureMatrixGapAnalysisResponse(BaseModel):
    comparison_matrix: Dict[str, Any]
    gaps_and_usps: Dict[str, Any]
    cluster_assignment: List[Dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "FeatureMatrixGapAnalysisRequest",
    "FeatureMatrixGapAnalysisResponse",
    "FeatureMatrixGapAnalysisResult",
]
