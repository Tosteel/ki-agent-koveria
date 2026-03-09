from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class PerformanceCellV05(BaseModel):
    name: str
    value: float | int | str | None = None
    unit: str = ""
    present: bool = False


class PriceCellV05(BaseModel):
    context: str
    raw: str = ""
    value: float | int | str | None = None
    currency: str = ""
    period: str = ""
    present: bool = False


class SoftFeatureCellV05(BaseModel):
    name: str
    available: bool = False


class CompetitorRowV05(BaseModel):
    competitor: str
    cluster: str = "unknown"
    performance_parameters: List[PerformanceCellV05] = Field(default_factory=list)
    metric_features: List[PerformanceCellV05] = Field(default_factory=list)
    price_indicators: List[PriceCellV05] = Field(default_factory=list)
    soft_features: List[SoftFeatureCellV05] = Field(default_factory=list)
    avg_price: float | None = None
    value_score: float | None = None


class ComparisonMatrixV05(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    baseline_product: str = ""
    performance_dimensions: List[str] = Field(default_factory=list)
    metric_dimensions: List[str] = Field(default_factory=list)
    price_dimensions: List[str] = Field(default_factory=list)
    soft_feature_dimensions: List[str] = Field(default_factory=list)
    baseline_row: CompetitorRowV05
    competitor_rows: List[CompetitorRowV05] = Field(default_factory=list)


class GapItemV05(BaseModel):
    feature_group: str
    feature: str
    status: str
    market_presence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation: str = ""


class UspItemV05(BaseModel):
    feature_group: str
    feature: str
    market_presence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    rarity_score: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""


class GapsAndUspsV05(BaseModel):
    gaps: List[GapItemV05] = Field(default_factory=list)
    usps: List[UspItemV05] = Field(default_factory=list)
    market_standards: List[str] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)


class ClusterAssignmentV05(BaseModel):
    competitor: str
    cluster: str
    avg_price: float | None = None
    value_score: float | None = None


class FeatureMatrixGapAnalysisV05Result(BaseModel):
    comparison_matrix: ComparisonMatrixV05
    gaps_and_usps: GapsAndUspsV05
    cluster_assignment: List[ClusterAssignmentV05] = Field(default_factory=list)


class FeatureMatrixGapAnalysisV05Request(BaseModel):
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    competitor_profile_results: Optional[Dict[str, Any]] = None
    competitor_profile_results_path: Optional[str] = None
    provider: str = "openai"

    @model_validator(mode="after")
    def _validate_input(self) -> "FeatureMatrixGapAnalysisV05Request":
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        if not self.competitor_profile_results and not (self.competitor_profile_results_path or "").strip():
            raise ValueError("Either competitor_profile_results or competitor_profile_results_path must be provided.")
        return self


class FeatureMatrixGapAnalysisV05Response(BaseModel):
    feature_matrix_gap: Dict[str, Any]


# Backward-compatible aliases for older imports.
FeatureMatrxGapAnalysisV05Result = FeatureMatrixGapAnalysisV05Result
FeatureMatrxGapAnalysisV05Request = FeatureMatrixGapAnalysisV05Request
FeatureMatrxGapAnalysisV05Response = FeatureMatrixGapAnalysisV05Response


__all__ = [
    "FeatureMatrixGapAnalysisV05Request",
    "FeatureMatrixGapAnalysisV05Response",
    "FeatureMatrixGapAnalysisV05Result",
    "FeatureMatrxGapAnalysisV05Request",
    "FeatureMatrxGapAnalysisV05Response",
    "FeatureMatrxGapAnalysisV05Result",
]
