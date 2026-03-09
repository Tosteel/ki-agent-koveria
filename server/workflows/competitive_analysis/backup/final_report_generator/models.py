from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ArtifactChunk(BaseModel):
    artifact: str
    key_points: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)


class PositioningPoint(BaseModel):
    name: str
    x: float
    y: float
    point_type: str = "competitor"


class PositioningDiagram(BaseModel):
    axis_x: str
    axis_y: str
    points: List[PositioningPoint] = Field(default_factory=list)
    interpretation: List[str] = Field(default_factory=list)


class SwotSummary(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    title: str
    action: str
    priority: str = "medium"
    horizon: str = "mid-term"
    evidence_refs: List[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    product_profile_brief: Dict[str, Any] = Field(default_factory=dict)
    competitor_overview: Dict[str, Any] = Field(default_factory=dict)
    feature_matrix_section: Dict[str, Any] = Field(default_factory=dict)
    gap_usp_analysis: Dict[str, Any] = Field(default_factory=dict)
    swot: SwotSummary
    positioning_diagram: PositioningDiagram
    strategic_recommendations: List[RecommendationItem] = Field(default_factory=list)
    executive_summary: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_items: List[str] = Field(default_factory=list)
    repaired: bool = False


class FinalReportResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    report_context: Dict[str, Any] = Field(default_factory=dict)
    artifact_chunks: List[ArtifactChunk] = Field(default_factory=list)
    final_report: FinalReport
    validation: ValidationReport
    extraction_warnings: List[str] = Field(default_factory=list)


class FinalReportRequest(BaseModel):
    artifacts: Optional[Dict[str, Any]] = None
    artifact_paths: Optional[Dict[str, str]] = None
    provider: str = "ionos"
    max_chars_per_artifact: int = Field(default=10000, ge=2000, le=50000)

    @model_validator(mode="after")
    def _validate_input(self) -> "FinalReportRequest":
        has_artifacts = isinstance(self.artifacts, dict) and bool(self.artifacts)
        has_paths = isinstance(self.artifact_paths, dict) and bool(self.artifact_paths)
        if not has_artifacts and not has_paths:
            raise ValueError("Either artifacts or artifact_paths must be provided.")
        return self


class FinalReportResponse(BaseModel):
    final_report: Dict[str, Any]
    validation: Dict[str, Any]
    report_context: Dict[str, Any]
    artifact_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


__all__ = [
    "FinalReportRequest",
    "FinalReportResponse",
    "FinalReportResult",
]
