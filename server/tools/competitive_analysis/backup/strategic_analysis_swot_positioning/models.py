from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class PrioritizedStatement(BaseModel):
    statement: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    impact: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = ""
    evidence_refs: List[str] = Field(default_factory=list)


class SwotData(BaseModel):
    strengths: List[PrioritizedStatement] = Field(default_factory=list)
    weaknesses: List[PrioritizedStatement] = Field(default_factory=list)
    opportunities: List[PrioritizedStatement] = Field(default_factory=list)
    threats: List[PrioritizedStatement] = Field(default_factory=list)


class PositioningData(BaseModel):
    market_space: str = ""
    primary_axis_x: str = "Preis"
    primary_axis_y: str = "Leistung"
    position_label: str = ""
    competitor_clusters: List[Dict[str, Any]] = Field(default_factory=list)


class StrategicImplication(BaseModel):
    title: str
    action: str
    horizon: str = "mid-term"
    priority: str = "medium"


class StrategicAnalysisResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    swot: SwotData
    positioning_data: PositioningData
    strategic_implications: List[StrategicImplication] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class StrategicAnalysisRequest(BaseModel):
    gaps_and_usps: Optional[Dict[str, Any]] = None
    gaps_and_usps_path: Optional[str] = None
    evidences: Optional[Dict[str, Any]] = None
    provider: str = "ionos"

    @model_validator(mode="after")
    def _validate_input(self) -> "StrategicAnalysisRequest":
        if not self.gaps_and_usps and not (self.gaps_and_usps_path or "").strip():
            raise ValueError("Either gaps_and_usps or gaps_and_usps_path must be provided.")
        return self


class StrategicAnalysisResponse(BaseModel):
    swot: Dict[str, Any]
    positioning_data: Dict[str, Any]
    strategic_implications: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


__all__ = [
    "StrategicAnalysisRequest",
    "StrategicAnalysisResponse",
    "StrategicAnalysisResult",
]
