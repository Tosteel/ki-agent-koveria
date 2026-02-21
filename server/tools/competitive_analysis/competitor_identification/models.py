from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CompetitorCandidate(BaseModel):
    name: str
    url: str
    source_type: str = "unknown"
    url_candidates: List[str] = Field(default_factory=list)
    url_provenance: Dict[str, str] = Field(default_factory=dict)
    snippet: str = ""
    source_query: str = ""
    cluster: str = "unknown"
    similarity_score: float = 0.0
    relevance_score: float = 0.0
    matched_dimensions: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)


class CompetitorList(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    generated_queries: List[str] = Field(default_factory=list)
    min_competitors_target: int = 5
    competitors: List[CompetitorCandidate] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorIdentificationRequest(BaseModel):
    analysis_plan: Optional[Dict[str, Any]] = None
    analysis_plan_path: Optional[str] = None
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    provider: str = "openai"
    max_queries: int = Field(default=12, ge=1, le=20)
    per_query_results: int = Field(default=6, ge=2, le=20)
    shortlist_size: int = Field(default=10, ge=3, le=50)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorIdentificationRequest":
        if not self.analysis_plan and not (self.analysis_plan_path or "").strip():
            raise ValueError("Either analysis_plan or analysis_plan_path must be provided.")
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        return self


class CompetitorIdentificationResponse(BaseModel):
    competitor_list: CompetitorList


__all__ = [
    "CompetitorCandidate",
    "CompetitorIdentificationRequest",
    "CompetitorIdentificationResponse",
    "CompetitorList",
]
