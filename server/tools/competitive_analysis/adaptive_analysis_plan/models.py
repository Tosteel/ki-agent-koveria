from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ComparisonDimension(BaseModel):
    name: str
    weight: float = Field(default=0.2, ge=0.0, le=1.0)
    rationale: str = ""
    required_fields: List[str] = Field(default_factory=list)


class SearchTerm(BaseModel):
    term: str
    intent: str = "generic"


class AnalysisScope(BaseModel):
    depth: str = "medium"
    breadth: str = "medium"
    include_regional: bool = True
    include_global: bool = True
    max_results_per_query: int = Field(default=20, ge=5, le=200)


class AnalysisPlan(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    product_category: str = "unknown"
    comparison_dimensions: List[ComparisonDimension] = Field(default_factory=list)
    extended_feature_schema: List[str] = Field(default_factory=list)
    search_terms: List[SearchTerm] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    min_competitors: int = Field(default=5, ge=2, le=50)
    relevance_criteria: List[str] = Field(default_factory=list)
    analysis_scope: AnalysisScope = Field(default_factory=AnalysisScope)
    notes: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class AdaptiveAnalysisPlanRequest(BaseModel):
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    provider: str = "ionos"
    max_context_chars: int = Field(default=14000, ge=2000, le=80000)

    @model_validator(mode="after")
    def _validate_input(self) -> "AdaptiveAnalysisPlanRequest":
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        return self


class AdaptiveAnalysisPlanResponse(BaseModel):
    analysis_plan: AnalysisPlan


__all__ = [
    "AdaptiveAnalysisPlanRequest",
    "AdaptiveAnalysisPlanResponse",
    "AnalysisPlan",
    "AnalysisScope",
    "ComparisonDimension",
    "SearchTerm",
]
