from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Step51RecommendationsResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step5.1_recommendations.json"
    customer_segements_recommendations: str = "-"
    actions_recommendations: str = "-"
    ratings_recommendations: str = "-"
    trend_items_recommendations: str = "-"
    competitor_comparison_recommendations: str = "-"
    source_urls: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step51RecommendationsRequest(BaseModel):
    matrix: Optional[Dict[str, Any]] = None
    matrix_path: Optional[str] = None
    provider: str = "ionos"
    max_companies: int = Field(default=120, ge=1, le=1000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step51RecommendationsRequest":
        has_inline = isinstance(self.matrix, dict) and bool(self.matrix)
        has_path = bool((self.matrix_path or "").strip())
        if not has_inline and not has_path:
            raise ValueError("Either matrix or matrix_path must be provided.")
        return self


class Step51RecommendationsResponse(BaseModel):
    recommendations: Step51RecommendationsResult


__all__ = [
    "Step51RecommendationsRequest",
    "Step51RecommendationsResponse",
    "Step51RecommendationsResult",
]

