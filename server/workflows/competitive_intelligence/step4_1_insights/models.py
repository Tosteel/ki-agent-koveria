from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Step41InsightsResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step4.1_insights.json"
    customer_segment_insights: str = "-"
    actions_insights: str = "-"
    ratings_insights: str = "-"
    trend_items_insights: str = "-"
    competitor_comparison_insights: str = "-"
    source_urls: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step41InsightsRequest(BaseModel):
    matrix: Optional[Dict[str, Any]] = None
    matrix_path: Optional[str] = None
    provider: str = "ionos"
    max_companies: int = Field(default=120, ge=1, le=1000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step41InsightsRequest":
        has_inline = isinstance(self.matrix, dict) and bool(self.matrix)
        has_path = bool((self.matrix_path or "").strip())
        if not has_inline and not has_path:
            raise ValueError("Either matrix or matrix_path must be provided.")
        return self


class Step41InsightsResponse(BaseModel):
    insights: Step41InsightsResult


__all__ = [
    "Step41InsightsRequest",
    "Step41InsightsResponse",
    "Step41InsightsResult",
]

