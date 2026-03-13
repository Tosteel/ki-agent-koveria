from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Step31TrendMatrixItem(BaseModel):
    trend_name: str
    match_score: float = 0.0
    trend_keywords: List[str] = Field(default_factory=list)
    summary: str = ""


class Step31CompanyMatrixProfile(BaseModel):
    company: str
    website: str = ""
    region: str = ""
    customer_segment_bullets: str = ""
    actions_bullets: str = ""
    ratings_bullets: str = ""
    press_coverage_bullets: str = ""
    google_rating: Optional[float] = None
    google_review_count: Optional[int] = None
    social_media_reach_bullets: str = ""
    trend_items: List[Step31TrendMatrixItem] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step31MatrixResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step3.1_matrix.json"
    profiles: List[Step31CompanyMatrixProfile] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step31MatrixRequest(BaseModel):
    competitor_trends: Optional[Dict[str, Any]] = None
    competitor_trends_path: Optional[str] = None
    provider: str = "ionos"
    max_companies: int = Field(default=120, ge=1, le=1000)
    max_bullets_per_section: int = Field(default=4, ge=1, le=10)
    max_trend_bullets: int = Field(default=2, ge=1, le=8)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step31MatrixRequest":
        has_inline = isinstance(self.competitor_trends, dict) and bool(self.competitor_trends)
        has_path = bool((self.competitor_trends_path or "").strip())
        if not has_inline and not has_path:
            raise ValueError("Either competitor_trends or competitor_trends_path must be provided.")
        return self


class Step31MatrixResponse(BaseModel):
    matrix: Step31MatrixResult


__all__ = [
    "Step31CompanyMatrixProfile",
    "Step31MatrixRequest",
    "Step31MatrixResponse",
    "Step31MatrixResult",
    "Step31TrendMatrixItem",
]
