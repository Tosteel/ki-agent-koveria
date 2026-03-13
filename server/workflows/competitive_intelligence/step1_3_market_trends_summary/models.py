from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Step13TrendSummaryItem(BaseModel):
    summary: str
    source_urls: List[str] = Field(default_factory=list)
    source_count: int = 0
    evidence_points: List[str] = Field(default_factory=list)


class Step13MarketTrendsSummaryResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step1.3_market_trends_summary.json"
    summaries: List[Step13TrendSummaryItem] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step13MarketTrendsSummaryRequest(BaseModel):
    market_trends_structured: Optional[Dict[str, Any]] = None
    market_trends_structured_path: Optional[str] = None
    provider: str = "ionos"
    similarity_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_summary_items: int = Field(default=12, ge=1, le=100)
    max_evidence_per_item: int = Field(default=6, ge=1, le=20)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step13MarketTrendsSummaryRequest":
        if not self.market_trends_structured and not (self.market_trends_structured_path or "").strip():
            raise ValueError("Either market_trends_structured or market_trends_structured_path must be provided.")
        return self


class Step13MarketTrendsSummaryResponse(BaseModel):
    market_trends_summary: Step13MarketTrendsSummaryResult


__all__ = [
    "Step13MarketTrendsSummaryRequest",
    "Step13MarketTrendsSummaryResponse",
    "Step13MarketTrendsSummaryResult",
    "Step13TrendSummaryItem",
]
