from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Step12MarketTrendSource(BaseModel):
    url: str
    originaltext: str
    originaltext_raw: str
    kernaussage: List[str] = Field(default_factory=list)


class Step12MarketTrendsStructuredResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step1.2_market_trends_structured.json"
    sources: List[Step12MarketTrendSource] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step12MarketTrendsStructuredRequest(BaseModel):
    market_trends_raw: Optional[Dict[str, Any]] = None
    market_trends_raw_path: Optional[str] = None
    provider: str = "ionos"
    use_view_website: bool = True
    view_query: Optional[str] = None
    view_max_matches: int = Field(default=8, ge=1, le=30)
    view_context_chars: int = Field(default=180, ge=60, le=600)
    view_timeout_ms: int = Field(default=15000, ge=2000, le=120000)
    max_sources: int = Field(default=12, ge=1, le=100)
    max_chars_per_source: int = Field(default=120000, ge=1000, le=500000)
    summary_bullets: int = Field(default=4, ge=1, le=10)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step12MarketTrendsStructuredRequest":
        if not self.market_trends_raw and not (self.market_trends_raw_path or "").strip():
            raise ValueError("Either market_trends_raw or market_trends_raw_path must be provided.")
        return self


class Step12MarketTrendsStructuredResponse(BaseModel):
    market_trends_structured: Step12MarketTrendsStructuredResult


__all__ = [
    "Step12MarketTrendSource",
    "Step12MarketTrendsStructuredRequest",
    "Step12MarketTrendsStructuredResponse",
    "Step12MarketTrendsStructuredResult",
]
