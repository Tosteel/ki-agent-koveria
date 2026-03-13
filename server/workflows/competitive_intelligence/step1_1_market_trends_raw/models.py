from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Step11MarketTrendsRawRequest(BaseModel):
    market_context: str = Field(..., min_length=1)
    search_sources: List[str] = Field(default_factory=list)
    provider: str = "brave"
    brave_stream: bool = True
    brave_language: Optional[str] = "it"
    brave_country: Optional[str] = "IT"
    brave_enable_research: bool = False
    brave_enable_citations: bool = True
    brave_enable_entities: bool = True
    timeout_s: int = Field(default=90, ge=10, le=300)

    @field_validator("market_context")
    @classmethod
    def _validate_market_context(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("market_context must not be empty.")
        return text

    @field_validator("search_sources", mode="before")
    @classmethod
    def _normalize_search_sources(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("search_sources must be a list of strings.")
        out: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out


class MarketTrendsRawResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "brave"
    output_file: str = "step1.1_market_trends_raw.json"
    market_context: str
    search_sources: List[str] = Field(default_factory=list)
    query: str
    raw_text: str = ""
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step11MarketTrendsRawResponse(BaseModel):
    market_trends_raw: MarketTrendsRawResult


__all__ = [
    "MarketTrendsRawResult",
    "Step11MarketTrendsRawRequest",
    "Step11MarketTrendsRawResponse",
]
