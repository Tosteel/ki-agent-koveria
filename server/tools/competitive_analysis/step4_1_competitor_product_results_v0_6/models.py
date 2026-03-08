from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ProductCompetitorSlim(BaseModel):
    product_name: str
    manufacturer: str
    url: str
    url_type: str = "unknown"
    relevance_score: float = 0.0
    similarity_score: float = 0.0


class CompetitorProductResultsV06(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    generated_queries: List[str] = Field(default_factory=list)
    competitors: List[ProductCompetitorSlim] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorProductResultsV06Request(BaseModel):
    competitor_search_results: Optional[Dict[str, Any]] = None
    competitor_search_results_path: Optional[str] = None
    provider: str = "openai"
    top_n: int = Field(default=20, ge=1, le=200)
    verbose_terminal: bool = False

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorProductResultsV06Request":
        if not self.competitor_search_results and not (self.competitor_search_results_path or "").strip():
            raise ValueError("Either competitor_search_results or competitor_search_results_path must be provided.")
        return self


class CompetitorProductResultsV06Response(BaseModel):
    competitor_product_results: CompetitorProductResultsV06


__all__ = [
    "ProductCompetitorSlim",
    "CompetitorProductResultsV06",
    "CompetitorProductResultsV06Request",
    "CompetitorProductResultsV06Response",
]

