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


class CompetitorSearchResultsV05(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    generated_queries: List[str] = Field(default_factory=list)
    competitors: List[ProductCompetitorSlim] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorSearchV05Request(BaseModel):
    analysis_plan: Optional[Dict[str, Any]] = None
    analysis_plan_path: Optional[str] = None
    provider: str = "openai"
    max_queries: int = Field(default=20, ge=1, le=50)
    per_query_results: int = Field(default=10, ge=3, le=20)
    max_candidates_to_check: int = Field(default=200, ge=10, le=1000)
    verbose_terminal: bool = False
    verbose_search_hits: bool = False

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorSearchV05Request":
        if not self.analysis_plan and not (self.analysis_plan_path or "").strip():
            raise ValueError("Either analysis_plan or analysis_plan_path must be provided.")
        return self


class CompetitorSearchV05Response(BaseModel):
    competitor_search_results: CompetitorSearchResultsV05


__all__ = [
    "ProductCompetitorSlim",
    "CompetitorSearchResultsV05",
    "CompetitorSearchV05Request",
    "CompetitorSearchV05Response",
]
