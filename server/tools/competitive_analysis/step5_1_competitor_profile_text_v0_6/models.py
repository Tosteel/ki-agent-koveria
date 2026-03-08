from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CompetitorProductTextV06(BaseModel):
    product_name: str
    manufacturer: str
    url: str
    url_type: str = "unknown"
    relevance_score: float = 0.0
    similarity_score: float = 0.0
    plain_text: str = ""


class CompetitorProfileTextResultsV06(BaseModel):
    schema_version: str = "1.0"
    provider: str = "brave"
    generated_queries: List[str] = Field(default_factory=list)
    competitors: List[CompetitorProductTextV06] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorProfileTextV06Request(BaseModel):
    competitor_product_results: Optional[Dict[str, Any]] = None
    competitor_product_results_path: Optional[str] = None
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    provider: str = "brave"
    max_competitors: int = Field(default=200, ge=1, le=1000)
    brave_enable_research: bool = True
    brave_stream: bool = True
    brave_language: Optional[str] = "de"
    brave_country: Optional[str] = "DE"
    verbose_terminal: bool = False

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorProfileTextV06Request":
        if not self.competitor_product_results and not (self.competitor_product_results_path or "").strip():
            raise ValueError("Either competitor_product_results or competitor_product_results_path must be provided.")
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        return self


class CompetitorProfileTextV06Response(BaseModel):
    competitor_profile_text: CompetitorProfileTextResultsV06


__all__ = [
    "CompetitorProductTextV06",
    "CompetitorProfileTextResultsV06",
    "CompetitorProfileTextV06Request",
    "CompetitorProfileTextV06Response",
]
