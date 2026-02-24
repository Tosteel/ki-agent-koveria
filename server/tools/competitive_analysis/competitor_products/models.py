from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ReferenceProduct(BaseModel):
    product_name: str
    category: str = ""
    url: str
    snippet: str = ""
    similarity_score: float = 0.0


class CompetitorWithProducts(BaseModel):
    name: str
    cluster: str = ""
    year_founded: int = 0
    headquarters_country: str = ""
    company_description: str = ""
    primary_business_segments: List[str] = Field(default_factory=list)
    relevance_in_reference_segment: str = ""
    competitor_type: str = "Direct competitor"
    company_website_url: str = ""
    brand_domain_whitelist: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    reference_products: List[ReferenceProduct] = Field(default_factory=list)


class CompetitorProductsResults(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    generated_queries: List[str] = Field(default_factory=list)
    min_competitors_target: int = 6
    competitors: List[CompetitorWithProducts] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitorProductsRequest(BaseModel):
    competitor_search_results: Optional[Dict[str, Any]] = None
    competitor_search_results_path: Optional[str] = None
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None
    provider: str = "openai"
    per_query_results: int = Field(default=8, ge=3, le=30)
    top_products_per_company: int = Field(default=3, ge=1, le=15)
    max_queries_per_company: int = Field(default=5, ge=2, le=12)
    semantic_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    feature_match_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    performance_similarity_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    price_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    emit_all_candidates: bool = False
    manufacturer_domain_only: bool = False
    verbose_terminal: bool = False

    @model_validator(mode="after")
    def _validate_inputs(self) -> "CompetitorProductsRequest":
        if not self.competitor_search_results and not (self.competitor_search_results_path or "").strip():
            raise ValueError("Either competitor_search_results or competitor_search_results_path must be provided.")
        if not self.product_profile and not (self.product_profile_path or "").strip():
            raise ValueError("Either product_profile or product_profile_path must be provided.")
        if (self.semantic_weight + self.feature_match_weight + self.performance_similarity_weight + self.price_weight) <= 0:
            raise ValueError("At least one scoring weight must be > 0.")
        return self


class CompetitorProductsResponse(BaseModel):
    competitor_products_results: CompetitorProductsResults


__all__ = [
    "CompetitorProductsRequest",
    "CompetitorProductsResponse",
    "CompetitorProductsResults",
    "CompetitorWithProducts",
    "ReferenceProduct",
]
