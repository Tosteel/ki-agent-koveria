from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from server.tools.competitive_analysis.backup.competitor_identification import CompetitorList


class CompetitorIdentificationQualityGateRequest(BaseModel):
    competitor_list: Optional[Dict[str, Any]] = None
    competitor_list_path: Optional[str] = None
    product_profile: Optional[Dict[str, Any]] = None
    product_profile_path: Optional[str] = None

    provider: str = "perplexity"
    min_relevance_score: float = Field(default=0.06, ge=0.0, le=1.0)
    drop_generic_listing_pages: bool = True
    drop_weak_unknown_candidates: bool = True
    drop_manufacturer_nodes_without_model_signal: bool = True
    dedupe_by_name_and_domain: bool = True
    enable_llm_snippet_validation: bool = True
    llm_min_keep_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    max_llm_checks: int = Field(default=20, ge=0, le=80)

    @model_validator(mode="after")
    def _validate_input(self) -> "CompetitorIdentificationQualityGateRequest":
        if not self.competitor_list and not (self.competitor_list_path or "").strip():
            raise ValueError("Either competitor_list or competitor_list_path must be provided.")
        return self


class CompetitorIdentificationQualityReport(BaseModel):
    total_input_competitors: int
    total_output_competitors: int
    dropped_competitors: int
    deduped_competitors: int
    drop_reasons: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class CompetitorIdentificationQualityGateResponse(BaseModel):
    competitor_list: CompetitorList
    quality_report: CompetitorIdentificationQualityReport


__all__ = [
    "CompetitorIdentificationQualityGateRequest",
    "CompetitorIdentificationQualityGateResponse",
    "CompetitorIdentificationQualityReport",
]
