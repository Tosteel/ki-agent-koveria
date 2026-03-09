from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class GapAnalysis(BaseModel):
    identified_gaps: List[str] = Field(default_factory=list)
    innovation_opportunities: List[str] = Field(default_factory=list)
    startup_search_fields: List[str] = Field(default_factory=list)
    startup_search_queries: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep3Request(BaseModel):
    workshop_analysis: Optional[Dict[str, Any]] = None
    workshop_analysis_path: Optional[str] = None
    company_profile: Optional[Dict[str, Any]] = None
    company_profile_path: Optional[str] = None
    provider: str = "ionos"
    max_queries: int = Field(default=16, ge=3, le=30)
    max_context_chars: int = Field(default=18000, ge=3000, le=70000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep3Request":
        has_w = isinstance(self.workshop_analysis, dict) and bool(self.workshop_analysis)
        has_w_path = bool((self.workshop_analysis_path or "").strip())
        has_c = isinstance(self.company_profile, dict) and bool(self.company_profile)
        has_c_path = bool((self.company_profile_path or "").strip())
        if not has_w and not has_w_path:
            raise ValueError("workshop_analysis or workshop_analysis_path is required")
        if not has_c and not has_c_path:
            raise ValueError("company_profile or company_profile_path is required")
        return self


class StartupMatchupStep3Response(BaseModel):
    gap_analysis: GapAnalysis


__all__ = [
    "GapAnalysis",
    "StartupMatchupStep3Request",
    "StartupMatchupStep3Response",
]
