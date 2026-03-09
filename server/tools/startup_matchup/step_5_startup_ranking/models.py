from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class RankedStartup(BaseModel):
    name: str = ""
    description: str = ""
    website: str = ""
    relevance_score: float = 0.0
    score_breakdown: Dict[str, float] = Field(default_factory=dict)


class StartupRankedList(BaseModel):
    startups: List[RankedStartup] = Field(default_factory=list)
    scoring_formula: str = ""
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep5Request(BaseModel):
    startup_structured_list: Optional[Dict[str, Any]] = None
    startup_structured_list_path: Optional[str] = None
    gap_analysis: Optional[Dict[str, Any]] = None
    gap_analysis_path: Optional[str] = None
    company_profile: Optional[Dict[str, Any]] = None
    company_profile_path: Optional[str] = None
    top_k: int = Field(default=25, ge=5, le=100)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep5Request":
        has_structured = isinstance(self.startup_structured_list, dict) and bool(self.startup_structured_list)
        has_structured_path = bool((self.startup_structured_list_path or "").strip())
        if not has_structured and not has_structured_path:
            raise ValueError("startup_structured_list or startup_structured_list_path is required")
        return self


class StartupMatchupStep5Response(BaseModel):
    startup_ranked_list: StartupRankedList


__all__ = [
    "RankedStartup",
    "StartupMatchupStep5Request",
    "StartupMatchupStep5Response",
    "StartupRankedList",
]
