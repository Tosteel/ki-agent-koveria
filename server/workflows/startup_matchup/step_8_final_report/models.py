from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FinalReportNarrative(BaseModel):
    executive_summary: str = ""
    company_profile: str = ""
    innovation_goals: str = ""
    gap_analysis: str = ""
    startup_search_fields: str = ""
    recommended_startups: str = ""
    conclusion_next_steps: str = ""


class RecommendedStartup(BaseModel):
    rank: int
    startup_name: str = ""
    relevance_score: float = 0.0
    short_description: str = ""
    profile: Dict[str, Any] = Field(default_factory=dict)


class FinalReport(BaseModel):
    report: FinalReportNarrative = Field(default_factory=FinalReportNarrative)
    company_profile: Dict[str, Any] = Field(default_factory=dict)
    innovation_goals: List[str] = Field(default_factory=list)
    identified_gaps: List[str] = Field(default_factory=list)
    startup_search_fields: List[str] = Field(default_factory=list)
    recommended_startups: List[RecommendedStartup] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep8Request(BaseModel):
    workshop_analysis: Optional[Dict[str, Any]] = None
    workshop_analysis_path: Optional[str] = None
    company_profile: Optional[Dict[str, Any]] = None
    company_profile_path: Optional[str] = None
    gap_analysis: Optional[Dict[str, Any]] = None
    gap_analysis_path: Optional[str] = None
    startup_ranked_list: Optional[Dict[str, Any]] = None
    startup_ranked_list_path: Optional[str] = None
    startup_profiles: Optional[Dict[str, Any]] = None
    startup_profiles_path: Optional[str] = None
    provider: str = "ionos"
    max_context_chars: int = Field(default=14000, ge=2000, le=60000)
    top_k: int = Field(default=10, ge=1, le=30)


class StartupMatchupStep8Response(BaseModel):
    final_report: FinalReport


__all__ = [
    "FinalReport",
    "RecommendedStartup",
    "StartupMatchupStep8Request",
    "StartupMatchupStep8Response",
]
