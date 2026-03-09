from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class StartupSearchResult(BaseModel):
    snippet: str = ""
    url: str = ""
    source: str = ""


class QueryLog(BaseModel):
    query: str
    result_excerpt: str = ""


class StartupCandidatesRaw(BaseModel):
    queries: List[str] = Field(default_factory=list)
    search_results: List[StartupSearchResult] = Field(default_factory=list)
    query_logs: List[QueryLog] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep4Request(BaseModel):
    gap_analysis: Optional[Dict[str, Any]] = None
    gap_analysis_path: Optional[str] = None
    max_queries: int = Field(default=16, ge=1, le=30)
    per_query_results: int = Field(default=8, ge=3, le=20)
    brave_enable_research: bool = True
    brave_stream: bool = True
    brave_language: Optional[str] = "de"
    brave_country: Optional[str] = "DE"

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep4Request":
        has_gap = isinstance(self.gap_analysis, dict) and bool(self.gap_analysis)
        has_gap_path = bool((self.gap_analysis_path or "").strip())
        if not has_gap and not has_gap_path:
            raise ValueError("gap_analysis or gap_analysis_path is required")
        return self


class StartupMatchupStep4Response(BaseModel):
    startup_candidates_raw: StartupCandidatesRaw


__all__ = [
    "QueryLog",
    "StartupCandidatesRaw",
    "StartupMatchupStep4Request",
    "StartupMatchupStep4Response",
    "StartupSearchResult",
]
