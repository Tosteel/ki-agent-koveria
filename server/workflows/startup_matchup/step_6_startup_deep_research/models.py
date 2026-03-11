from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class DeepResearchItem(BaseModel):
    name: str = ""
    website: str = ""
    domain: str = ""
    query: str = ""
    raw_text: str = ""
    relevance_score: float = 0.0
    source: str = "brave_answers"


class StartupDeepProfilesRaw(BaseModel):
    selected_startups: List[str] = Field(default_factory=list)
    startup_research: List[DeepResearchItem] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep6Request(BaseModel):
    startup_ranked_list: Optional[Dict[str, Any]] = None
    startup_ranked_list_path: Optional[str] = None
    top_n: int = Field(default=10, ge=1, le=30)
    brave_enable_research: bool = False
    brave_stream: bool = True
    brave_language: Optional[str] = "de"
    brave_country: Optional[str] = "DE"

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep6Request":
        has_ranked = isinstance(self.startup_ranked_list, dict) and bool(self.startup_ranked_list)
        has_ranked_path = bool((self.startup_ranked_list_path or "").strip())
        if not has_ranked and not has_ranked_path:
            raise ValueError("startup_ranked_list or startup_ranked_list_path is required")
        return self


class StartupMatchupStep6Response(BaseModel):
    startup_deep_profiles_raw: StartupDeepProfilesRaw


__all__ = [
    "DeepResearchItem",
    "StartupDeepProfilesRaw",
    "StartupMatchupStep6Request",
    "StartupMatchupStep6Response",
]
