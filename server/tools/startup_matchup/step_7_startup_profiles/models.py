from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class StructuredStartupProfile(BaseModel):
    name: str = ""
    founding_year: str = ""
    location: str = ""
    technology_focus: List[str] = Field(default_factory=list)
    description: str = ""
    why_relevant: str = ""
    relevance_score: float = 0.0
    website: str = ""
    domain: str = ""


class StartupProfiles(BaseModel):
    startups: List[StructuredStartupProfile] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep7Request(BaseModel):
    startup_deep_profiles_raw: Optional[Dict[str, Any]] = None
    startup_deep_profiles_raw_path: Optional[str] = None
    gap_analysis: Optional[Dict[str, Any]] = None
    gap_analysis_path: Optional[str] = None
    startup_ranked_list: Optional[Dict[str, Any]] = None
    startup_ranked_list_path: Optional[str] = None
    provider: str = "ionos"
    max_context_chars: int = Field(default=18000, ge=3000, le=70000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep7Request":
        has_raw = isinstance(self.startup_deep_profiles_raw, dict) and bool(self.startup_deep_profiles_raw)
        has_raw_path = bool((self.startup_deep_profiles_raw_path or "").strip())
        if not has_raw and not has_raw_path:
            raise ValueError("startup_deep_profiles_raw or startup_deep_profiles_raw_path is required")
        return self


class StartupMatchupStep7Response(BaseModel):
    startup_profiles: StartupProfiles


__all__ = [
    "StartupMatchupStep7Request",
    "StartupMatchupStep7Response",
    "StartupProfiles",
    "StructuredStartupProfile",
]
