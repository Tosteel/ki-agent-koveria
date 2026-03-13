from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class StructuredSection(BaseModel):
    summary: str = ""
    source_urls: List[str] = Field(default_factory=list)


class RatingsReachSection(StructuredSection):
    google_rating: Optional[float] = None
    google_review_count: Optional[int] = None
    social_reach: str = ""


class Step23CompetitorProfile(BaseModel):
    company: str
    website: str = ""
    region: str = ""
    company_profile_target_audience: StructuredSection = Field(default_factory=StructuredSection)
    offers_actions: StructuredSection = Field(default_factory=StructuredSection)
    ratings_reach: RatingsReachSection = Field(default_factory=RatingsReachSection)
    press_coverage: StructuredSection = Field(default_factory=StructuredSection)
    source_urls: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step23CompetitorProfileStructuredResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step2.3_competitor_profile_structured.json"
    profiles: List[Step23CompetitorProfile] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step23CompetitorProfileStructuredRequest(BaseModel):
    competitor_profile_raw: Optional[Dict[str, Any]] = None
    competitor_profile_raw_path: Optional[str] = None
    provider: str = "ionos"
    max_context_chars: int = Field(default=22000, ge=4000, le=60000)
    max_companies: int = Field(default=120, ge=1, le=1000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step23CompetitorProfileStructuredRequest":
        has_inline = isinstance(self.competitor_profile_raw, dict) and bool(self.competitor_profile_raw)
        has_path = bool((self.competitor_profile_raw_path or "").strip())
        if not has_inline and not has_path:
            raise ValueError("Either competitor_profile_raw or competitor_profile_raw_path must be provided.")
        return self


class Step23CompetitorProfileStructuredResponse(BaseModel):
    competitor_profile_structured: Step23CompetitorProfileStructuredResult


__all__ = [
    "RatingsReachSection",
    "Step23CompetitorProfile",
    "Step23CompetitorProfileStructuredRequest",
    "Step23CompetitorProfileStructuredResponse",
    "Step23CompetitorProfileStructuredResult",
    "StructuredSection",
]
