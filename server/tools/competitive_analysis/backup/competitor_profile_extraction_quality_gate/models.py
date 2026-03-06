from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from server.tools.competitive_analysis.backup.competitor_profile_extraction.models import CompetitorProfiles


class CompetitorProfileExtractionQualityGateRequest(BaseModel):
    competitor_profiles: Optional[Dict[str, Any]] = None
    competitor_profiles_path: Optional[str] = None
    provider: str = "perplexity"
    enrich_prices: bool = True
    max_price_pages_per_competitor: int = Field(default=4, ge=1, le=12)
    require_model_token_hits: int = Field(default=2, ge=1, le=5)
    verbose_progress: bool = True
    drop_unverified_features: bool = False

    @model_validator(mode="after")
    def _validate_input(self) -> "CompetitorProfileExtractionQualityGateRequest":
        if not self.competitor_profiles and not (self.competitor_profiles_path or "").strip():
            raise ValueError("Either competitor_profiles or competitor_profiles_path must be provided.")
        return self


class CompetitorProfileExtractionQualityGateResponse(BaseModel):
    competitor_profiles: CompetitorProfiles


__all__ = [
    "CompetitorProfileExtractionQualityGateRequest",
    "CompetitorProfileExtractionQualityGateResponse",
]
