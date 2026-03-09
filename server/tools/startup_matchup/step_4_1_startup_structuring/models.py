from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class StructuredStartupCandidate(BaseModel):
    name: str = ""
    description: str = ""
    url: str = ""


class StartupStructuredList(BaseModel):
    startups: List[StructuredStartupCandidate] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep41Request(BaseModel):
    startup_candidates_raw: Optional[Dict[str, Any]] = None
    startup_candidates_raw_path: Optional[str] = None
    provider: str = "ionos"
    max_context_chars: int = Field(default=6000, ge=1000, le=20000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep41Request":
        has_raw = isinstance(self.startup_candidates_raw, dict) and bool(self.startup_candidates_raw)
        has_raw_path = bool((self.startup_candidates_raw_path or "").strip())
        if not has_raw and not has_raw_path:
            raise ValueError("startup_candidates_raw or startup_candidates_raw_path is required")
        return self


class StartupMatchupStep41Response(BaseModel):
    startup_structured_list: StartupStructuredList


__all__ = [
    "StartupMatchupStep41Request",
    "StartupMatchupStep41Response",
    "StartupStructuredList",
    "StructuredStartupCandidate",
]
