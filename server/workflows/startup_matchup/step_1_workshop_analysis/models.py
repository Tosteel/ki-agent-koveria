from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class WorkshopAnalysis(BaseModel):
    innovation_goals: List[str] = Field(default_factory=list)
    strategic_fields: List[str] = Field(default_factory=list)
    problem_statements: List[str] = Field(default_factory=list)
    technology_interests: List[str] = Field(default_factory=list)
    target_use_cases: List[str] = Field(default_factory=list)
    extracted_topics: List[str] = Field(default_factory=list)
    source_path: str = ""
    extraction_warnings: List[str] = Field(default_factory=list)


class StartupMatchupStep1Request(BaseModel):
    workshop_document_path: Optional[str] = None
    workshop_text: Optional[str] = None
    provider: str = "ionos"
    max_chars: int = Field(default=60000, ge=1000, le=300000)
    max_context_chars: int = Field(default=18000, ge=2000, le=60000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep1Request":
        has_path = bool((self.workshop_document_path or "").strip())
        has_text = bool((self.workshop_text or "").strip())
        if not has_path and not has_text:
            raise ValueError("Either workshop_document_path or workshop_text must be provided.")
        return self


class StartupMatchupStep1Response(BaseModel):
    workshop_analysis: WorkshopAnalysis


__all__ = [
    "StartupMatchupStep1Request",
    "StartupMatchupStep1Response",
    "WorkshopAnalysis",
]
