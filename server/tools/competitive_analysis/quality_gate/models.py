from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CompetitiveQualityGateRequest(BaseModel):
    artifact: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None
    step: Optional[int] = Field(default=None, ge=1, le=10)
    mode: str = Field(default="validate", pattern="^(validate|validate_and_repair)$")
    provider: str = "openai"
    max_context_chars: int = Field(default=18000, ge=2000, le=80000)

    @model_validator(mode="after")
    def _validate_input(self) -> "CompetitiveQualityGateRequest":
        if not self.artifact and not (self.artifact_path or "").strip():
            raise ValueError("Either artifact or artifact_path must be provided.")
        return self


class QualityIssue(BaseModel):
    severity: str
    code: str
    message: str
    path: str = ""


class CompetitiveQualityGateReport(BaseModel):
    step_detected: Optional[int] = None
    root_key: str = ""
    score: float = 0.0
    issues: List[QualityIssue] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    repaired: bool = False


class CompetitiveQualityGateResponse(BaseModel):
    artifact: Dict[str, Any]
    quality_report: CompetitiveQualityGateReport


__all__ = [
    "CompetitiveQualityGateRequest",
    "CompetitiveQualityGateReport",
    "CompetitiveQualityGateResponse",
    "QualityIssue",
]

