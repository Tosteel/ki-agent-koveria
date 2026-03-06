from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class StrateticAnalysisSwotPositioningV05Request(BaseModel):
    feature_matrix_gap: Optional[Dict[str, Any]] = None
    feature_matrix_gap_path: Optional[str] = None
    comparison_matrix: Optional[Dict[str, Any]] = None
    gaps_and_usps: Optional[Dict[str, Any]] = None
    evidences: Optional[Dict[str, Any]] = None
    provider: str = "openai"

    @model_validator(mode="after")
    def _validate_input(self) -> "StrateticAnalysisSwotPositioningV05Request":
        has_full = bool(self.feature_matrix_gap) or bool((self.feature_matrix_gap_path or "").strip())
        has_parts = bool(self.comparison_matrix) and bool(self.gaps_and_usps)
        if not has_full and not has_parts:
            raise ValueError(
                "Provide either feature_matrix_gap/feature_matrix_gap_path or both comparison_matrix and gaps_and_usps."
            )
        return self


class StrateticAnalysisSwotPositioningV05Response(BaseModel):
    swot: Dict[str, Any]
    positioning_data: Dict[str, Any]
    strategic_implications: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


__all__ = [
    "StrateticAnalysisSwotPositioningV05Request",
    "StrateticAnalysisSwotPositioningV05Response",
]
