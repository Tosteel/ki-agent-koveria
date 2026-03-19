from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


DecisionType = Literal["auto_accept", "auto_decline", "human_review", "need_clarification"]


class BookingExtractFactsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Mail-/Thread-Text für Fakt-Extraktion.")
    timezone: str = Field(default="Europe/Berlin")


class BookingExtractFactsResponse(BaseModel):
    facts: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    missing_candidates: List[str] = Field(default_factory=list)
    text: str = ""


class BookingValidateCompletenessRequest(BaseModel):
    facts: Dict[str, Any] = Field(default_factory=dict)
    required_fields: List[str] = Field(default_factory=list)


class BookingValidateCompletenessResponse(BaseModel):
    complete: bool = False
    missing_fields: List[str] = Field(default_factory=list)
    present_fields: List[str] = Field(default_factory=list)
    text: str = ""


class BookingDecisionEngineRequest(BaseModel):
    facts: Dict[str, Any] = Field(default_factory=dict)
    profile_rules: Dict[str, Any] = Field(default_factory=dict)
    completeness: Dict[str, Any] = Field(default_factory=dict)
    distance: Dict[str, Any] = Field(default_factory=dict)
    quote: Dict[str, Any] = Field(default_factory=dict)


class BookingDecisionEngineResponse(BaseModel):
    decision: DecisionType = "need_clarification"
    reasons: List[str] = Field(default_factory=list)
    flags: Dict[str, Any] = Field(default_factory=dict)
    text: str = ""
