from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


DecisionType = Literal["auto_accept", "auto_decline", "human_review", "need_clarification"]


class BookingExtractFactsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Mail-/Thread-Text für Fakt-Extraktion.")
    timezone: str = Field(default="Europe/Berlin")
    required_fields: List[str] = Field(default_factory=list)


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
    require_price_confirmation: bool = True


class BookingDecisionEngineResponse(BaseModel):
    decision: DecisionType = "need_clarification"
    reasons: List[str] = Field(default_factory=list)
    flags: Dict[str, Any] = Field(default_factory=dict)
    text: str = ""


class BookingReplyScoreRequest(BaseModel):
    user_message: str = Field(default="", description="Originale Kundenanfrage inkl. optionalem Thread.")
    draft_reply: str = Field(..., min_length=1, description="Geplanter Antwortentwurf.")
    booking_decision: str = Field(default="", description="Aktuelle Booking-Entscheidung (z. B. auto_accept).")
    facts: Dict[str, Any] = Field(default_factory=dict, description="Extrahierte Booking-Fakten.")
    required_fields: List[str] = Field(default_factory=list, description="Pflichtfelder aus dem Profil.")
    missing_fields: List[str] = Field(default_factory=list, description="Aktuell fehlende Pflichtfelder.")
    knowledge_evidence: List[str] = Field(default_factory=list, description="Optionale Evidenzquellen (RAG/Web).")
    require_actionable: bool = Field(
        default=True,
        description="Wenn true, soll die Antwort einen klaren naechsten Prozessschritt enthalten.",
    )


class BookingReplyScoreResponse(BaseModel):
    total_score: float = 0.0
    verdict: str = "needs_review"
    dimensions: Dict[str, float] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    next_step: str = ""
    text: str = ""
    model: str = ""
    fallback_used: bool = False


class BookingInstructionCheckRequest(BaseModel):
    instructions: List[str] = Field(default_factory=list, description="Freitext-Instruktionen aus dem Assistant-Profil.")
    user_message: str = Field(default="", description="Originale Kundenanfrage inkl. optionalem Thread.")
    draft_reply: str = Field(..., min_length=1, description="Geplante Antwort, die geprueft werden soll.")
    booking_decision: str = Field(default="", description="Aktuelle Booking-Entscheidung.")
    facts: Dict[str, Any] = Field(default_factory=dict, description="Extrahierte Buchungsfakten.")


class BookingInstructionCheckResponse(BaseModel):
    allowed: bool = True
    confidence: float = 0.0
    risk_level: str = "low"
    violations: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    reason: str = ""
    text: str = ""
    model: str = ""
    fallback_used: bool = False
