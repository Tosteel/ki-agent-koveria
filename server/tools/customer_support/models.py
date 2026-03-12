from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ScoreReplyRequest(BaseModel):
    user_message: str = Field(default="", description="Originale Nutzeranfrage.")
    draft_reply: str = Field(..., min_length=1, description="Antwortentwurf, der bewertet werden soll.")
    knowledge_evidence: List[str] = Field(
        default_factory=list,
        description="Optionale Evidenzpunkte (z. B. aus RAG/Web), auf die sich die Antwort stuetzt.",
    )
    require_actionable: bool = Field(
        default=True,
        description="Wenn True, soll die Antwort moeglichst konkrete naechste Schritte enthalten.",
    )


class ScoreReplyResponse(BaseModel):
    total_score: float = 0.0
    verdict: str = "needs_review"
    dimensions: Dict[str, float] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    text: str = ""


class CreateReviewTicketRequest(BaseModel):
    title: str = Field(..., min_length=3, description="Kurzer Ticket-Titel.")
    user_message: str = Field(default="", description="Originale Nutzeranfrage.")
    draft_reply: str = Field(default="", description="Aktueller Antwortentwurf.")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score aus score_reply (0..1).")
    reasons: List[str] = Field(default_factory=list, description="Gruende fuer Review/Handover.")
    priority: str = Field(default="medium", description="low|medium|high|urgent")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Optionale Zusatzinfos als Key/Value.")


class CreateReviewTicketResponse(BaseModel):
    ticket_id: str = ""
    status: str = "open"
    created_at: str = ""
    updated_at: str = ""
    title: str = ""
    score: float = 0.0
    priority: str = "medium"
    text: str = ""


class UpdateReviewTicketRequest(BaseModel):
    ticket_id: str = Field(..., min_length=1, description="ID des bestehenden Tickets.")
    status: str = Field(default="", description="Neuer Status, z. B. in_review|approved|rejected|closed.")
    reviewer_note: str = Field(default="", description="Optionale Notiz des Reviewers.")
    assignee: str = Field(default="", description="Optionaler Bearbeitername.")
    draft_reply: str = Field(default="", description="Optional aktualisierter Antwortentwurf.")
    score: float | None = Field(default=None, ge=0.0, le=1.0, description="Optional neuer Score (0..1).")
    resolution: str = Field(default="", description="Optionale Abschlussinfo.")


class UpdateReviewTicketResponse(BaseModel):
    ticket_id: str = ""
    status: str = ""
    updated_at: str = ""
    assignee: str = ""
    notes_count: int = 0
    text: str = ""


class PolicyCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Antworttext, der geprueft werden soll.")
    policy_profile: str = Field(default="default", description="Optionales Policy-Profil.")
    strict_mode: bool = Field(default=True, description="Wenn True, werden mehr Verstosse blockierend behandelt.")


class PolicyCheckResponse(BaseModel):
    allowed: bool = False
    risk_level: str = "medium"
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    redacted_text: str = ""
    text: str = ""


__all__ = [
    "ScoreReplyRequest",
    "ScoreReplyResponse",
    "CreateReviewTicketRequest",
    "CreateReviewTicketResponse",
    "UpdateReviewTicketRequest",
    "UpdateReviewTicketResponse",
    "PolicyCheckRequest",
    "PolicyCheckResponse",
]
