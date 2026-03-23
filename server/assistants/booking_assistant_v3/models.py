from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


ReviewKind = Literal["rule_review", "final_confirmation"]
ActionType = Literal["approve", "offer", "reject", "final_confirmation", "final_rejection"]


class BookingAssistantV3RunRequest(BaseModel):
    provider: str = Field(default="ionos")
    mailbox: str = Field(default="INBOX")
    limit: int = Field(default=10, ge=1, le=50)
    assistant_profile_name: str = Field(default="booking_default")
    assistant_codename: str = Field(default="")
    auto_send_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    web_sources: List[str] = Field(default_factory=list)
    web_whitelist_domains: List[str] = Field(default_factory=list)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    max_context_chars: int = Field(default=12000, ge=1000, le=50000)
    include_thread: bool = Field(default=True)
    strict_policy: bool = Field(default=True)
    trace_steps: bool = Field(default=True)
    profile_bootstrap: bool = Field(default=True)
    profile_instructions_add: List[str] = Field(default_factory=list)
    profile_rules_patch: Dict[str, Any] = Field(default_factory=dict)


class BookingAssistantV3RunItem(BaseModel):
    mail_id: str
    thread_id: str = ""
    subject: str = ""
    from_email: str = ""
    intent: str = ""
    decision: Literal["auto_sent", "needs_human", "skipped", "failed"] = "skipped"
    booking_decision: str = ""
    review_id: str = ""
    sent: bool = False
    score_total: float = 0.0
    reason: str = ""


class BookingAssistantV3RunResponse(BaseModel):
    ok: bool = True
    version: str = "3.0"
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    lock_blocked: bool = False
    lock_reason: str = ""
    processed_count: int = 0
    sent_count: int = 0
    review_count: int = 0
    skipped_count: int = 0
    items: List[BookingAssistantV3RunItem] = Field(default_factory=list)


class BookingAssistantV3ReviewItem(BaseModel):
    id: str
    kind: ReviewKind
    status: str
    created_at: str
    updated_at: str = ""
    mail_id: str
    thread_id: str = ""
    mailbox: str = "INBOX"
    from_email: str = ""
    subject: str = ""
    mail_text: str = ""
    draft_body: str = ""
    booking_decision: str = ""
    score_total: float = 0.0
    score: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    action_templates: Dict[str, str] = Field(default_factory=dict)
    required_fields_status: Dict[str, Any] = Field(default_factory=dict)
    selected_action: str = ""
    sent: bool = False
    send_result: Dict[str, Any] = Field(default_factory=dict)


class BookingAssistantV3ReviewsResponse(BaseModel):
    ok: bool = True
    version: str = "3.0"
    reviews: List[BookingAssistantV3ReviewItem] = Field(default_factory=list)


class BookingAssistantV3PendingNextResponse(BaseModel):
    ok: bool = True
    version: str = "3.0"
    has_pending: bool = False
    review: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, str] = Field(default_factory=dict)
    text: str = ""


class BookingAssistantV3PendingApplyRequest(BaseModel):
    provider: str = Field(default="ionos")
    action: ActionType
    edited_body: str = ""
    subject: str = ""
    reason: str = ""
    send_to_customer: bool = True


class BookingAssistantV3ReviewActionResponse(BaseModel):
    ok: bool = True
    version: str = "3.0"
    review_id: str
    status: str
    sent: bool = False
    reason: str = ""


class BookingAssistantV3StatusResponse(BaseModel):
    ok: bool = True
    version: str = "3.0"
    since: str = ""
    generated_at: str = ""
    incoming_count: int = 0
    pending_rule_review_count: int = 0
    pending_final_confirmation_count: int = 0
    rejected_count: int = 0
    auto_sent_count: int = 0
    top_blockers: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    text: str = ""


class BookingAssistantV3OperatorChatRequest(BaseModel):
    provider: str = Field(default="ionos")
    assistant_profile_name: str = Field(default="booking_default")
    message: str = Field(..., min_length=1)


class BookingAssistantV3OperatorChatResponse(BaseModel):
    ok: bool = True
    version: str = "3.0"
    intent: str = ""
    action_taken: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    text: str = ""


class BookingAssistantV3SimpleActionRequest(BaseModel):
    provider: str = Field(default="ionos")
    edited_body: str = ""
    subject: str = ""
    reason: str = ""
    send_to_customer: bool = True
