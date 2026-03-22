from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class BookingAssistantRunRequest(BaseModel):
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
    # Operator can initialize/update profile through this request (dialog setup/update).
    profile_bootstrap: bool = Field(default=True)
    profile_instructions_add: List[str] = Field(default_factory=list)
    profile_rules_patch: Dict[str, Any] = Field(default_factory=dict)


class BookingAssistantRunItem(BaseModel):
    mail_id: str
    subject: str = ""
    from_email: str = ""
    decision: Literal["auto_sent", "needs_human", "skipped", "failed"] = "skipped"
    booking_decision: str = ""
    review_id: str = ""
    sent: bool = False
    score_total: float = 0.0
    reason: str = ""


class BookingAssistantRunResponse(BaseModel):
    ok: bool = True
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    lock_blocked: bool = False
    lock_reason: str = ""
    processed_count: int = 0
    sent_count: int = 0
    review_count: int = 0
    skipped_count: int = 0
    items: List[BookingAssistantRunItem] = Field(default_factory=list)


class BookingAssistantReviewItem(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str = ""
    mail_id: str
    mailbox: str = "INBOX"
    from_email: str = ""
    subject: str = ""
    mail_text: str = ""
    draft_body: str = ""
    booking_decision: str = ""
    score_total: float = 0.0
    score: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    ticket_id: str = ""
    reason: str = ""
    action_templates: Dict[str, str] = Field(default_factory=dict)
    selected_action: str = ""
    sent: bool = False
    send_result: Dict[str, Any] = Field(default_factory=dict)


class BookingAssistantReviewsResponse(BaseModel):
    ok: bool = True
    reviews: List[BookingAssistantReviewItem] = Field(default_factory=list)


class BookingAssistantApproveRequest(BaseModel):
    provider: str = Field(default="ionos")
    edited_body: str = ""
    subject: str = ""


class BookingAssistantRejectRequest(BaseModel):
    provider: str = Field(default="ionos")
    edited_body: str = ""
    subject: str = ""
    send_to_customer: bool = True
    reason: str = ""


class BookingAssistantCounterofferRequest(BaseModel):
    provider: str = Field(default="ionos")
    edited_body: str = ""
    subject: str = ""


class BookingAssistantReviewActionResponse(BaseModel):
    ok: bool = True
    review_id: str
    status: str
    sent: bool = False
    reason: str = ""


class BookingAssistantStatusMeetingResponse(BaseModel):
    ok: bool = True
    since: str = ""
    generated_at: str = ""
    total_processed: int = 0
    confirmed_count: int = 0
    rejected_count: int = 0
    pending_count: int = 0
    confirmed_items: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_items: List[Dict[str, Any]] = Field(default_factory=list)
    pending_items: List[Dict[str, Any]] = Field(default_factory=list)
    top_blockers: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    recent_activity: List[Dict[str, Any]] = Field(default_factory=list)
    text: str = ""


class BookingAssistantPendingNextResponse(BaseModel):
    ok: bool = True
    has_pending: bool = False
    review: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, str] = Field(default_factory=dict)
    text: str = ""


class BookingAssistantPendingApplyRequest(BaseModel):
    provider: str = Field(default="ionos")
    action: Literal["approve", "reject", "counteroffer"]
    edited_body: str = ""
    subject: str = ""
    reason: str = ""
    send_to_customer: bool = True


class BookingAssistantOperatorChatRequest(BaseModel):
    provider: str = Field(default="ionos")
    assistant_profile_name: str = Field(default="booking_default")
    message: str = Field(..., min_length=1)
    trace_steps: bool = Field(default=False)


class BookingAssistantOperatorChatResponse(BaseModel):
    ok: bool = True
    intent: str = ""
    action_taken: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    text: str = ""
