from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class MailAssistantRunRequest(BaseModel):
    provider: str = Field(default="ionos")
    mailbox: str = Field(default="INBOX")
    limit: int = Field(default=10, ge=1, le=50)
    auto_send_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    web_sources: List[str] = Field(default_factory=list)
    web_whitelist_domains: List[str] = Field(default_factory=list)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    max_context_chars: int = Field(default=10000, ge=1000, le=50000)
    include_thread: bool = Field(default=True)
    include_attachments: bool = Field(default=True)
    strict_policy: bool = Field(default=True)
    trace_steps: bool = Field(default=True, description="Wenn True, werden alle Tool-Teilschritte im Terminal geloggt.")


class MailAssistantRunItem(BaseModel):
    mail_id: str
    subject: str = ""
    from_email: str = ""
    decision: Literal["auto_sent", "needs_human", "skipped", "failed"] = "skipped"
    score_total: float = 0.0
    risk: str = ""
    review_id: str = ""
    sent: bool = False
    reason: str = ""


class MailAssistantRunResponse(BaseModel):
    ok: bool = True
    processed_count: int = 0
    sent_count: int = 0
    review_count: int = 0
    skipped_count: int = 0
    items: List[MailAssistantRunItem] = Field(default_factory=list)


class MailAssistantReviewItem(BaseModel):
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
    score_total: float = 0.0
    score: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    ticket_id: str = ""
    reason: str = ""
    sent: bool = False
    send_result: Dict[str, Any] = Field(default_factory=dict)


class MailAssistantReviewsResponse(BaseModel):
    ok: bool = True
    reviews: List[MailAssistantReviewItem] = Field(default_factory=list)


class MailAssistantApproveRequest(BaseModel):
    provider: str = Field(default="ionos")
    edited_body: str = ""
    subject: str = ""


class MailAssistantRejectRequest(BaseModel):
    reason: str = ""


class MailAssistantReviewActionResponse(BaseModel):
    ok: bool = True
    review_id: str
    status: str
    sent: bool = False
    reason: str = ""
