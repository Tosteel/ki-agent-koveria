from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BookingAssistantReviewsToolRequest(BaseModel):
    status: str = Field(default="pending")


class BookingAssistantStatusMeetingToolRequest(BaseModel):
    since: str = Field(default="")


class BookingAssistantPendingNextToolRequest(BaseModel):
    pass


class BookingAssistantPendingApplyToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    action: Literal["approve", "reject", "counteroffer"]
    edited_body: str = Field(default="")
    subject: str = Field(default="")
    reason: str = Field(default="")
    send_to_customer: bool = Field(default=True)


class BookingAssistantReviewApproveToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    edited_body: str = Field(default="")
    subject: str = Field(default="")


class BookingAssistantReviewRejectToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    edited_body: str = Field(default="")
    subject: str = Field(default="")
    send_to_customer: bool = Field(default=True)
    reason: str = Field(default="")


class BookingAssistantReviewCounterofferToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    edited_body: str = Field(default="")
    subject: str = Field(default="")


__all__ = [
    "BookingAssistantReviewsToolRequest",
    "BookingAssistantStatusMeetingToolRequest",
    "BookingAssistantPendingNextToolRequest",
    "BookingAssistantPendingApplyToolRequest",
    "BookingAssistantReviewApproveToolRequest",
    "BookingAssistantReviewRejectToolRequest",
    "BookingAssistantReviewCounterofferToolRequest",
]
