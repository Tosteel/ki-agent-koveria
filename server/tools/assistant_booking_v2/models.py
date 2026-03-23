from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BookingAssistantV2ReviewsToolRequest(BaseModel):
    status: str = Field(default="pending")


class BookingAssistantV2StatusMeetingToolRequest(BaseModel):
    since: str = Field(default="")


class BookingAssistantV2PendingNextToolRequest(BaseModel):
    pass


class BookingAssistantV2PendingApplyToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    action: Literal["approve", "reject", "counteroffer"]
    edited_body: str = Field(default="")
    subject: str = Field(default="")
    reason: str = Field(default="")
    send_to_customer: bool = Field(default=True)


class BookingAssistantV2ReviewApproveToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    edited_body: str = Field(default="")
    subject: str = Field(default="")


class BookingAssistantV2ReviewRejectToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    edited_body: str = Field(default="")
    subject: str = Field(default="")
    send_to_customer: bool = Field(default=True)
    reason: str = Field(default="")


class BookingAssistantV2ReviewCounterofferToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    edited_body: str = Field(default="")
    subject: str = Field(default="")
