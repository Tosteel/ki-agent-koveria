from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BookingAssistantV3StatusToolRequest(BaseModel):
    since: str = Field(default="")


class BookingAssistantV3ReviewsToolRequest(BaseModel):
    status: str = Field(default="pending")
    kind: str = Field(default="")


class BookingAssistantV3PendingNextToolRequest(BaseModel):
    queue: str = Field(default="any")


class BookingAssistantV3PendingApplyToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    action: Literal["approve", "offer", "reject", "final_confirmation", "final_rejection"]
    edited_body: str = Field(default="")
    subject: str = Field(default="")
    reason: str = Field(default="")
    send_to_customer: bool = Field(default=True)


class BookingAssistantV3SimpleActionToolRequest(BaseModel):
    review_id: str = Field(..., min_length=1)
    provider: str = Field(default="ionos")
    edited_body: str = Field(default="")
    subject: str = Field(default="")
    reason: str = Field(default="")
    send_to_customer: bool = Field(default=True)
