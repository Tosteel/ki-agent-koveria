from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class MailSendRequest(BaseModel):
    to: List[str] = Field(default_factory=list)
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    attachment_paths: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    bcc: List[str] = Field(default_factory=list)
    from_email: str = ""
    reply_to: str = ""
    is_html: bool = False


class MailSendResponse(BaseModel):
    sent: bool = True
    message_id: str = ""
    recipients: List[str] = Field(default_factory=list)
    subject: str = ""
    attachments: List[str] = Field(default_factory=list)


__all__ = ["MailSendRequest", "MailSendResponse"]
