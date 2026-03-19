from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class GmailSendRequest(BaseModel):
    to: List[str] = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    cc: List[str] = Field(default_factory=list)
    bcc: List[str] = Field(default_factory=list)
    reply_to: str = ""
    from_email: str = ""
    is_html: bool = False


class GmailSendResponse(BaseModel):
    sent: bool = True
    message_id: str = ""
    thread_id: str = ""
    to: List[str] = Field(default_factory=list)
    subject: str = ""


class GmailAnswerRequest(BaseModel):
    mail_id: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    mailbox: str = Field(default="INBOX")
    subject: str = ""
    reply_to_all: bool = Field(default=False)
    is_html: bool = Field(default=False)


class GmailAnswerResponse(BaseModel):
    sent: bool = True
    message_id: str = ""
    thread_id: str = ""
    to: List[str] = Field(default_factory=list)
    subject: str = ""


class GmailInboxFetchRequest(BaseModel):
    limit: int = Field(10, ge=1, le=50)
    mailbox: str = Field(default="INBOX")
    unread_only: bool = Field(default=False)


class GmailUnansweredFetchRequest(BaseModel):
    limit: int = Field(10, ge=1, le=50)
    mailbox: str = Field(default="INBOX")


class GmailReadRequest(BaseModel):
    mail_id: str = Field(..., min_length=1)
    mailbox: str = Field(default="INBOX")
    include_html: bool = Field(default=False)
    max_chars: int = Field(default=20000, ge=500, le=200000)


class GmailReadThreadRequest(BaseModel):
    mail_id: str = Field(..., min_length=1)
    mailbox: str = Field(default="INBOX")
    max_messages: int = Field(default=20, ge=1, le=100)
    include_html: bool = Field(default=False)
    max_chars: int = Field(default=8000, ge=500, le=200000)


class GmailInboxItem(BaseModel):
    uid: str = ""
    from_email: str = ""
    subject: str = ""
    date: str = ""
    snippet: str = ""


class GmailInboxFetchResponse(BaseModel):
    mailbox: str = "INBOX"
    count: int = 0
    mail_id: str = ""
    emails: List[GmailInboxItem] = Field(default_factory=list)
    text: str = ""


class GmailReadResponse(BaseModel):
    mailbox: str = "INBOX"
    mail_id: str = ""
    thread_id: str = ""
    from_email: str = ""
    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    subject: str = ""
    date: str = ""
    message_id: str = ""
    in_reply_to: str = ""
    references: str = ""
    has_attachments: bool = False
    attachment_names: List[str] = Field(default_factory=list)
    body_text: str = ""
    body_html: str = ""
    text: str = ""


class GmailThreadItem(BaseModel):
    mail_id: str = ""
    thread_id: str = ""
    from_email: str = ""
    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    subject: str = ""
    date: str = ""
    message_id: str = ""
    in_reply_to: str = ""
    references: str = ""
    body_text: str = ""
    body_html: str = ""
    text: str = ""


class GmailReadThreadResponse(BaseModel):
    mailbox: str = "INBOX"
    mail_id: str = ""
    thread_id: str = ""
    count: int = 0
    messages: List[GmailThreadItem] = Field(default_factory=list)
    text: str = ""
