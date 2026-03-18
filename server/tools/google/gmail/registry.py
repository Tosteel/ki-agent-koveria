from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .gmail import (
    answer_mail,
    fetch_inbox_mails,
    fetch_unanswered_mails,
    read_mail,
    read_mail_thread,
    send_mail,
)
from .models import (
    GmailAnswerRequest,
    GmailAnswerResponse,
    GmailInboxFetchRequest,
    GmailInboxFetchResponse,
    GmailReadRequest,
    GmailReadResponse,
    GmailReadThreadRequest,
    GmailReadThreadResponse,
    GmailSendRequest,
    GmailSendResponse,
    GmailUnansweredFetchRequest,
)


def register(registry: ToolRegistry) -> None:
    def tool_gmail_send_mail(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = GmailSendRequest(**args)
        result = send_mail(
            to=req.to,
            subject=req.subject,
            body=req.body,
            cc=req.cc,
            bcc=req.bcc,
            reply_to=req.reply_to,
            from_email=req.from_email,
            is_html=req.is_html,
        )
        return GmailSendResponse(**result).model_dump()

    def tool_gmail_answer_mail(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = GmailAnswerRequest(**args)
        result = answer_mail(
            mail_id=req.mail_id,
            body=req.body,
            mailbox=req.mailbox,
            subject=req.subject,
            reply_to_all=req.reply_to_all,
            is_html=req.is_html,
        )
        return GmailAnswerResponse(**result).model_dump()

    def tool_gmail_fetch_inbox_mails(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = GmailInboxFetchRequest(**args)
        result = fetch_inbox_mails(
            limit=req.limit,
            mailbox=req.mailbox,
            unread_only=req.unread_only,
        )
        return GmailInboxFetchResponse(**result).model_dump()

    def tool_gmail_fetch_unanswered_mails(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = GmailUnansweredFetchRequest(**args)
        result = fetch_unanswered_mails(
            limit=req.limit,
            mailbox=req.mailbox,
        )
        return GmailInboxFetchResponse(**result).model_dump()

    def tool_gmail_read_mail(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = GmailReadRequest(**args)
        result = read_mail(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            include_html=req.include_html,
            max_chars=req.max_chars,
        )
        return GmailReadResponse(**result).model_dump()

    def tool_gmail_read_mail_thread(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = GmailReadThreadRequest(**args)
        result = read_mail_thread(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            max_messages=req.max_messages,
            include_html=req.include_html,
            max_chars=req.max_chars,
        )
        return GmailReadThreadResponse(**result).model_dump()

    registry.register(
        "gmail_send_mail",
        tool_gmail_send_mail,
        request_model=GmailSendRequest,
        response_model=GmailSendResponse,
    )
    registry.register(
        "gmail_answer_mail",
        tool_gmail_answer_mail,
        request_model=GmailAnswerRequest,
        response_model=GmailAnswerResponse,
    )
    registry.register(
        "gmail_fetch_inbox_mails",
        tool_gmail_fetch_inbox_mails,
        request_model=GmailInboxFetchRequest,
        response_model=GmailInboxFetchResponse,
    )
    registry.register(
        "gmail_fetch_unanswered_mails",
        tool_gmail_fetch_unanswered_mails,
        request_model=GmailUnansweredFetchRequest,
        response_model=GmailInboxFetchResponse,
    )
    registry.register(
        "gmail_read_mail",
        tool_gmail_read_mail,
        request_model=GmailReadRequest,
        response_model=GmailReadResponse,
    )
    registry.register(
        "gmail_read_mail_thread",
        tool_gmail_read_mail_thread,
        request_model=GmailReadThreadRequest,
        response_model=GmailReadThreadResponse,
    )

