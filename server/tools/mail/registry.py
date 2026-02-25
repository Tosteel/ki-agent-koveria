from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .mail import answer_mail, fetch_inbox_mails, fetch_unanswered_mails, send_mail
from .models import (
    MailAnswerRequest,
    MailAnswerResponse,
    MailInboxFetchRequest,
    MailInboxFetchResponse,
    MailUnansweredFetchRequest,
    MailSendRequest,
    MailSendResponse,
)


def register(registry: ToolRegistry) -> None:
    def tool_send_mail(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailSendRequest(**args)
        result = send_mail(
            to=req.to,
            subject=req.subject,
            body=req.body,
            attachments=req.attachments,
            work_dir=ctx.settings.user_work_dir(ctx.user_id),
            cc=req.cc,
            bcc=req.bcc,
            from_email=req.from_email,
            reply_to=req.reply_to,
            is_html=req.is_html,
        )
        return MailSendResponse(**result).model_dump()

    def tool_fetch_inbox_mails(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailInboxFetchRequest(**args)
        result = fetch_inbox_mails(
            limit=req.limit,
            mailbox=req.mailbox,
            unread_only=req.unread_only,
        )
        return MailInboxFetchResponse(**result).model_dump()

    def tool_fetch_unanswered_mails(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailUnansweredFetchRequest(**args)
        result = fetch_unanswered_mails(
            limit=req.limit,
            mailbox=req.mailbox,
        )
        return MailInboxFetchResponse(**result).model_dump()

    def tool_answer_mail(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailAnswerRequest(**args)
        result = answer_mail(
            mail_id=req.mail_id,
            body=req.body,
            mailbox=req.mailbox,
            subject=req.subject,
            reply_to_all=req.reply_to_all,
            is_html=req.is_html,
        )
        return MailAnswerResponse(**result).model_dump()

    registry.register(
        "send_mail",
        tool_send_mail,
        request_model=MailSendRequest,
        response_model=MailSendResponse,
    )
    registry.register(
        "fetch_inbox_mails",
        tool_fetch_inbox_mails,
        request_model=MailInboxFetchRequest,
        response_model=MailInboxFetchResponse,
    )
    registry.register(
        "fetch_unanswered_mails",
        tool_fetch_unanswered_mails,
        request_model=MailUnansweredFetchRequest,
        response_model=MailInboxFetchResponse,
    )
    registry.register(
        "answer_mail",
        tool_answer_mail,
        request_model=MailAnswerRequest,
        response_model=MailAnswerResponse,
    )
