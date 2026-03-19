from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .mail import (
    mail_answer,
    mail_classify,
    mail_compose_clarification,
    mail_fetch_inbox,
    mail_fetch_unanswered,
    mail_read,
    mail_read_attachments,
    mail_read_thread,
    mail_send,
)
from .models import (
    MailAnswerRequest,
    MailAnswerResponse,
    MailClassifyRequest,
    MailClassifyResponse,
    MailComposeClarificationRequest,
    MailComposeClarificationResponse,
    MailInboxFetchRequest,
    MailInboxFetchResponse,
    MailReadAttachmentsRequest,
    MailReadAttachmentsResponse,
    MailReadRequest,
    MailReadResponse,
    MailReadThreadRequest,
    MailReadThreadResponse,
    MailUnansweredFetchRequest,
    MailSendRequest,
    MailSendResponse,
)


def register(registry: ToolRegistry) -> None:
    def tool_send_mail(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailSendRequest(**args)
        result = mail_send(
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
        result = mail_fetch_inbox(
            limit=req.limit,
            mailbox=req.mailbox,
            unread_only=req.unread_only,
        )
        return MailInboxFetchResponse(**result).model_dump()

    def tool_fetch_unanswered_mails(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailUnansweredFetchRequest(**args)
        result = mail_fetch_unanswered(
            limit=req.limit,
            mailbox=req.mailbox,
        )
        return MailInboxFetchResponse(**result).model_dump()

    def tool_answer_mail(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailAnswerRequest(**args)
        result = mail_answer(
            mail_id=req.mail_id,
            body=req.body,
            mailbox=req.mailbox,
            subject=req.subject,
            reply_to_all=req.reply_to_all,
            is_html=req.is_html,
        )
        return MailAnswerResponse(**result).model_dump()

    def tool_read_mail(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailReadRequest(**args)
        result = mail_read(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            include_html=req.include_html,
            max_chars=req.max_chars,
        )
        return MailReadResponse(**result).model_dump()

    def tool_read_mail_thread(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailReadThreadRequest(**args)
        result = mail_read_thread(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            max_messages=req.max_messages,
            include_html=req.include_html,
            max_chars=req.max_chars,
        )
        return MailReadThreadResponse(**result).model_dump()

    def tool_read_mail_attachments(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailReadAttachmentsRequest(**args)
        result = mail_read_attachments(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            include_content=req.include_content,
            max_attachment_chars=req.max_attachment_chars,
            extract_text_pdf=req.extract_text_pdf,
        )
        return MailReadAttachmentsResponse(**result).model_dump()

    def tool_classify_mail(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailClassifyRequest(**args)
        result = mail_classify(
            text=req.text,
            subject=req.subject,
            body_text=req.body_text,
            from_email=req.from_email,
        )
        return MailClassifyResponse(**result).model_dump()

    def tool_compose_clarification_mail(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailComposeClarificationRequest(**args)
        result = mail_compose_clarification(
            missing_fields=req.missing_fields,
            known_facts=req.known_facts,
            salutation=req.salutation,
            closing=req.closing,
        )
        return MailComposeClarificationResponse(**result).model_dump()

    registry.register(
        "mail_send",
        tool_send_mail,
        request_model=MailSendRequest,
        response_model=MailSendResponse,
    )
    registry.register(
        "mail_fetch_inbox",
        tool_fetch_inbox_mails,
        request_model=MailInboxFetchRequest,
        response_model=MailInboxFetchResponse,
    )
    registry.register(
        "mail_fetch_unanswered",
        tool_fetch_unanswered_mails,
        request_model=MailUnansweredFetchRequest,
        response_model=MailInboxFetchResponse,
    )
    registry.register(
        "mail_answer",
        tool_answer_mail,
        request_model=MailAnswerRequest,
        response_model=MailAnswerResponse,
    )
    registry.register(
        "mail_read",
        tool_read_mail,
        request_model=MailReadRequest,
        response_model=MailReadResponse,
    )
    registry.register(
        "mail_read_thread",
        tool_read_mail_thread,
        request_model=MailReadThreadRequest,
        response_model=MailReadThreadResponse,
    )
    registry.register(
        "mail_read_attachments",
        tool_read_mail_attachments,
        request_model=MailReadAttachmentsRequest,
        response_model=MailReadAttachmentsResponse,
    )
    registry.register(
        "mail_classify",
        tool_classify_mail,
        request_model=MailClassifyRequest,
        response_model=MailClassifyResponse,
    )
    registry.register(
        "mail_compose_clarification",
        tool_compose_clarification_mail,
        request_model=MailComposeClarificationRequest,
        response_model=MailComposeClarificationResponse,
    )
