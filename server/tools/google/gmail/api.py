from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

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


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/tools/google/gmail/send", response_model=GmailSendResponse)
    def gmail_send_route(
        req: GmailSendRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> GmailSendResponse:
        ensure_user_dirs(s, user_id)
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
        return GmailSendResponse(**result)

    @router.post("/tools/google/gmail/answer", response_model=GmailAnswerResponse)
    def gmail_answer_route(
        req: GmailAnswerRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> GmailAnswerResponse:
        ensure_user_dirs(s, user_id)
        result = answer_mail(
            mail_id=req.mail_id,
            body=req.body,
            mailbox=req.mailbox,
            subject=req.subject,
            reply_to_all=req.reply_to_all,
            is_html=req.is_html,
        )
        return GmailAnswerResponse(**result)

    @router.post("/tools/google/gmail/fetch-inbox", response_model=GmailInboxFetchResponse)
    def gmail_fetch_inbox_route(
        req: GmailInboxFetchRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> GmailInboxFetchResponse:
        ensure_user_dirs(s, user_id)
        result = fetch_inbox_mails(
            limit=req.limit,
            mailbox=req.mailbox,
            unread_only=req.unread_only,
        )
        return GmailInboxFetchResponse(**result)

    @router.post("/tools/google/gmail/fetch-unanswered", response_model=GmailInboxFetchResponse)
    def gmail_fetch_unanswered_route(
        req: GmailUnansweredFetchRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> GmailInboxFetchResponse:
        ensure_user_dirs(s, user_id)
        result = fetch_unanswered_mails(
            limit=req.limit,
            mailbox=req.mailbox,
        )
        return GmailInboxFetchResponse(**result)

    @router.post("/tools/google/gmail/read", response_model=GmailReadResponse)
    def gmail_read_route(
        req: GmailReadRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> GmailReadResponse:
        ensure_user_dirs(s, user_id)
        result = read_mail(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            include_html=req.include_html,
            max_chars=req.max_chars,
        )
        return GmailReadResponse(**result)

    @router.post("/tools/google/gmail/read-thread", response_model=GmailReadThreadResponse)
    def gmail_read_thread_route(
        req: GmailReadThreadRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> GmailReadThreadResponse:
        ensure_user_dirs(s, user_id)
        result = read_mail_thread(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            max_messages=req.max_messages,
            include_html=req.include_html,
            max_chars=req.max_chars,
        )
        return GmailReadThreadResponse(**result)

    return router
