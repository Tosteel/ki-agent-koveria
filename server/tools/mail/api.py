from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .mail import mail_classify, mail_read, mail_read_attachments, mail_read_thread, mail_send
from .models import (
    MailClassifyRequest,
    MailClassifyResponse,
    MailReadAttachmentsRequest,
    MailReadAttachmentsResponse,
    MailReadRequest,
    MailReadResponse,
    MailReadThreadRequest,
    MailReadThreadResponse,
    MailSendRequest,
    MailSendResponse,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/tools/mail/send', response_model=MailSendResponse)
    def mail_send(
        req: MailSendRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailSendResponse:
        ensure_user_dirs(s, user_id)
        result = mail_send(
            to=req.to,
            subject=req.subject,
            body=req.body,
            attachments=req.attachments,
            work_dir=s.user_work_dir(user_id),
            cc=req.cc,
            bcc=req.bcc,
            from_email=req.from_email,
            reply_to=req.reply_to,
            is_html=req.is_html,
        )
        return MailSendResponse(**result)

    @router.post('/tools/mail/read', response_model=MailReadResponse)
    def mail_read(
        req: MailReadRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailReadResponse:
        ensure_user_dirs(s, user_id)
        result = mail_read(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            include_html=req.include_html,
            max_chars=req.max_chars,
        )
        return MailReadResponse(**result)

    @router.post('/tools/mail/read-thread', response_model=MailReadThreadResponse)
    def mail_read_thread_route(
        req: MailReadThreadRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailReadThreadResponse:
        ensure_user_dirs(s, user_id)
        result = mail_read_thread(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            max_messages=req.max_messages,
            include_html=req.include_html,
            max_chars=req.max_chars,
        )
        return MailReadThreadResponse(**result)

    @router.post('/tools/mail/read-attachments', response_model=MailReadAttachmentsResponse)
    def mail_read_attachments_route(
        req: MailReadAttachmentsRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailReadAttachmentsResponse:
        ensure_user_dirs(s, user_id)
        result = mail_read_attachments(
            mail_id=req.mail_id,
            mailbox=req.mailbox,
            include_content=req.include_content,
            max_attachment_chars=req.max_attachment_chars,
            extract_text_pdf=req.extract_text_pdf,
        )
        return MailReadAttachmentsResponse(**result)

    @router.post('/tools/mail/classify', response_model=MailClassifyResponse)
    def mail_classify_route(
        req: MailClassifyRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailClassifyResponse:
        ensure_user_dirs(s, user_id)
        result = mail_classify(
            text=req.text,
            subject=req.subject,
            body_text=req.body_text,
            from_email=req.from_email,
        )
        return MailClassifyResponse(**result)

    return router
