from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .mail import send_mail
from .models import MailSendRequest, MailSendResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/mail/send', response_model=MailSendResponse)
    def mail_send(
        req: MailSendRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailSendResponse:
        ensure_user_dirs(s, user_id)
        result = send_mail(
            to=req.to,
            subject=req.subject,
            body=req.body,
            attachment_paths=req.attachment_paths,
            work_dir=s.user_work_dir(user_id),
            cc=req.cc,
            bcc=req.bcc,
            from_email=req.from_email,
            reply_to=req.reply_to,
            is_html=req.is_html,
        )
        return MailSendResponse(**result)

    return router
