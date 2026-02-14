from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .mail import send_mail
from .models import MailSendRequest, MailSendResponse


def register(registry: ToolRegistry) -> None:
    def tool_send_mail(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = MailSendRequest(**args)
        result = send_mail(
            to=req.to,
            subject=req.subject,
            body=req.body,
            attachment_paths=req.attachment_paths,
            work_dir=ctx.settings.user_work_dir(ctx.user_id),
            cc=req.cc,
            bcc=req.bcc,
            from_email=req.from_email,
            reply_to=req.reply_to,
            is_html=req.is_html,
        )
        return MailSendResponse(**result).model_dump()

    registry.register("send_mail", tool_send_mail, request_model=MailSendRequest)
