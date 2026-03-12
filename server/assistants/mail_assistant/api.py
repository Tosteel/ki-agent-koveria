from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import (
    MailAssistantApproveRequest,
    MailAssistantRejectRequest,
    MailAssistantReviewActionResponse,
    MailAssistantReviewsResponse,
    MailAssistantRunRequest,
    MailAssistantRunResponse,
)
from .service import approve_review, get_reviews, reject_review, run_once

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/assistants/mail-assistant/run-once", response_model=MailAssistantRunResponse)
    def mail_assistant_run_once(
        req: MailAssistantRunRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> MailAssistantRunResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return run_once(user_id=user_id, settings=s, api_key=token, req=req)

    @router.get("/assistants/mail-assistant/reviews", response_model=MailAssistantReviewsResponse)
    def mail_assistant_reviews(
        status: str = Query(default="pending"),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailAssistantReviewsResponse:
        ensure_user_dirs(s, user_id)
        return get_reviews(user_id=user_id, settings=s, status=status)

    @router.post(
        "/assistants/mail-assistant/reviews/{review_id}/approve",
        response_model=MailAssistantReviewActionResponse,
    )
    def mail_assistant_review_approve(
        review_id: str,
        req: MailAssistantApproveRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> MailAssistantReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return approve_review(
            user_id=user_id,
            settings=s,
            api_key=token,
            review_id=review_id,
            req=req,
        )

    @router.post(
        "/assistants/mail-assistant/reviews/{review_id}/reject",
        response_model=MailAssistantReviewActionResponse,
    )
    def mail_assistant_review_reject(
        review_id: str,
        req: MailAssistantRejectRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailAssistantReviewActionResponse:
        ensure_user_dirs(s, user_id)
        return reject_review(user_id=user_id, settings=s, review_id=review_id, req=req)

    return router

