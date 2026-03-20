from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import (
    BookingAssistantApproveRequest,
    BookingAssistantCounterofferRequest,
    BookingAssistantRejectRequest,
    BookingAssistantReviewActionResponse,
    BookingAssistantReviewsResponse,
    BookingAssistantRunRequest,
    BookingAssistantRunResponse,
)
from .service import approve_review, counteroffer_review, get_reviews, reject_review, run_once

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/assistants/booking-assistant/run-once", response_model=BookingAssistantRunResponse)
    def booking_assistant_run_once(
        req: BookingAssistantRunRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantRunResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return run_once(user_id=user_id, settings=s, api_key=token, req=req)

    @router.get("/assistants/booking-assistant/reviews", response_model=BookingAssistantReviewsResponse)
    def booking_assistant_reviews(
        status: str = Query(default="pending"),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantReviewsResponse:
        ensure_user_dirs(s, user_id)
        return get_reviews(user_id=user_id, settings=s, status=status)

    @router.post(
        "/assistants/booking-assistant/reviews/{review_id}/approve",
        response_model=BookingAssistantReviewActionResponse,
    )
    def booking_assistant_review_approve(
        review_id: str,
        req: BookingAssistantApproveRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return approve_review(user_id=user_id, settings=s, api_key=token, review_id=review_id, req=req)

    @router.post(
        "/assistants/booking-assistant/reviews/{review_id}/reject",
        response_model=BookingAssistantReviewActionResponse,
    )
    def booking_assistant_review_reject(
        review_id: str,
        req: BookingAssistantRejectRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return reject_review(user_id=user_id, settings=s, api_key=token, review_id=review_id, req=req)

    @router.post(
        "/assistants/booking-assistant/reviews/{review_id}/counteroffer",
        response_model=BookingAssistantReviewActionResponse,
    )
    def booking_assistant_review_counteroffer(
        review_id: str,
        req: BookingAssistantCounterofferRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return counteroffer_review(user_id=user_id, settings=s, api_key=token, review_id=review_id, req=req)

    return router
