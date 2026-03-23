from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import (
    BookingAssistantV2ApproveRequest,
    BookingAssistantV2CounterofferRequest,
    BookingAssistantV2OperatorChatRequest,
    BookingAssistantV2OperatorChatResponse,
    BookingAssistantV2PendingApplyRequest,
    BookingAssistantV2PendingNextResponse,
    BookingAssistantV2RejectRequest,
    BookingAssistantV2ReviewActionResponse,
    BookingAssistantV2ReviewsResponse,
    BookingAssistantV2RunRequest,
    BookingAssistantV2RunResponse,
    BookingAssistantV2StatusMeetingResponse,
)
from .service import (
    apply_pending_action,
    approve_review,
    counteroffer_review,
    get_pending_next,
    get_reviews,
    get_status_meeting,
    operator_chat,
    reject_review,
    run_once,
)

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/assistants/booking-assistant-v2/run-once",
        response_model=BookingAssistantV2RunResponse,
        deprecated=True,
    )
    def booking_assistant_v2_run_once(
        req: BookingAssistantV2RunRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV2RunResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return run_once(user_id=user_id, settings=s, api_key=token, req=req)

    @router.get(
        "/assistants/booking-assistant-v2/reviews",
        response_model=BookingAssistantV2ReviewsResponse,
        deprecated=True,
    )
    def booking_assistant_v2_reviews(
        status: str = Query(default="pending"),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantV2ReviewsResponse:
        ensure_user_dirs(s, user_id)
        return get_reviews(user_id=user_id, settings=s, status=status)

    @router.get(
        "/assistants/booking-assistant-v2/status-meeting",
        response_model=BookingAssistantV2StatusMeetingResponse,
        deprecated=True,
        summary="Status meeting for Booking Assistant v2",
    )
    def booking_assistant_v2_status_meeting(
        since: str = Query(default=""),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantV2StatusMeetingResponse:
        ensure_user_dirs(s, user_id)
        return get_status_meeting(user_id=user_id, settings=s, since=since)

    @router.get(
        "/assistants/booking-assistant-v2/pending/next",
        response_model=BookingAssistantV2PendingNextResponse,
        deprecated=True,
    )
    def booking_assistant_v2_pending_next(
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantV2PendingNextResponse:
        ensure_user_dirs(s, user_id)
        return get_pending_next(user_id=user_id, settings=s)

    @router.post(
        "/assistants/booking-assistant-v2/pending/{review_id}/apply",
        response_model=BookingAssistantV2ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v2_pending_apply(
        review_id: str,
        req: BookingAssistantV2PendingApplyRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV2ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return apply_pending_action(
            user_id=user_id,
            settings=s,
            api_key=token,
            review_id=review_id,
            req=req,
        )

    @router.post(
        "/assistants/booking-assistant-v2/operator-chat",
        response_model=BookingAssistantV2OperatorChatResponse,
        deprecated=True,
    )
    def booking_assistant_v2_operator_chat(
        req: BookingAssistantV2OperatorChatRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV2OperatorChatResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return operator_chat(user_id=user_id, settings=s, api_key=token, req=req)

    @router.post(
        "/assistants/booking-assistant-v2/reviews/{review_id}/approve",
        response_model=BookingAssistantV2ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v2_review_approve(
        review_id: str,
        req: BookingAssistantV2ApproveRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV2ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return approve_review(user_id=user_id, settings=s, api_key=token, review_id=review_id, req=req)

    @router.post(
        "/assistants/booking-assistant-v2/reviews/{review_id}/reject",
        response_model=BookingAssistantV2ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v2_review_reject(
        review_id: str,
        req: BookingAssistantV2RejectRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV2ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return reject_review(user_id=user_id, settings=s, api_key=token, review_id=review_id, req=req)

    @router.post(
        "/assistants/booking-assistant-v2/reviews/{review_id}/counteroffer",
        response_model=BookingAssistantV2ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v2_review_counteroffer(
        review_id: str,
        req: BookingAssistantV2CounterofferRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV2ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return counteroffer_review(user_id=user_id, settings=s, api_key=token, review_id=review_id, req=req)

    return router
