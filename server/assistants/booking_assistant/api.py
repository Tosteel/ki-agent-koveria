from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import (
    BookingAssistantApproveRequest,
    BookingAssistantCounterofferRequest,
    BookingAssistantOperatorChatRequest,
    BookingAssistantOperatorChatResponse,
    BookingAssistantPendingApplyRequest,
    BookingAssistantPendingNextResponse,
    BookingAssistantRejectRequest,
    BookingAssistantReviewActionResponse,
    BookingAssistantReviewsResponse,
    BookingAssistantRunRequest,
    BookingAssistantRunResponse,
    BookingAssistantStatusMeetingResponse,
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

    @router.get(
        "/assistants/booking-assistant/status-meeting",
        response_model=BookingAssistantStatusMeetingResponse,
        deprecated=True,
        summary="[Deprecated] Status meeting (use agent/ask tool booking_assistant_status_meeting)",
    )
    def booking_assistant_status_meeting(
        since: str = Query(default=""),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantStatusMeetingResponse:
        ensure_user_dirs(s, user_id)
        return get_status_meeting(user_id=user_id, settings=s, since=since)

    @router.get(
        "/assistants/booking-assistant/pending/next",
        response_model=BookingAssistantPendingNextResponse,
        deprecated=True,
        summary="[Deprecated] Next pending (use agent/ask tool booking_assistant_pending_next)",
    )
    def booking_assistant_pending_next(
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantPendingNextResponse:
        ensure_user_dirs(s, user_id)
        return get_pending_next(user_id=user_id, settings=s)

    @router.post(
        "/assistants/booking-assistant/pending/{review_id}/apply",
        response_model=BookingAssistantReviewActionResponse,
    )
    def booking_assistant_pending_apply(
        review_id: str,
        req: BookingAssistantPendingApplyRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantReviewActionResponse:
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
        "/assistants/booking-assistant/operator-chat",
        response_model=BookingAssistantOperatorChatResponse,
        deprecated=True,
        summary="[Deprecated] Operator chat (use agent/ask with assistant_id=booking-assistant)",
    )
    def booking_assistant_operator_chat(
        req: BookingAssistantOperatorChatRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantOperatorChatResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return operator_chat(user_id=user_id, settings=s, api_key=token, req=req)

    @router.post(
        "/assistants/booking-assistant/reviews/{review_id}/approve",
        response_model=BookingAssistantReviewActionResponse,
        deprecated=True,
        summary="[Deprecated] Direct approve (use pending/{review_id}/apply or agent/ask tool)",
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
        deprecated=True,
        summary="[Deprecated] Direct reject (use pending/{review_id}/apply or agent/ask tool)",
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
        deprecated=True,
        summary="[Deprecated] Direct counteroffer (use pending/{review_id}/apply or agent/ask tool)",
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
