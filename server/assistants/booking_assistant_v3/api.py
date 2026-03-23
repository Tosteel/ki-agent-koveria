from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import (
    BookingAssistantV3OperatorChatRequest,
    BookingAssistantV3OperatorChatResponse,
    BookingAssistantV3PendingApplyRequest,
    BookingAssistantV3PendingNextResponse,
    BookingAssistantV3ReviewActionResponse,
    BookingAssistantV3ReviewsResponse,
    BookingAssistantV3RunRequest,
    BookingAssistantV3RunResponse,
    BookingAssistantV3SimpleActionRequest,
    BookingAssistantV3StatusResponse,
)
from .service import (
    apply_pending_action,
    get_pending_next,
    get_reviews,
    get_status,
    operator_chat,
    run_once,
)

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/assistants/booking-assistant-v3/run-once", response_model=BookingAssistantV3RunResponse)
    def booking_assistant_v3_run_once(
        req: BookingAssistantV3RunRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV3RunResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return run_once(user_id=user_id, settings=s, api_key=token, req=req)

    @router.get("/assistants/booking-assistant-v3/status", response_model=BookingAssistantV3StatusResponse)
    def booking_assistant_v3_status(
        since: str = Query(default=""),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantV3StatusResponse:
        ensure_user_dirs(s, user_id)
        return get_status(user_id=user_id, settings=s, since=since)

    @router.get("/assistants/booking-assistant-v3/reviews", response_model=BookingAssistantV3ReviewsResponse)
    def booking_assistant_v3_reviews(
        status: str = Query(default="pending"),
        kind: str = Query(default=""),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantV3ReviewsResponse:
        ensure_user_dirs(s, user_id)
        return get_reviews(user_id=user_id, settings=s, status=status, kind=kind)

    @router.get("/assistants/booking-assistant-v3/pending/next", response_model=BookingAssistantV3PendingNextResponse)
    def booking_assistant_v3_pending_next(
        queue: str = Query(default="any"),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingAssistantV3PendingNextResponse:
        ensure_user_dirs(s, user_id)
        return get_pending_next(user_id=user_id, settings=s, queue=queue)

    @router.post("/assistants/booking-assistant-v3/pending/{review_id}/apply", response_model=BookingAssistantV3ReviewActionResponse)
    def booking_assistant_v3_pending_apply(
        review_id: str,
        req: BookingAssistantV3PendingApplyRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV3ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return apply_pending_action(user_id=user_id, settings=s, api_key=token, review_id=review_id, req=req)

    @router.post("/assistants/booking-assistant-v3/operator-chat", response_model=BookingAssistantV3OperatorChatResponse)
    def booking_assistant_v3_operator_chat(
        req: BookingAssistantV3OperatorChatRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV3OperatorChatResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        return operator_chat(user_id=user_id, settings=s, api_key=token, req=req)

    @router.post(
        "/assistants/booking-assistant-v3/reviews/{review_id}/approve",
        response_model=BookingAssistantV3ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v3_review_approve(
        review_id: str,
        req: BookingAssistantV3SimpleActionRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV3ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        # Deprecated alias: unify all write actions through pending/{review_id}/apply.
        return apply_pending_action(
            user_id=user_id,
            settings=s,
            api_key=token,
            review_id=review_id,
            req=BookingAssistantV3PendingApplyRequest(
                provider=req.provider,
                action="approve",
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )

    @router.post(
        "/assistants/booking-assistant-v3/reviews/{review_id}/offer",
        response_model=BookingAssistantV3ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v3_review_offer(
        review_id: str,
        req: BookingAssistantV3SimpleActionRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV3ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        # Deprecated alias: unify all write actions through pending/{review_id}/apply.
        return apply_pending_action(
            user_id=user_id,
            settings=s,
            api_key=token,
            review_id=review_id,
            req=BookingAssistantV3PendingApplyRequest(
                provider=req.provider,
                action="offer",
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )

    @router.post(
        "/assistants/booking-assistant-v3/reviews/{review_id}/reject",
        response_model=BookingAssistantV3ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v3_review_reject(
        review_id: str,
        req: BookingAssistantV3SimpleActionRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV3ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        # Deprecated alias: unify all write actions through pending/{review_id}/apply.
        return apply_pending_action(
            user_id=user_id,
            settings=s,
            api_key=token,
            review_id=review_id,
            req=BookingAssistantV3PendingApplyRequest(
                provider=req.provider,
                action="reject",
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )

    @router.post(
        "/assistants/booking-assistant-v3/reviews/{review_id}/final-confirmation",
        response_model=BookingAssistantV3ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v3_review_final_confirmation(
        review_id: str,
        req: BookingAssistantV3SimpleActionRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV3ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        # Deprecated alias: unify all write actions through pending/{review_id}/apply.
        return apply_pending_action(
            user_id=user_id,
            settings=s,
            api_key=token,
            review_id=review_id,
            req=BookingAssistantV3PendingApplyRequest(
                provider=req.provider,
                action="final_confirmation",
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )

    @router.post(
        "/assistants/booking-assistant-v3/reviews/{review_id}/final-rejection",
        response_model=BookingAssistantV3ReviewActionResponse,
        deprecated=True,
    )
    def booking_assistant_v3_review_final_rejection(
        review_id: str,
        req: BookingAssistantV3SimpleActionRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> BookingAssistantV3ReviewActionResponse:
        ensure_user_dirs(s, user_id)
        token = credentials.credentials if credentials is not None else ""
        # Deprecated alias: unify all write actions through pending/{review_id}/apply.
        return apply_pending_action(
            user_id=user_id,
            settings=s,
            api_key=token,
            review_id=review_id,
            req=BookingAssistantV3PendingApplyRequest(
                provider=req.provider,
                action="final_rejection",
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )

    return router
